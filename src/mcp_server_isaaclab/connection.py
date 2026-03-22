"""SSH tunnel + HTTP client manager for communicating with remote Isaac Lab instance."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import asyncssh
import httpx

logger = logging.getLogger(__name__)

REMOTE_AGENT_PORT = 8421


@dataclass
class BrevConnection:
    """Manages SSH tunnel and HTTP communication with a Brev GPU instance."""

    host: str | None = None
    user: str = "ubuntu"
    key_path: str | None = None
    ssh_port: int = 22
    local_port: int = REMOTE_AGENT_PORT
    remote_port: int = REMOTE_AGENT_PORT

    _ssh_conn: asyncssh.SSHClientConnection | None = field(default=None, repr=False)
    _tunnel_listener: Any = field(default=None, repr=False)
    _http: httpx.AsyncClient | None = field(default=None, repr=False)

    @property
    def connected(self) -> bool:
        return self._ssh_conn is not None and self._http is not None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.local_port}"

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._http

    async def connect(
        self,
        host: str | None = None,
        user: str | None = None,
        key_path: str | None = None,
    ) -> dict:
        """Establish SSH tunnel to Brev instance and verify remote agent is running."""
        if host:
            self.host = host
        if user:
            self.user = user
        if key_path:
            self.key_path = key_path

        if not self.host:
            raise ValueError("No host specified. Provide a Brev instance IP or hostname.")

        # Close any existing connection
        await self.disconnect()

        logger.info("Connecting to %s@%s ...", self.user, self.host)

        # Build SSH connection options
        connect_kwargs: dict[str, Any] = {
            "host": self.host,
            "port": self.ssh_port,
            "username": self.user,
            "known_hosts": None,  # Accept unknown hosts for cloud instances
        }
        if self.key_path:
            connect_kwargs["client_keys"] = [self.key_path]

        self._ssh_conn = await asyncssh.connect(**connect_kwargs)

        # Forward local port → remote agent port
        self._tunnel_listener = await self._ssh_conn.forward_local_port(
            listen_host="127.0.0.1",
            listen_port=self.local_port,
            dest_host="127.0.0.1",
            dest_port=self.remote_port,
        )
        logger.info(
            "SSH tunnel established: localhost:%d → %s:%d",
            self.local_port,
            self.host,
            self.remote_port,
        )

        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(connect=10, read=300, write=30, pool=10),
        )

        # Health check
        try:
            resp = await self._http.get("/health")
            resp.raise_for_status()
            info = resp.json()
            logger.info("Remote agent healthy: %s", info)
            return {"status": "connected", "host": self.host, "agent_info": info}
        except Exception as e:
            logger.warning("Remote agent not responding: %s", e)
            return {
                "status": "tunnel_up_agent_unreachable",
                "host": self.host,
                "error": str(e),
                "hint": "Run `isaaclab-remote-agent` on the Brev instance, or use deploy script.",
            }

    async def disconnect(self) -> dict:
        """Tear down SSH tunnel and HTTP client."""
        if self._http:
            await self._http.aclose()
            self._http = None
        if self._tunnel_listener:
            self._tunnel_listener.close()
            self._tunnel_listener = None
        if self._ssh_conn:
            self._ssh_conn.close()
            await asyncio.sleep(0.1)
            self._ssh_conn = None
        return {"status": "disconnected"}

    async def request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> dict:
        """Make an HTTP request to the remote agent through the SSH tunnel."""
        resp = await self.http.request(method, path, **kwargs)
        resp.raise_for_status()
        return resp.json()

    async def get(self, path: str, **kwargs) -> dict:
        return await self.request("GET", path, **kwargs)

    async def post(self, path: str, **kwargs) -> dict:
        return await self.request("POST", path, **kwargs)

    async def delete(self, path: str, **kwargs) -> dict:
        return await self.request("DELETE", path, **kwargs)


# Global singleton
connection = BrevConnection()
