"""AndroidDriver — ADB-based device driver (Remote via PC Agent).

Wraps WebSocket client to communicate with the Healing Server
and pass commands to the PC Agent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
import os
import base64
from typing import Any, Dict, List, Optional

import websockets

from droidrun.tools.driver.base import DeviceDriver

logger = logging.getLogger("droidrun")

HEALING_SERVER_WS_URL = os.environ.get("HEALING_SERVER_WS_URL", "ws://localhost:8765/ws/agent/control")


class AndroidDriver(DeviceDriver):
    """Remote Android device I/O via PC Agent over WebSocket."""

    supported = {
        "tap",
        "swipe",
        "input_text",
        "press_key",
        "start_app",
        "screenshot",
        "get_ui_tree",
        "get_date",
        "get_apps",
        "list_packages",
        "install_app",
        "drag",
    }

    def __init__(
        self,
        serial: str | None = None,
        use_tcp: bool = False,
        remote_tcp_port: int = 8080, 
    ) -> None:
        self._serial = serial
        self._use_tcp = use_tcp
        self._remote_tcp_port = remote_tcp_port
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._connected = False
        
        self._agent_id = os.environ.get("AGENT_ID", "default_agent") 
        self._pending_results: Dict[str, asyncio.Future] = {}
        self._listen_task = None

    # -- lifecycle -----------------------------------------------------------

    async def connect(self) -> None:
        if self._connected:
            return

        logger.info(f"Connecting to Healing Server at {HEALING_SERVER_WS_URL} for device {self._serial}")
        self._ws = await websockets.connect(HEALING_SERVER_WS_URL, ping_interval=None)  
        
        self._listen_task = asyncio.create_task(self._listen_loop())
        self._connected = True

    async def _listen_loop(self):
        try:
            async for message in self._ws:
                data = json.loads(message)
                if data.get("type") == "command_result":
                    cmd_id = data.get("command_id")
                    if cmd_id in self._pending_results:
                        future = self._pending_results[cmd_id]
                        if not future.done():
                            future.set_result(data)
        except Exception as e:
            logger.error(f"WebSocket listener error: {e}")
            self._connected = False

    async def _send_command_to_agent(self, cmd_type: str, payload: dict) -> Any:
        await self.ensure_connected()
        cmd_id = str(uuid.uuid4())
        payload["type"] = cmd_type
        payload["command_id"] = cmd_id
        payload["device_id"] = self._serial
        request = {
            "type": "forward_to_agent",
            "agent_id": self._agent_id,
            "payload": payload
        }
        
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending_results[cmd_id] = future
        
        await self._ws.send(json.dumps(request))
        
        try:
            result = await asyncio.wait_for(future, timeout=30.0)
            return result.get("data", {})
        finally:
            self._pending_results.pop(cmd_id, None)

    async def ensure_connected(self) -> None:
        if not self._connected:
            await self.connect()

    # -- input actions -------------------------------------------------------

    async def tap(self, x: int, y: int) -> None:
        await self._send_command_to_agent("tap_absolute", {"x": x, "y": y})

    async def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: float = 1000,
    ) -> None:
        await self._send_command_to_agent("swipe", {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "duration": int(duration_ms)})

    async def input_text(self, text: str, clear: bool = False) -> bool:
        await self._send_command_to_agent("input_text", {"text": text, "clear": clear})
        return True

    async def press_key(self, keycode: int) -> None:
        await self._send_command_to_agent("press_key", {"keycode": keycode})

    async def drag(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration: float = 3.0,
    ) -> None:
        # We can simulate drag with swipe for now
        await self.swipe(x1, y1, x2, y2, duration * 1000)

    # -- app management ------------------------------------------------------

    async def start_app(self, package: str, activity: Optional[str] = None) -> str:
        res = await self._send_command_to_agent("start_app", {"package": package, "activity": activity})
        return res.get("msg", "")

    async def install_app(self, path: str, **kwargs) -> str:
        reinstall = kwargs.get("reinstall", False)
        grant_permissions = kwargs.get("grant_permissions", True)
        res = await self._send_command_to_agent("install_app", {"path": path, "reinstall": reinstall, "grant_permissions": grant_permissions})
        return res.get("msg", "")

    async def get_apps(self, include_system: bool = True) -> List[Dict[str, str]]:
        return []

    async def list_packages(self, include_system: bool = False) -> List[str]:
        return []

    # -- state / observation -------------------------------------------------

    async def screenshot(self, hide_overlay: bool = True) -> bytes:
        res = await self._send_command_to_agent("take_screenshot_b64", {})
        if "b64" in res:
            return base64.b64decode(res["b64"])
        return b""

    async def get_ui_tree(self) -> Dict[str, Any]:
        res = await self._send_command_to_agent("portal_get_state", {})
        return res.get("state", {})

    async def get_date(self) -> str:
        res = await self._send_command_to_agent("execute_shell", {"cmd": "date"})
        return res.get("output", "")
