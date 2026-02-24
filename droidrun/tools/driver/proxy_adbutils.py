"""Proxy ADB Utilities - Replaces async_adbutils with Remote PC Agent calls via WebSockets."""

import os
import base64
from typing import List, Optional

from droidrun.tools.driver.android import AndroidDriver


class ProxyAdbDevice:
    """Mock AdbDevice that routes commands through AndroidDriver."""
    
    def __init__(self, serial: str):
        self.serial = serial
        self.driver = AndroidDriver(serial=serial)

    async def shell(self, cmd: str) -> str:
        res = await self.driver._send_command_to_agent("execute_shell", {"cmd": cmd})
        return res.get("output", "")

    async def install(self, path: str, uninstall: bool = False, flags: List[str] = None, silent: bool = False):
        with open(path, "rb") as f:
            b64_content = base64.b64encode(f.read()).decode("utf-8")
            
        # PC agent saves it locally to the same filename in its working dir
        remote_path = f"temp_install_{os.path.basename(path)}"
        await self.driver._send_command_to_agent("push_file_b64", {"path": remote_path, "b64_content": b64_content})
        
        grant = flags is not None and "-g" in flags
        # Call the actual install_app command on the agent
        res = await self.driver._send_command_to_agent("install_app", {"path": remote_path, "reinstall": uninstall, "grant_permissions": grant})
        return res.get("msg", "")

    async def list_packages(self, *args, **kwargs) -> List[str]:
        res = await self.driver._send_command_to_agent("list_packages", {})
        return res.get("packages", [])

# Alias for type hinting
AdbDevice = ProxyAdbDevice

class MockDeviceObj:
    """Simple wrapper for list_devices output."""
    def __init__(self, serial: str):
        self.serial = serial


class ProxyAdb:
    """Mock adb module."""

    async def device(self, serial: Optional[str] = None) -> ProxyAdbDevice | None:
        """Returns a ProxyAdbDevice."""
        if serial is None:
            # Get the first available device from list
            devices = await self.list()
            if not devices:
                return None
            serial = devices[0].serial
            
        return ProxyAdbDevice(serial)

    async def list(self) -> List[MockDeviceObj]:
        """Returns list of MockDeviceObj."""
        # Use a dummy driver to route the global command
        driver = AndroidDriver(serial="dummy")
        res = await driver._send_command_to_agent("list_devices", {})
        # PC agent returns: [{"device_id": ..., "name": ...}, ...]
        devices = res.get("devices", [])
        return [MockDeviceObj(d["device_id"]) for d in devices]

    async def connect(self, serial: str) -> str:
        """Connect ADB."""
        driver = AndroidDriver(serial="dummy")
        res = await driver._send_command_to_agent("adb_connect", {"serial": serial})
        return res.get("msg", "")

    async def disconnect(self, serial: str, raise_error: bool = False) -> bool:
        """Disconnect ADB."""
        driver = AndroidDriver(serial="dummy")
        res = await driver._send_command_to_agent("adb_disconnect", {"serial": serial})
        # If no error in msg, return True
        msg = res.get("msg", "")
        return not msg.startswith("Error")

adb = ProxyAdb()
