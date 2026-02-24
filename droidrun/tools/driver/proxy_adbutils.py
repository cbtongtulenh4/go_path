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
            
        remote_path = f"temp_install_{os.path.basename(path)}"
        await self.driver._send_command_to_agent("push_file_b64", {"path": remote_path, "b64_content": b64_content})
        
        grant = flags is not None and "-g" in flags
        res = await self.driver._send_command_to_agent("install_app", {"path": remote_path, "reinstall": uninstall, "grant_permissions": grant})
        return res.get("msg", "")

    async def list_packages(self, *args, **kwargs) -> List[str]:
        res = await self.driver._send_command_to_agent("list_packages", {})
        return res.get("packages", [])

    async def forward_port(self, remote_port: int, local_port: Optional[int] = None) -> int:
        """Forward a local port to a remote port."""
        res = await self.driver._send_command_to_agent("forward_port", {"remote_port": remote_port, "local_port": local_port})
        if "error" in res:
            raise Exception(res["error"])
        return res.get("local_port", 0)

    async def forward_list(self):
        """List active forwards."""
        res = await self.driver._send_command_to_agent("forward_list", {})
        
        class ForwardItem:
            def __init__(self, serial, local, remote):
                self.serial = serial
                self.local = local
                self.remote = remote
                
        forws = res.get("forwards", [])
        for f in forws:
            yield ForwardItem(f["serial"], f["local"], f["remote"])

    async def screenshot_bytes(self) -> bytes:
        """Take screenshot and return bytes."""
        return await self.driver.screenshot(hide_overlay=True)

AdbDevice = ProxyAdbDevice


class MockDeviceObj:
    def __init__(self, serial: str):
        self.serial = serial


class ProxyAdb:

    async def device(self, serial: Optional[str] = None) -> ProxyAdbDevice | None:
        if serial is None:
            devices = await self.list()
            if not devices:
                return None
            serial = devices[0].serial
            
        return ProxyAdbDevice(serial)

    async def list(self) -> List[MockDeviceObj]:
        driver = AndroidDriver(serial="dummy")
        res = await driver._send_command_to_agent("list_devices", {})
        devices = res.get("devices", [])
        return [MockDeviceObj(d["device_id"]) for d in devices]

    async def connect(self, serial: str) -> str:
        driver = AndroidDriver(serial="dummy")
        res = await driver._send_command_to_agent("adb_connect", {"serial": serial})
        return res.get("msg", "")

    async def disconnect(self, serial: str, raise_error: bool = False) -> bool:
        driver = AndroidDriver(serial="dummy")
        res = await driver._send_command_to_agent("adb_disconnect", {"serial": serial})
        msg = res.get("msg", "")
        return not msg.startswith("Error")

adb = ProxyAdb()
