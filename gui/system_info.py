from __future__ import annotations

import os
import platform


def cpu_name() -> str:
    if os.name == "nt":
        # Registry lookup is instant, unlike spawning PowerShell/WMI at startup.
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            if value:
                return " ".join(str(value).split())
        except OSError:
            pass
        env_name = os.environ.get("PROCESSOR_IDENTIFIER", "").strip()
        if env_name:
            return env_name
    fallback = platform.processor().strip()
    return fallback or "Unknown CPU"
