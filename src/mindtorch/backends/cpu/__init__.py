"""CPU backend module for mindtorch.

This module provides CPU-related backend functionality similar to torch.backends.cpu.
"""


def get_cpu_capability() -> str:
    """Get the CPU capability string.

    Returns a string indicating the CPU's SIMD capability level.
    Possible return values: "AVX512", "AVX2", "AVX", "DEFAULT"

    Returns:
        str: The CPU capability string.
    """
    try:
        import platform
        import subprocess

        system = platform.system()

        if system == "Darwin":
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "hw.optional.avx2_0"],
                    capture_output=True,
                    text=True,
                    timeout=1
                )
                if result.returncode == 0 and result.stdout.strip() == "1":
                    return "AVX2"

                result = subprocess.run(
                    ["sysctl", "-n", "hw.optional.avx1_0"],
                    capture_output=True,
                    text=True,
                    timeout=1
                )
                if result.returncode == 0 and result.stdout.strip() == "1":
                    return "AVX"
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        elif system == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()
                    if "avx512" in cpuinfo.lower():
                        return "AVX512"
                    elif "avx2" in cpuinfo.lower():
                        return "AVX2"
                    elif "avx" in cpuinfo.lower():
                        return "AVX"
            except (FileNotFoundError, PermissionError):
                pass
    except Exception:
        pass

    return "DEFAULT"
