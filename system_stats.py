"""
system_stats.py — GPU VRAM and system RAM monitoring utilities.

Detects Intel and NVIDIA GPUs and reports memory usage.
Falls back to system RAM when GPU stats are unavailable.
"""

import os
import subprocess
import functools
import psutil


@functools.lru_cache(maxsize=1)
def _is_wsl2() -> bool:
    """Detect if running inside WSL2 by checking /proc/version."""
    try:
        with open("/proc/version", "r") as f:
            version_str = f.read().lower()
        return "microsoft" in version_str or "wsl" in version_str
    except (FileNotFoundError, PermissionError):
        return False


def get_system_stats(active_device: str | None = None) -> dict:
    """
    Collect GPU VRAM, CPU utilization, and system RAM statistics.

    Args:
        active_device: The device the model is running on (e.g., "GPU", "CPU").

    Returns:
        dict with 'gpu' (optional), 'ram', and 'cpu' stats.
    """
    wsl2 = _is_wsl2()

    # CPU utilization (non-blocking, uses cached value from last interval)
    cpu_percent = psutil.cpu_percent(interval=None)
    cpu_count = psutil.cpu_count(logical=True)

    stats = {
        "ram": _get_ram_stats(wsl2=wsl2),
        "cpu": {
            "percent": cpu_percent,
            "cores": cpu_count,
        },
        "gpu": None,
        "wsl2": wsl2,
    }

    # Try to detect GPU stats
    device = (active_device or "").upper()
    if device in ("GPU", "AUTO", "XPU") or "HETERO" in device or "XPU" in device:
        # Try NVIDIA first, then Intel
        gpu_stats = _get_nvidia_gpu_stats()
        if not gpu_stats and not wsl2:
            gpu_stats = _get_intel_gpu_stats()
        stats["gpu"] = gpu_stats

    return stats


def _get_ram_stats(wsl2: bool = False) -> dict:
    """Get system RAM stats using psutil."""
    mem = psutil.virtual_memory()
    result = {
        "total_bytes": mem.total,
        "used_bytes": mem.used,
        "free_bytes": mem.available,
        "percent": mem.percent,
        "total_display": _format_bytes(mem.total),
        "used_display": _format_bytes(mem.used),
        "free_display": _format_bytes(mem.available),
        "wsl2": wsl2,
    }
    if wsl2:
        result["label"] = "WSL2 Memory"
        result["note"] = (
            "Showing memory allocated to the WSL2 VM, "
            "not the full host system RAM."
        )
    return result


def _get_nvidia_gpu_stats() -> dict | None:
    """
    Try to get NVIDIA GPU memory stats via nvidia-smi.
    Returns None if NVIDIA GPU is not available.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,nounits,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            return None

        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            return None

        name = parts[0]
        total_mb = int(parts[1])
        used_mb = int(parts[2])
        free_mb = int(parts[3])

        total_bytes = total_mb * 1024 * 1024
        used_bytes = used_mb * 1024 * 1024
        free_bytes = free_mb * 1024 * 1024
        percent = round((used_mb / total_mb) * 100, 1) if total_mb > 0 else 0

        return {
            "name": name,
            "type": "nvidia",
            "total_bytes": total_bytes,
            "used_bytes": used_bytes,
            "free_bytes": free_bytes,
            "percent": percent,
            "total_display": _format_bytes(total_bytes),
            "used_display": _format_bytes(used_bytes),
            "free_display": _format_bytes(free_bytes),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return None


def _get_intel_gpu_stats_via_xpu_smi() -> dict | None:
    """Try to get Intel XPU stats using xpu-smi CLI."""
    import subprocess
    import json
    import re
    try:
        # Check if xpu-smi is available and get device list
        result = subprocess.run(
            ["xpu-smi", "discovery", "-j"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            return None

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return None

        device_list = data.get("device_list", [])
        if not device_list:
            return None

        # Target the first device (device_id 0 or first in list)
        device = device_list[0]
        device_id = device.get("device_id", 0)
        gpu_name = device.get("device_name", "Intel GPU")

        # Parse physical memory size (e.g., "16384 MiB" or "16 GB" or an int)
        total_bytes = None
        mem_size_str = device.get("memory_physical_size") or device.get("memory_size")
        if mem_size_str:
            if isinstance(mem_size_str, (int, float)):
                total_bytes = int(mem_size_str)
            else:
                match = re.search(r"(\d+(?:\.\d+)?)\s*([kmgt]i?b?)?", str(mem_size_str), re.IGNORECASE)
                if match:
                    val = float(match.group(1))
                    unit = (match.group(2) or "").lower()
                    multipliers = {
                        "": 1, "b": 1, "kb": 1024, "kib": 1024,
                        "mb": 1024**2, "mib": 1024**2,
                        "gb": 1024**3, "gib": 1024**3,
                        "tb": 1024**4, "tib": 1024**4,
                    }
                    total_bytes = int(val * multipliers.get(unit, 1))

        # Get dynamic stats
        stats_result = subprocess.run(
            ["xpu-smi", "stats", "-d", str(device_id), "-j"],
            capture_output=True,
            text=True,
            timeout=5
        )

        used_bytes = None
        percent = None

        if stats_result.returncode == 0:
            try:
                stats_data = json.loads(stats_result.stdout)
                stats_device_list = stats_data.get("device_list", [])
                if stats_device_list:
                    stats_device = stats_device_list[0]
                    # The stats are often in stats_device.get("stats", {}) or stats_device
                    stats_dict = stats_device.get("stats", {}) or stats_device

                    # Try to find memory used
                    mem_used = stats_dict.get("memory_used")
                    if mem_used is None:
                        # Try to iterate through keys to find something like memory_used
                        for k, v in stats_dict.items():
                            if "memory" in k and "used" in k:
                                mem_used = v
                                break

                    # If it's a dict like {"value": 15, "unit": "MiB"}, parse it
                    if isinstance(mem_used, dict):
                        val = mem_used.get("value")
                        unit = str(mem_used.get("unit") or "mib").lower()
                        if val is not None:
                            multipliers = {
                                "b": 1, "kb": 1024, "kib": 1024,
                                "mb": 1024**2, "mib": 1024**2,
                                "gb": 1024**3, "gib": 1024**3,
                            }
                            used_bytes = int(float(val) * multipliers.get(unit, 1024**2))
                    elif mem_used is not None:
                        match = re.search(r"(\d+(?:\.\d+)?)\s*([kmgt]i?b?)?", str(mem_used), re.IGNORECASE)
                        if match:
                            val = float(match.group(1))
                            unit = (match.group(2) or "mib").lower()
                            multipliers = {
                                "": 1024**2, "b": 1, "kb": 1024, "kib": 1024,
                                "mb": 1024**2, "mib": 1024**2,
                                "gb": 1024**3, "gib": 1024**3,
                            }
                            used_bytes = int(val * multipliers.get(unit, 1024**2))

                    # Try to find memory utilization
                    mem_util = stats_dict.get("memory_utilization")
                    if mem_util is None:
                        for k, v in stats_dict.items():
                            if "memory" in k and "util" in k:
                                mem_util = v
                                break
                    if isinstance(mem_util, dict):
                        percent = mem_util.get("value")
                    elif mem_util is not None:
                        try:
                            percent = float(mem_util)
                        except ValueError:
                            pass
            except Exception as e:
                print(f"[system_stats] Warning: Failed to parse xpu-smi stats: {e}")

        # Fallbacks for missing metrics
        if used_bytes is None and percent is not None and total_bytes is not None:
            used_bytes = int(total_bytes * (percent / 100.0))

        if percent is None and used_bytes is not None and total_bytes is not None and total_bytes > 0:
            percent = round((used_bytes / total_bytes) * 100, 1)

        if total_bytes is not None:
            if used_bytes is None:
                used_bytes = 0
            free_bytes = total_bytes - used_bytes
            return {
                "name": gpu_name,
                "type": "intel",
                "total_bytes": total_bytes,
                "used_bytes": used_bytes,
                "free_bytes": free_bytes,
                "percent": percent,
                "total_display": _format_bytes(total_bytes),
                "used_display": _format_bytes(used_bytes) if used_bytes is not None else "N/A",
                "free_display": _format_bytes(free_bytes) if used_bytes is not None else "N/A",
            }

    except Exception as e:
        print(f"[system_stats] Warning: Exception in _get_intel_gpu_stats_via_xpu_smi: {e}")

    return None


def _get_intel_gpu_stats() -> dict | None:
    """
    Try to get Intel GPU memory stats.

    First tries xpu-smi (Intel XPU System Management Interface) for discrete GPUs,
    then falls back to DRM sysfs and shared memory estimation for integrated/other GPUs.
    """
    # Try xpu-smi first for discrete Intel XPUs
    xpu_stats = _get_intel_gpu_stats_via_xpu_smi()
    if xpu_stats:
        return xpu_stats

    try:
        # Try to find Intel GPU via DRM subsystem
        drm_path = "/sys/class/drm"
        if not os.path.isdir(drm_path):
            return None

        for card_dir in sorted(os.listdir(drm_path)):
            if not card_dir.startswith("card") or "-" in card_dir:
                continue

            card_path = os.path.join(drm_path, card_dir)

            # Check if this is an Intel GPU
            device_vendor_path = os.path.join(card_path, "device", "vendor")
            if os.path.isfile(device_vendor_path):
                with open(device_vendor_path, "r") as f:
                    vendor = f.read().strip()
                # Intel vendor ID is 0x8086
                if vendor != "0x8086":
                    continue
            else:
                continue

            # Try to get GPU name
            gpu_name = "Intel GPU"
            device_name_path = os.path.join(card_path, "device", "label")
            if os.path.isfile(device_name_path):
                with open(device_name_path, "r") as f:
                    gpu_name = f.read().strip()

            # For Intel iGPUs, VRAM is shared with system RAM
            # Report the lmem (local memory) if available (discrete Intel GPUs)
            lmem_path = os.path.join(card_path, "device", "resource")

            # Try to get memory info from i915 driver
            mem_info = _get_intel_i915_mem_info(card_path)
            if mem_info:
                mem_info["name"] = gpu_name
                mem_info["type"] = "intel"
                return mem_info

            # Fallback: Intel iGPU shares system RAM, estimate usage
            mem = psutil.virtual_memory()
            # Intel iGPU typically can use up to half of system RAM
            # but is limited by the max allocation size (~1GB on many systems)
            max_alloc = 1024 * 1024 * 1024  # 1GB typical max for iGPU
            estimated_total = min(mem.total // 2, 4 * 1024 * 1024 * 1024)  # Up to 4GB

            return {
                "name": gpu_name + " (Shared Memory)",
                "type": "intel_shared",
                "total_bytes": estimated_total,
                "used_bytes": None,  # Can't accurately measure shared GPU usage
                "free_bytes": None,
                "percent": None,
                "total_display": _format_bytes(estimated_total),
                "used_display": "N/A",
                "free_display": "N/A",
                "max_alloc_bytes": max_alloc,
                "max_alloc_display": _format_bytes(max_alloc),
                "shared": True,
                "note": "Intel iGPU shares system RAM. Max single allocation: " + _format_bytes(max_alloc),
            }

        return None

    except Exception:
        return None


def _get_intel_i915_mem_info(card_path: str) -> dict | None:
    """Try to read i915 driver memory info from debugfs or sysfs."""
    try:
        # Try the i915 memory regions info (available in newer kernels)
        mem_regions_path = os.path.join(card_path, "device", "drm")

        # Check for local memory (discrete Intel GPUs like Arc)
        for entry in os.listdir(card_path):
            lmem_path = os.path.join(card_path, entry)
            if "gt" in entry and os.path.isdir(lmem_path):
                for gt_entry in os.listdir(lmem_path):
                    if "mem" in gt_entry.lower():
                        mem_file = os.path.join(lmem_path, gt_entry)
                        if os.path.isfile(mem_file):
                            with open(mem_file, "r") as f:
                                content = f.read().strip()
                            # Parse if possible
                            return _parse_intel_mem_file(content)
    except (PermissionError, FileNotFoundError, OSError):
        pass

    return None


def _parse_intel_mem_file(content: str) -> dict | None:
    """Parse Intel GPU memory file content."""
    try:
        total = used = free = 0
        for line in content.split("\n"):
            line = line.strip().lower()
            if "total" in line:
                total = _extract_bytes_from_line(line)
            elif "used" in line:
                used = _extract_bytes_from_line(line)
            elif "free" in line or "avail" in line:
                free = _extract_bytes_from_line(line)

        if total > 0:
            if free == 0 and used > 0:
                free = total - used
            percent = round((used / total) * 100, 1) if total > 0 else 0
            return {
                "total_bytes": total,
                "used_bytes": used,
                "free_bytes": free,
                "percent": percent,
                "total_display": _format_bytes(total),
                "used_display": _format_bytes(used),
                "free_display": _format_bytes(free),
            }
    except Exception:
        pass
    return None


def _extract_bytes_from_line(line: str) -> int:
    """Extract byte value from a line containing memory information."""
    import re
    # Try to find a number (possibly with units)
    match = re.search(r"(\d+)\s*(bytes?|[kmgt]i?b?)?", line, re.IGNORECASE)
    if not match:
        return 0

    value = int(match.group(1))
    unit = (match.group(2) or "").lower().strip()

    multipliers = {
        "": 1, "b": 1, "byte": 1, "bytes": 1,
        "k": 1024, "kb": 1024, "kib": 1024,
        "m": 1024**2, "mb": 1024**2, "mib": 1024**2,
        "g": 1024**3, "gb": 1024**3, "gib": 1024**3,
        "t": 1024**4, "tb": 1024**4, "tib": 1024**4,
    }

    return value * multipliers.get(unit, 1)


def _format_bytes(num_bytes: int) -> str:
    """Format bytes into a human-readable string."""
    if num_bytes is None:
        return "N/A"

    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            if unit in ("B", "KB"):
                return f"{num_bytes:.0f} {unit}"
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0

    return f"{num_bytes:.1f} PB"
