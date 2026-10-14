"""Collect run provenance into run.json.

    python bench/meta.py --base-url http://127.0.0.1:8000 --config <run>/config.yaml --out <run>/run.json

Records hardware, driver/CUDA, engine build, git SHA, Slurm job id and the
experiment config. Hostnames, usernames and absolute paths are deliberately
left out; hosts are described by the `host_label` in the experiment config.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path

import httpx
import yaml


def sh(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception:
        return None


def hardware() -> dict:
    hw: dict = {"os": f"{platform.system()} {platform.release()}", "arch": platform.machine()}
    if shutil.which("nvidia-smi"):
        q = sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit,pcie.link.gen.max,"
                "pcie.link.width.current", "--format=csv,noheader"])
        hw["gpus"] = [dict(zip(["name", "driver", "memory", "power_limit", "pcie_gen_max", "pcie_width"],
                               [s.strip() for s in line.split(",")])) for line in (q or "").splitlines()]
        hw["cuda"] = sh(["bash", "-lc", "nvcc --version | tail -1"]) or sh(
            ["bash", "-lc", "nvidia-smi | grep -o 'CUDA Version: [0-9.]*'"])
    if platform.system() == "Linux":
        hw["cpu"] = sh(["bash", "-lc", "lscpu | sed -n 's/^Model name: *//p' | head -1"])
        hw["cpu_cores"] = sh(["nproc"])
        mem = sh(["bash", "-lc", "awk '/MemTotal/ {print $2}' /proc/meminfo"])
        hw["ram_gb"] = round(int(mem) / 1024**2, 1) if mem else None
        hw["kernel"] = platform.release()
    elif platform.system() == "Darwin":
        hw["cpu"] = sh(["sysctl", "-n", "machdep.cpu.brand_string"])
        hw["cpu_cores"] = sh(["sysctl", "-n", "hw.ncpu"])
        mem = sh(["sysctl", "-n", "hw.memsize"])
        hw["ram_gb"] = round(int(mem) / 1024**3, 1) if mem else None
        hw["model"] = sh(["sysctl", "-n", "hw.model"])
        # Laptops run other apps: record memory pressure so a noisy run can be spotted.
        vm = sh(["vm_stat"]) or ""
        stat = {l.split(":")[0].strip(): l.split(":")[1].strip(" .") for l in vm.splitlines()[1:] if ":" in l}
        page = 16384
        hw["memory_state"] = {
            "free_pct": (sh(["bash", "-lc", "memory_pressure | sed -n 's/.*free percentage: //p'"]) or "").rstrip("%"),
            "compressor_gb": round(int(stat.get("Pages occupied by compressor", 0)) * page / 1e9, 2),
            "swapouts_total": int(stat.get("Swapouts", 0)),
        }
    return hw


def engine_info(base_url: str | None) -> dict:
    if not base_url:
        return {}
    info: dict = {}
    root = base_url.rstrip("/").removesuffix("/v1")
    for path in ("/version", "/props", "/get_server_info", "/v1/models"):
        try:
            r = httpx.get(root + path, timeout=10)
            if r.status_code == 200:
                js = r.json()
                if path == "/props":  # llama.cpp: keep build info and key settings, not the chat template
                    js = {k: js.get(k) for k in ("build_info", "model_path", "total_slots",
                                                 "default_generation_settings") if k in js}
                    if js.get("model_path"):
                        js["model_path"] = Path(js["model_path"]).name
                info[path] = js
        except Exception:
            pass
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url")
    ap.add_argument("--config", help="experiment config YAML (engine flags, model, provider, price)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text()) if a.config else {}
    run = {
        "collected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": sh(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(sh(["git", "status", "--porcelain", "--untracked-files=no"])),
        "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID"),
        "hardware": hardware(),
        "engine": engine_info(a.base_url),
        "config": cfg,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(run, indent=2, default=str))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
