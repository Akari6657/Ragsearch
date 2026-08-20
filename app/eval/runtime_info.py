"""Runtime hardware provenance shared by retrieval evaluation reports."""

from __future__ import annotations

import warnings
from typing import Any


def collect_accelerator_info() -> dict[str, Any]:
    """Return JSON-safe Torch and accelerator facts for reproducibility."""
    info: dict[str, Any] = {
        "torch_version": None,
        "torch_cuda_version": None,
        "cuda_available": False,
        "device_count": 0,
        "device_index": None,
        "device_name": None,
    }
    try:
        import torch

        info["torch_version"] = str(torch.__version__)
        info["torch_cuda_version"] = (
            str(torch.version.cuda) if torch.version.cuda is not None else None
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Can't initialize NVML",
                category=UserWarning,
            )
            info["cuda_available"] = bool(torch.cuda.is_available())
            info["device_count"] = int(torch.cuda.device_count())
            if info["cuda_available"]:
                device_index = int(torch.cuda.current_device())
                info["device_index"] = device_index
                info["device_name"] = torch.cuda.get_device_name(device_index)
    except (ImportError, OSError, RuntimeError) as exc:
        info["detection_error"] = f"{type(exc).__name__}: {exc}"
    return info
