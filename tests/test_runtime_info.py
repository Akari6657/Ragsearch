"""Tests for evaluation hardware provenance."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from app.eval.runtime_info import collect_accelerator_info


class FakeCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_count():
        return 1

    @staticmethod
    def current_device():
        return 0

    @staticmethod
    def get_device_name(index):
        assert index == 0
        return "Test GPU"


def test_collect_accelerator_info_records_cuda_device(monkeypatch):
    fake_torch = SimpleNamespace(
        __version__="test-torch",
        version=SimpleNamespace(cuda="test-cuda"),
        cuda=FakeCuda(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    info = collect_accelerator_info()

    assert info == {
        "torch_version": "test-torch",
        "torch_cuda_version": "test-cuda",
        "cuda_available": True,
        "device_count": 1,
        "device_index": 0,
        "device_name": "Test GPU",
    }
