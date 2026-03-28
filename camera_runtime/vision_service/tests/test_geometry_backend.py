from __future__ import annotations

import numpy as np
import pytest

from vision_service.app import geometry_backend as gb


def test_normalize_geom_backend_defaults_to_auto() -> None:
    assert gb.normalize_geom_backend(None) == "auto"
    assert gb.normalize_geom_backend("invalid") == "auto"
    assert gb.normalize_geom_backend("TORCH") == "torch"


def test_resolve_geom_backend_matrix() -> None:
    assert gb.resolve_geom_backend("cpu", torch_cuda_available=True, cuml_available=True) == "cpu"
    assert gb.resolve_geom_backend("torch", torch_cuda_available=False, cuml_available=False) == "cpu"
    assert gb.resolve_geom_backend("cuml", torch_cuda_available=True, cuml_available=False) == "torch"
    assert gb.resolve_geom_backend("auto", torch_cuda_available=False, cuml_available=False) == "cpu"
    assert gb.resolve_geom_backend("auto", torch_cuda_available=True, cuml_available=False) == "torch"
    assert gb.resolve_geom_backend("auto", torch_cuda_available=True, cuml_available=True) == "cuml"


def test_build_geometry_backend_prefers_requested_cpu() -> None:
    backend, selection = gb.build_geometry_backend(requested_backend="cpu", yolo_device="cuda:0")
    assert backend.name == "cpu"
    assert selection.requested == "cpu"
    assert selection.resolved == "cpu"


def test_build_geometry_backend_uses_torch_when_available(monkeypatch) -> None:
    class FakeTorchBackend(gb.CPUReferenceBackend):
        name = "torch"
        is_gpu = True

        def __init__(self, device: str = "cuda:0") -> None:
            self.device = device

    monkeypatch.setattr(gb, "detect_torch_cuda_available", lambda: True)
    monkeypatch.setattr(gb, "detect_cuml_available", lambda: False)
    monkeypatch.setattr(gb, "TorchCudaBackend", FakeTorchBackend)
    backend, selection = gb.build_geometry_backend(requested_backend="auto", yolo_device="cuda:0")
    assert backend.name == "torch"
    assert selection.resolved == "torch"


def test_build_geometry_backend_falls_back_when_torch_ctor_fails(monkeypatch) -> None:
    class BrokenTorchBackend:
        def __init__(self, device: str = "cuda:0") -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(gb, "detect_torch_cuda_available", lambda: True)
    monkeypatch.setattr(gb, "detect_cuml_available", lambda: False)
    monkeypatch.setattr(gb, "TorchCudaBackend", BrokenTorchBackend)
    backend, selection = gb.build_geometry_backend(requested_backend="torch", yolo_device="cuda:0")
    assert backend.name == "cpu"
    assert selection.resolved == "cpu"


@pytest.mark.skipif(not gb.detect_torch_cuda_available(), reason="requires torch cuda")
def test_torch_backend_depth_points_matches_cpu_reference() -> None:
    rng = np.random.default_rng(7)
    depth = rng.integers(750, 1250, size=(120, 160), dtype=np.uint16)
    bbox = (18, 22, 130, 108)

    cpu_backend = gb.CPUReferenceBackend()
    try:
        torch_backend = gb.TorchCudaBackend("cuda:0")
    except RuntimeError as exc:
        pytest.skip(f"torch backend probe failed: {exc}")
    cpu_points, cpu_uv = cpu_backend.depth_roi_to_points(
        depth=depth,
        bbox_xyxy=bbox,
        depth_scale=1.0,
        fx=130.0,
        fy=128.0,
        cx=80.0,
        cy=60.0,
        min_depth_mm=80.0,
        max_depth_mm=5000.0,
    )
    torch_points, torch_uv = torch_backend.depth_roi_to_points(
        depth=depth,
        bbox_xyxy=bbox,
        depth_scale=1.0,
        fx=130.0,
        fy=128.0,
        cx=80.0,
        cy=60.0,
        min_depth_mm=80.0,
        max_depth_mm=5000.0,
    )
    assert cpu_points.shape == torch_points.shape
    assert cpu_uv.shape == torch_uv.shape
    assert float(np.max(np.abs(cpu_points - torch_points))) < 1e-3
    assert float(np.max(np.abs(cpu_uv - torch_uv))) < 1e-6
