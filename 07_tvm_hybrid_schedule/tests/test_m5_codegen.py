"""Codegen path: CUDA sm_70 target + relax.build (skip launch without GPU)."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

tvm = pytest.importorskip("tvm")
pytest.importorskip("tvm.relax")

from cuda_codegen import (
    classify_kernel_kind,
    compute_version_to_arch,
    cuda_sm70_target,
    detect_cuda_target_from_device,
    is_cuda_target,
    parse_target,
)


def test_default_cli_target_is_cuda_sm70():
    t = parse_target(None)
    assert is_cuda_target(t)
    assert str(t.attrs.get("arch")) == "sm_70"
    assert t.host is not None
    assert t.host.kind.name == "llvm"


def test_bare_cuda_alias_is_sm70_not_sm50():
    t = parse_target("cuda")
    assert str(t.attrs.get("arch")) == "sm_70"


def test_compute_version_to_arch():
    assert compute_version_to_arch("7.0") == "sm_70"
    assert compute_version_to_arch("8.6") == "sm_86"


def test_pipeline_default_is_hardware_or_llvm():
    from hybrid_pipeline import _default_target

    t = _default_target()
    live = detect_cuda_target_from_device(0)
    if live is None:
        assert t.kind.name == "llvm"
    else:
        assert t.kind.name == "cuda"
        assert str(t.attrs.get("arch")) == str(live.attrs.get("arch"))


def test_classify_kernel_kind_unscheduled_on_legalized_only():
    from relax_models import build_matmul_module, legalize

    mod = legalize(build_matmul_module(32, 32, 32), tvm.target.Target("llvm"))
    kinds = classify_kernel_kind(mod)
    assert kinds
    assert all(v == "unscheduled" for v in kinds.values())


def _nvcc_available() -> bool:
    return shutil.which("nvcc") is not None


@pytest.mark.skipif(not _nvcc_available(), reason="nvcc required for CUDA codegen")
def test_codegen_emits_cuda_source():
    from cuda_codegen import (
        build_cuda_executable,
        export_artifacts,
        prepare_cuda_compile_env,
    )
    from hybrid_pipeline import apply_hybrid_pipeline
    from relax_models import build_matmul_module
    from schedule_config import ScheduleConfig

    target = cuda_sm70_target()
    with tempfile.TemporaryDirectory() as tmp:
        cfg = ScheduleConfig(
            force_mode="dlight",
            enable_byoc_stage6=False,
            work_dir=tmp,
        )
        mod, _ = apply_hybrid_pipeline(
            build_matmul_module(64, 64, 64),
            target=target,
            config=cfg,
            work_dir=tmp,
            skip_metaschedule_tune=True,
        )
        prepare_cuda_compile_env(os.path.join(tmp, "kernel_dump"))
        ex = build_cuda_executable(mod, target)
        artifacts = export_artifacts(ex, tmp)
        assert os.path.isfile(artifacts["so"])
        assert "cu" in artifacts
        src = open(artifacts["cu"], encoding="utf-8").read()
        assert "__global__" in src
        assert "blockIdx" in src
