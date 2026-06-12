"""Test that CUDA staging source files exist.

This test does NOT require CUDA compilation. It only checks file presence.
"""

from pathlib import Path

CUDA_DIR = Path(__file__).resolve().parent.parent / "cuda"

EXPECTED_HEADERS = [
    "include/kernel_status.h",
    "include/request_desc.h",
    "include/kv_page_table.h",
    "include/queue_desc.h",
    "include/research_kernel_metrics.h",
    "include/stage_scheduler.cuh",
    "include/stage_prefill.cuh",
    "include/stage_decode.cuh",
    "include/stage_spec_verify.cuh",
    "include/stage_commit.cuh",
    "include/stage_kv.cuh",
    "include/stage_sparse_kv_select.cuh",
]

EXPECTED_SOURCES = [
    "src/xl_persistent_megakernel.cu",
    "src/baseline_host_decode_kernel.cu",
    "src/resident_scheduler_kernel.cu",
    "src/kv_prefetch_planner_kernel.cu",
    "src/compacted_decode_kernel.cu",
    "src/resident_sparse_decode_pipeline_kernel.cu",
    "src/sparse_kv_gather_kernel.cu",
    "src/verify_commit_kernel.cu",
    "src/host_launcher.cpp",
]

EXPECTED_BUILD = [
    "CMakeLists.txt",
]


def test_cuda_headers_exist():
    for name in EXPECTED_HEADERS:
        path = CUDA_DIR / name
        assert path.is_file(), f"Missing CUDA header: {path}"


def test_cuda_sources_exist():
    for name in EXPECTED_SOURCES:
        path = CUDA_DIR / name
        assert path.is_file(), f"Missing CUDA source: {path}"


def test_cmake_lists_exist():
    for name in EXPECTED_BUILD:
        path = CUDA_DIR / name
        assert path.is_file(), f"Missing CUDA build file: {path}"


def test_backward_compat_stubs_exist():
    """Old-path stubs must still exist for backward compatibility."""
    stub = CUDA_DIR / "persistent_decode_stub.cu"
    assert stub.is_file(), f"Missing backward-compat stub: {stub}"
    old_header = CUDA_DIR / "request_desc.h"
    assert old_header.is_file(), f"Missing backward-compat header: {old_header}"
