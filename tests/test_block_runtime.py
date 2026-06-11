"""Tests for block speculative runtime."""

from megakernel_lab.block_runtime import BlockSpeculativeRuntime
from megakernel_lab.block_spec_decode import DFlashStyleDrafter


class TestBlockSpeculativeRuntime:
    def test_block_runtime_generates_max_new_tokens(self) -> None:
        runtime = BlockSpeculativeRuntime()
        metrics = runtime.run(max_new_tokens=32)
        assert metrics.committed_tokens == 32

    def test_average_accepted_prefix_len_gt_one(self) -> None:
        runtime = BlockSpeculativeRuntime()
        metrics = runtime.run(max_new_tokens=64)
        assert metrics.average_accepted_prefix_len > 1.0

    def test_block_spec_uses_fewer_iterations_than_serial(self) -> None:
        serial = BlockSpeculativeRuntime(drafter=DFlashStyleDrafter(block_size=1))
        block = BlockSpeculativeRuntime(drafter=DFlashStyleDrafter(block_size=8))
        serial_metrics = serial.run(max_new_tokens=64)
        block_metrics = block.run(max_new_tokens=64)
        assert block_metrics.iterations < serial_metrics.iterations

    def test_persistent_sim_reports_one_launch(self) -> None:
        from megakernel_lab.bench import BenchmarkMode, BenchmarkRunner

        runner = BenchmarkRunner(
            batch_sizes=[1],
            block_sizes=[4],
            modes=[BenchmarkMode.BLOCK_SPECULATIVE_PERSISTENT_SIM],
        )
        df = runner.run()
        row = df.iloc[0]
        assert row["host_kernel_launches"] == 1
        assert row["host_synchronizations"] == 1

    def test_host_orchestrated_reports_many_launches(self) -> None:
        from megakernel_lab.bench import BenchmarkMode, BenchmarkRunner

        runner = BenchmarkRunner(
            batch_sizes=[1],
            block_sizes=[4],
            modes=[BenchmarkMode.BLOCK_SPECULATIVE_HOST_ORCHESTRATED],
        )
        df = runner.run()
        row = df.iloc[0]
        assert row["host_kernel_launches"] > 1
        assert row["host_synchronizations"] > 1

    def test_block_speculative_tracks_draft_blocks(self) -> None:
        runtime = BlockSpeculativeRuntime()
        metrics = runtime.run(max_new_tokens=32)
        assert metrics.draft_blocks > 0
        assert metrics.total_draft_tokens > 0

    def test_rejected_tokens_are_tracked(self) -> None:
        runtime = BlockSpeculativeRuntime()
        metrics = runtime.run(max_new_tokens=64)
        assert metrics.rejected_tokens >= 0
