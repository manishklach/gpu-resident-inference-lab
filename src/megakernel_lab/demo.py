"""Small runnable demo for the persistent decode simulator."""

from .backend import CPUStubBackend
from .bench import BenchmarkMode, BenchmarkRunner
from .config import RuntimeConfig
from .runtime import PersistentDecodeRuntime
from .spec_decode import AcceptancePolicy, DraftBlockProposer, SpeculativeVerifier
from .state import RequestState


def run_demo(block_size: int, mismatch_stride: int, reject_draft_blocks: bool = False) -> None:
    """Run one decode simulation and print a concise trace."""
    config = RuntimeConfig(
        block_size=block_size,
        max_new_tokens=12,
        eos_token_id=0,
        page_size=4,
        max_pages=16,
        num_layers=2,
        num_prefill_workers=1,
        num_decode_workers=1,
    )
    runtime = PersistentDecodeRuntime(
        config=config,
        proposer=DraftBlockProposer(block_size=block_size, eos_token_id=config.eos_token_id),
        verifier=SpeculativeVerifier(
            AcceptancePolicy(
                mismatch_stride=mismatch_stride,
                reject_draft_blocks=reject_draft_blocks,
            )
        ),
        backend=CPUStubBackend(),
    )

    request = RequestState(
        request_id=1,
        prompt_tokens=[101, 102],
        target_tokens=[11, 12, 13, 14, 15, 16, 0],
        max_new_tokens=config.max_new_tokens,
        eos_token_id=config.eos_token_id,
        priority=1,
        layer_ids=[0, 1],
    )
    runtime.submit(request)
    result = runtime.run()[0]

    print(
        f"block_size={block_size} mismatch_stride={mismatch_stride} "
        f"reject_draft_blocks={reject_draft_blocks}"
    )
    for step, trace in enumerate(result.traces, start=1):
        print(
            f"step={step} worker={trace.worker_id} proposed={trace.proposed_tokens} "
            f"accepted={trace.accepted_tokens} fallback={trace.used_fallback_serial}"
        )
    print(f"final_sequence={result.full_sequence}")
    print(f"ttft_ms={result.ttft_ms:.3f} acceptance_rate={result.acceptance_rate:.3f}")
    print(f"kv_report={runtime.kv_cache.residency_report()}")
    print()


def main() -> None:
    """Compare serial decode with larger speculative blocks, then run benchmarks."""
    print("=== Decode Mode Comparison ===")
    print()

    print("Serial-style decode")
    run_demo(block_size=1, mismatch_stride=0)

    print("Speculative block decode")
    run_demo(block_size=4, mismatch_stride=0)

    print("Speculative block decode with forced tail rejections")
    run_demo(block_size=4, mismatch_stride=3)

    print("Speculative block decode with serial fallback")
    run_demo(block_size=4, mismatch_stride=0, reject_draft_blocks=True)

    print("=== Benchmark Modes ===")
    print()

    runner = BenchmarkRunner(batch_sizes=[1, 4, 8], block_sizes=[1, 2, 4])

    for mode in BenchmarkMode:
        print(f"--- {mode.value} ---")
        df = runner.run(modes=[mode])
        print(df.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
