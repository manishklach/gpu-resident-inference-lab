"""Small runnable demo for the persistent decode simulator."""

from .config import RuntimeConfig
from .runtime import PersistentDecodeRuntime
from .spec_decode import AcceptancePolicy, DraftBlockProposer, SpeculativeVerifier
from .state import DecodeRequest


def run_demo(block_size: int, mismatch_stride: int) -> None:
    """Run one decode simulation and print a concise trace."""
    config = RuntimeConfig(block_size=block_size, max_new_tokens=12, eos_token_id=0)
    runtime = PersistentDecodeRuntime(
        config=config,
        proposer=DraftBlockProposer(block_size=block_size, eos_token_id=config.eos_token_id),
        verifier=SpeculativeVerifier(AcceptancePolicy(mismatch_stride=mismatch_stride)),
    )

    request = DecodeRequest(
        request_id=1,
        prompt_tokens=[101, 102],
        target_tokens=[11, 12, 13, 14, 15, 16, 0],
        max_new_tokens=config.max_new_tokens,
        eos_token_id=config.eos_token_id,
    )
    runtime.submit(request)
    result = runtime.run()[0]

    print(f"block_size={block_size} mismatch_stride={mismatch_stride}")
    for step, trace in enumerate(result.traces, start=1):
        print(
            f"step={step} proposed={trace.proposed_tokens} "
            f"accepted={trace.accepted_tokens}"
        )
    print(f"final_sequence={result.full_sequence}")
    print(f"kv_positions={runtime.kv_cache.positions_for(result.request_id)}")
    print()


def main() -> None:
    """Compare serial decode with larger speculative blocks."""
    print("Serial-style decode")
    run_demo(block_size=1, mismatch_stride=0)

    print("Speculative block decode")
    run_demo(block_size=4, mismatch_stride=0)

    print("Speculative block decode with forced tail rejections")
    run_demo(block_size=4, mismatch_stride=3)


if __name__ == "__main__":
    main()
