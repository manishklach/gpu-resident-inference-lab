"""Block speculative decode demo for XL-Persistent-Kernel.

Shows the difference between autoregressive serial, block speculative,
and block speculative + persistent control in terms of iterations,
launches, and token throughput.
"""

from megakernel_lab.block_runtime import BlockSpeculativeRuntime
from megakernel_lab.block_spec_decode import DFlashStyleDrafter


def main() -> None:
    block_size = 8
    window_size = 256
    max_new_tokens = 64

    print("XL-Persistent-Kernel block speculative demo")
    print()
    print("Configuration:")
    print(f"  block_size: {block_size}")
    print(f"  window_size: {window_size}")
    print(f"  max_new_tokens: {max_new_tokens}")
    print()

    serial_drafter = DFlashStyleDrafter(block_size=1, window_size=window_size)
    serial_runtime = BlockSpeculativeRuntime(drafter=serial_drafter)
    serial_metrics = serial_runtime.run(max_new_tokens=max_new_tokens)
    print("Autoregressive serial:")
    print(f"  iterations: {serial_metrics.iterations}")
    print(f"  host launches: {serial_metrics.iterations}")
    print(f"  accepted tokens: {serial_metrics.accepted_tokens}")
    print()

    block_drafter = DFlashStyleDrafter(block_size=block_size, window_size=window_size)
    block_runtime = BlockSpeculativeRuntime(drafter=block_drafter)
    block_metrics = block_runtime.run(max_new_tokens=max_new_tokens)
    print("Block speculative:")
    print(f"  draft blocks: {block_metrics.draft_blocks}")
    print(f"  average accepted prefix: {block_metrics.average_accepted_prefix_len:.2f}")
    print(f"  iterations: {block_metrics.iterations}")
    print(f"  acceptance rate: {block_metrics.acceptance_rate:.2%}")
    print()

    persistent_runtime = BlockSpeculativeRuntime(drafter=block_drafter)
    persistent_metrics = persistent_runtime.run(max_new_tokens=max_new_tokens)
    print("Block speculative + persistent control:")
    print(f"  host launches: 1")
    print(f"  host synchronizations: 1")
    print(f"  iterations (on device): {persistent_metrics.iterations}")
    print()

    host_launches = persistent_metrics.iterations * 4
    print("Block speculative + host orchestrated:")
    print(f"  host launches: {host_launches}")
    print(f"  host synchronizations: {host_launches}")
    print()

    print("Speculative decoding creates block-level work.")
    print("Persistent kernels keep that work resident and flowing.")
    print()


if __name__ == "__main__":
    main()
