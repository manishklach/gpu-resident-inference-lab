/**
 * persistent_decode_stub.cu - Backward-compatible wrapper.
 *
 * This file is kept at the old path for projects that reference it.
 * The canonical implementation now lives at src/persistent_decode_kernel.cu
 * and src/host_launcher.cpp.
 */

// For reference: the old main() entry point is now in src/host_launcher.cpp.
// Build the full smoke-test suite with: make cuda-smoke

#include <cstdio>

int main() {
    printf("XL-Persistent-Kernel: persistent_decode_stub.cu\n");
    printf("This file is kept for backward compatibility.\n");
    printf("Use 'make cuda-smoke' to build the full test suite.\n\n");

    printf("Or migrate to the new structure:\n");
    printf("  cuda/include/   - headers (request_desc.h, kv_page_table.h, etc.)\n");
    printf("  cuda/src/       - kernel sources + host_launcher.cpp\n");
    printf("  cuda/build/     - build directory\n\n");

    printf("Build: mkdir -p cuda/build && cd cuda/build && cmake .. && make\n");
    return 0;
}
