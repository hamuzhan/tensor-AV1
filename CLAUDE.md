# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SVT-AV1** (Scalable Video Technology for AV1) is a production-quality AV1-compliant software encoder library. It was initially founded by Intel in partnership with Netflix and is now maintained by the Alliance of Open Media (AOM). The encoder targets a wide range of applications, from premium VOD to real-time encoding and live transcoding.

**Key Technologies:**
- AV1-compliant video compression
- Supports 8-bit and 10-bit encoding
- HDR and SDR video support
- Multi-pass encoding
- Award winner: IBC 2025 Innovation Award

## Build Commands

### Linux/macOS (Recommended Development Environment)

**Release Build:**
```bash
cd Build
cmake .. -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
make -j $(nproc)
```

**Debug Build:**
```bash
cd Build
cmake .. -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug
make -j $(nproc)
```

**With Link-Time Optimization (LTO):**
```bash
cd Build
cmake .. -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DENABLE_LTO=ON
make -j $(nproc)
```

**With Profile-Guided Optimization (PGO):**
```bash
cd Build
cmake .. -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DENABLE_PGO=ON
make -j $(nproc)
```

**Common Build Options:**
- `-DMINIMAL_BUILD=ON` — Enable minimal build (reduces feature set)
- `-DLOG_QUIET=ON` — Disable all logging from encoder
- `-DRTC_BUILD=ON` — Optimize for RTC (disable unused features)
- `-DCOMPILE_C_ONLY=ON` — Compile only C code without SIMD optimizations

Binaries are output to `Bin/Release` or `Bin/Debug`.

### Windows

Refer to `Docs/Build-Guide.md` for Visual Studio build instructions.

## Testing

### Build Tests

```bash
cd Build
cmake .. -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
make -j $(nproc) test
```

### Run Unit Tests

```bash
./Bin/Release/SvtAv1UnitTests
# List available tests
./Bin/Release/SvtAv1UnitTests --gtest_list_tests
# Run tests matching a pattern (e.g., transform-related tests)
./Bin/Release/SvtAv1UnitTests --gtest_filter="*transform*"
```

### Run API Tests

```bash
./Bin/Release/SvtAv1ApiTests
# With filter
./Bin/Release/SvtAv1ApiTests --gtest_filter="*specific_test*"
```

### Run End-to-End Tests

```bash
# Requires test vectors (auto-downloaded if not present)
./Bin/Release/SvtAv1E2ETests
```

Test vectors are downloaded to `SVT_AV1_TEST_VECTOR_PATH` or `test/vectors` by default.

## Code Style & Formatting

SVT-AV1 follows a strict code style based on dav1d with modifications. The authoritative source is `.clang-format`.

**Key Style Rules (see `STYLE.md` for complete details):**
- **Indentation:** 4 spaces, no tabs
- **Line Length:** Max 120 characters
- **Naming:** `CamelCase` for types, `under_score` for variables/functions
- **Pointers:** Aligned to the left (next to type): `int* pointer`
- **Braces:** Always required for control statements, on same line as statement
- **Const Usage:** Use const extensively; in forward declarations, use const only for const-arrays

**Auto-format Code:**
```bash
clang-format -i <file>
```

**Check Style:**
```bash
# Lint style issues in a test file
./test/stylecheck.sh <file>
```

## High-Level Architecture

SVT-AV1 is built around a **process-based architecture** with multiple encoding stages:

### Core Design Principles
1. **Process-Based:** Encoder is organized as independent processes (threads) that handle specific tasks
2. **Stateless Processes:** All state is communicated via FIFO buffers between processes
3. **Multi-Level Parallelism:**
   - **Process level:** Different processes run simultaneously on different tasks
   - **Picture level:** Multiple process instances can encode different pictures
   - **Segment level:** Picture can be divided into segments processed in parallel

### Encoding Pipeline (High-Level Flow)

```
Source Input
    ↓
Picture Decision Process
    ↓
Resource Coordinator & Picture Manager
    ↓
Motion Estimation
    ↓
Initial Rate Control (determines initial QP)
    ↓
Picture Analysis & Preprocessing
    ↓
Encoding Loop
    ├─ Intra/Inter Prediction
    ├─ Mode Decision
    ├─ Transform & Quantization
    └─ Loop Filtering (Deblock, CDEF, Restoration)
    ↓
Entropy Coding & Packetization
    ↓
Output Bitstream
```

### Key Processes

- **Picture Manager:** Controls prediction structure and reference buffer
- **Motion Estimation:** Computes motion vectors for inter prediction
- **Rate Control:** Determines quantization parameters (QP)
- **Encoding Loop:** Main compression pipeline
- **Filtering:** In-loop deblocking, CDEF, restoration filters
- **Entropy Coder:** Final bitstream generation

### Source Partitioning

- **Picture:** Single frame of video
- **Tile:** Independently decodable rectangular region of SBs
- **Super Block (SB):** 64x64 or 128x128 luma samples
- **Block:** Result of recursive SB partitioning
- **Transform Block:** Region for transform operations

## Directory Structure

```
Source/
├── API/                  # Public API definitions (svt-av1*.h)
├── App/                  # Encoder application (SvtAv1EncApp)
└── Lib/
    ├── Codec/           # Core encoder algorithms (largest, most complex)
    ├── Globals/         # Global constants and utilities
    ├── ASM_*            # Platform-specific assembly (SSE2, AVX2, AVX512, NEON, SVE, etc.)
    └── C_DEFAULT/       # C reference implementations

test/
├── api_test/            # API tests (verify encoder with different parameters)
├── e2e_test/            # End-to-end tests (verify bitstream quality)
├── benchmarking/        # Performance benchmarking suite
└── [individual test files].cc  # Unit tests for specific modules

Docs/
├── Build-Guide.md       # Platform-specific build instructions
├── svt-av1-encoder-design.md  # Detailed architecture documentation
├── Parameters.md        # Complete encoder parameter reference
├── Appendix-*.md        # Technical deep-dives (Mode Decision, Rate Control, Transforms, etc.)
└── CommonQuestions.md   # FAQ on encoding, presets, performance tuning

Config/                  # Configuration files and presets

cmake/                   # CMake build system modules
```

## API Naming Conventions

SVT-AV1 must coexist with other AV1 libraries (libaom, libdav1d) when statically linked. Naming conventions prevent symbol conflicts:

**Public API functions:**
```c
svt_av1_enc_init_handle()    // All public API functions start with svt_av1_
svt_av1_enc_stream_header_init()
```

**Internal symbols (not publicly accessible):**
```c
svt_aom_*                    // Use svt_aom_ prefix for internal symbols
```

When porting code from other libraries, maintain these prefixes strictly.

## Key Implementation Files

**Core Encoder Control:**
- `Source/Lib/Codec/EbEncHandle.c/h` — Encoder instance management
- `Source/Lib/Codec/EbSvtAv1Enc.c` — Main encoder implementation
- `Source/Lib/Codec/EbPictureManager.c` — Picture management and prediction structure

**Motion & Rate Control:**
- `Source/Lib/Codec/EbMotionEstimation.c` — Motion estimation process
- `Source/Lib/Codec/EbRateControlProcess.c` — Rate control (QP determination)

**Encoding Loop:**
- `Source/Lib/Codec/EbCodingLoop.c` — Main compression algorithm
- `Source/Lib/Codec/EbModeDecision.c` — Mode selection
- `Source/Lib/Codec/EbInterPrediction.c` — Inter prediction
- `Source/Lib/Codec/EbIntraPrediction.c` — Intra prediction

**Transforms & Quantization:**
- `Source/Lib/Codec/EbTransforms.c` — Forward/inverse transforms
- `Source/Lib/Codec/EbQuantize.c` — Quantization

**Filtering:**
- `Source/Lib/Codec/EbDeblock.c` — Deblocking filter
- `Source/Lib/Codec/EbCdef.c` — CDEF filter
- `Source/Lib/Codec/EbRestoration.c` — Restoration filter

**Entropy Coding:**
- `Source/Lib/Codec/EbEntropyCoding.c` — Bitstream encoding
- `Source/Lib/Codec/EbBitstreamWriter.c` — Bitstream assembly

## Documentation References

- **Complete Parameter List:** `Docs/Parameters.md`
- **System Requirements:** `Docs/System-Requirements.md`
- **Encoder Design (detailed):** `Docs/svt-av1-encoder-design.md`
- **Technical Appendices:** `Docs/Appendix-*.md` (Mode Decision, Rate Control, Transforms, etc.)
- **Contribution Guide:** `Docs/Contribute.md`
- **Encoding FAQs:** `Docs/CommonQuestions.md`
- **FFmpeg Integration:** `Docs/Ffmpeg.md`

## Common Development Tasks

### Adding a New Encoder Parameter

1. Define in `Source/API/EbSvtAv1Enc.h` (public API) or control set struct
2. Add to relevant control/picture parameter structures (see `EbEncHandle.h`)
3. Handle validation in parameter validation functions
4. Update `Docs/Parameters.md`
5. Add corresponding logic in encoding pipeline
6. Add API/unit tests

### Debugging Encoder Output

The encoder can output reconstructed frames for verification:
```c
// Enable reconstruction output via API
svt_av1_enc_set_parameter(handle, SVT_AV1_ENC_PARAM_RECON_ENABLED, 1);
```

### Performance Profiling

LTO and PGO builds provide significant performance improvements (10-30%). For optimization work:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_LTO=ON -DENABLE_PGO=ON
```

Profile with `perf` or `VTune` on the output binary.

## Testing Strategy

- **Unit Tests:** Test individual algorithms in isolation (transform, quantize, SADs, etc.)
- **API Tests:** Verify encoder behaves correctly with different parameter combinations
- **E2E Tests:** Full encoding verification against reference bitstreams

Most changes should include corresponding test updates or new tests.

## Important Notes

- **In-Source Builds Not Recommended:** Use a separate `Build` directory
- **NASM Required:** x86-64 builds require NASM 2.14+ (or yasm as fallback)
- **Test Vectors:** E2E tests auto-download test vectors on first run
- **Parallel Make:** Always use `-j $(nproc)` to parallelize builds (can be slow otherwise)
- **CMake Version:** Requires CMake 3.16+
- **GCC/Clang:** GCC 5.4+ or modern Clang recommended
