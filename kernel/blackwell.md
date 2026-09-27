# RTX 50 series (Blackwell, sm_120): the two fixes in v1.1

Both found and proven on an RTX 5060 Ti 16GB (sm_120, driver 595.91.07, CUDA 12.8.93), Sep 26 and 27 2026. RTX 30 and 40
cards are not affected by either, and the cuda12.4 sm_86 + sm_89 bundle stays as it was.

## 1. The PTQ1_0 mat-vec kernel did not wait for its input

**What you saw:** a source build of the `bonsai2` branch up to `285542d98` on an RTX 50 card, head off, turns greedy output
into a run of `!` within the first few tokens (NaN logits). With the head on the text stays coherent but changes from run
to run, so `GGML_CUDA_BATCH_INVARIANT=1` no longer makes the head lossless.

**Why:** on sm_90 and newer the CUDA backend launches kernels with programmatic dependent launch (PDL), so a kernel can
start while the one in front of it is still running, and every consumer must call `ggml_cuda_pdl_sync()` before it reads
what its producer writes. The activation quantizer in front of the PTQ1_0 mat-vec releases the next kernel at once, and
`mul_mat_vec_ptq1_0_pt` never waited, so it could read the activation buffer before it was written. The pool hands every
mat-vec the same address, so the kernel read the previous mat-vec's activation: harmless when that was the same
activation, wrong otherwise, and NaN when stale quant bytes landed where the fp16 scales are. Below sm_90 the backend
launches without PDL and `ggml_cuda_pdl_sync()` compiles to nothing, which is why the RTX 3060, the RTX 4070 and the
prebuilt bundle never showed it: their machine code is byte-identical with and without the fix.

**The fix:** one `ggml_cuda_pdl_sync()` in the kernel after its pointer setup and before its first global load, where
`mul_mat_vec_q` has its own. No switch turned off: CUDA graphs, fusion and PDL stay on. A new `test-backend-ops` case,
`MUL_MAT_VEC_DEP`, runs a PTQ1_0 mat-vec on a norm output right after a K = 17408 PTQ1_0 mat-vec; it fails with NaN before
the fix and passes after.

**Proof on the RTX 5060 Ti:** 18 fresh server sessions, six arms (head off, n-max 1, n-max 2, with and without the knob),
each arm byte-identical across its three sessions, logprob gap 0.0000 between sessions; with the knob, head off equals
n-max 1 and n-max 2 byte for byte (lossless, as on the 3060); compute-sanitizer memcheck, initcheck, synccheck and
racecheck clean on the kernel; llama-bench tg128 52.85 tok/s fixed against 52.72 with PDL switched off on the unfixed
build (no cost). Until you run v1.1, `GGML_CUDA_PDL=0` in front of the server command avoids the race at no measured cost.

## 2. The flash-attention q4_0 V dequant lived on the stack

**What you saw:** long-context decode with a q4_0 K/V cache ran far below q8_0 on RTX 50 cards, and fell behind an RTX 3060
from about 16K tokens of context. llama-bench tg128 on the RTX 5060 Ti at 64K depth: q4_0 12.47 tok/s, q8_0 30.54, f16
35.49. Stock builds show the same curve; it is not this repo's kernel.

**Why:** one-token decode with a quantized K/V cache runs llama.cpp's flash-attention vector kernel. `dequantize_V_q4_0`
copied the 4 quant bytes into a local `int` and read them back through an `int8_t` pointer to it. nvcc 12.8 for sm_120 kept
that local on the stack: the one-query q4_0 kernel had a 256-byte stack frame and 128 generic 16-bit stores inside the loop
over the KV cache, while the sm_86 and sm_89 builds of the same source and the sm_120 q8_0 kernel had none. It was not
register pressure: at head size 64 the kernel used 199 registers and still spilled 64 bytes.

**The fix:** build the value in a register from two 2-byte loads (the q4_0 block is only 2-byte aligned) and take each
byte with a shift. Same arithmetic. The stack is gone at head sizes 64, 128 and 256, the 128 stores are gone, and the
two-query kernel the head's verify batch uses drops to the stack of its q8_0 twin.

**Proof on the RTX 5060 Ti:** `test-backend-ops -o FLASH_ATTN_EXT` 362/362 on CUDA0 against the CPU, including new decode
cases at head size 256 with 24 query heads over 4 KV heads; greedy text byte-identical before and after, with the head and
without; prefill unchanged. Decode, the same session:

| | fresh | 16K | 32K | 64K |
| --- | ---: | ---: | ---: | ---: |
| llama-bench tg128, q4_0 K/V, before | 52.82 | 28.97 | 20.16 | 12.41 |
| after | 53.59 | 43.98 | 38.28 | 29.40 |

With the MTP head at 119,489 tokens of context: 10.43 tok/s before, 22.03 after (the RTX 3060 does 13.64). The same
`dequantize_V_q4_0` code is in upstream ggml-org/llama.cpp as of Sep 27 2026, and the q4_1, q5_0 and q5_1 dequants read
their quants the same way; only q4_0 is fixed and measured here.
