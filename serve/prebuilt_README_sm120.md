# bonsai2-small-gpu v1.1, Linux x86-64, CUDA 12.8, sm_120 (RTX 50 series)

A self-contained build of the PrismML llama.cpp fork for Ternary Bonsai 2 27B on RTX 50 series cards, carrying:

- the PTQ1_0 mat-vec kernel for small batches (PrismML-Eng/llama.cpp pull request 218) and the `GGML_CUDA_BATCH_INVARIANT=1`
  switch that makes a token's logits bit-identical whether it is decoded alone or inside a batch of up to 4, so speculative
  decoding is lossless; 5 and more columns take the MMQ tile path;
- the Blackwell fix for that kernel: it now waits on programmatic dependent launch before it reads its activations. Without
  it, head-off decode on RTX 50 cards turns into a run of `!` within a few tokens. A regression test comes with it;
- a fix in llama.cpp's flash-attention vector kernel: nvcc 12.8 for sm_120 kept the q4_0 V dequant's scratch value on the
  stack inside the loop over the KV cache, so long-context decode with a q4_0 KV cache ran at less than half speed. On an
  RTX 5060 Ti 16GB, llama-bench tg128 at 64K depth goes from 12.41 to 29.40 tok/s with the same output byte for byte;
- underneath, PrismML's release `prism-b10743-adfffbe`, with pull request 214 (the branch-free PTQ1_0 MMQ tile loader, 2x
  prefill), 216 (the 4-column GDN warp layout on all Ampere and newer NVIDIA) and 205 (the Hadamard fix for the qwen35 MTP
  draft graph, which lets the grafted Qwen 3.8 MTP head draft from the trunk's own embedding table).

Source: branch `bonsai2` of github.com/sudoingX/llama.cpp (tag `bonsai2-v1.1`), commit ff41412: `prism-b10743-adfffbe` plus the ten
commits of the `pr-ptq1-mmv` series, the PDL wait and its test, and the q4_0 V dequant fix and its tests. Build:
`cmake -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON -DLLAMA_CURL=OFF
-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DCMAKE_BUILD_TYPE=Release`, CUDA 12.8, GCC 13, Ubuntu 24.04.

## Runs on

- NVIDIA GPUs of compute capability 12.0: the GeForce RTX 50 series, desktop and laptop, and the RTX PRO Blackwell cards.
  Tested on an RTX 5060 Ti 16GB. RTX 30 and 40 series: take the cuda12.4 sm86-sm89 bundle. DGX Spark (GB10, compute
  capability 12.1) is not covered by this build.
- NVIDIA driver 570 or newer. No CUDA toolkit needed: `lib/` carries the CUDA 12.8 runtime and cuBLAS.
- Linux x86-64 with glibc 2.38 or newer and a libstdc++ from GCC 13 or newer (Ubuntu 24.04, Fedora 39 or newer; on Ubuntu 22.04 or Debian 12 build the branch from source) and a CPU with AVX2. On Windows, run it inside WSL2 (see the model card).

## Quick start, 16 GB card (head, vision, the full 262144 window)

```
hf download sudoingx/Ternary-Bonsai-2-27B-PTQ1_0-MTP-GGUF Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf --local-dir ~/models/bonsai2-27b
hf download prism-ml/Ternary-Bonsai-2-27B-gguf Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf --local-dir ~/models/bonsai2-27b
tar xzf bonsai2-small-gpu-linux-x64-cuda12.8-sm120-ff41412.tar.gz && cd bonsai2-small-gpu-linux-x64-cuda12.8-sm120-ff41412
./serve-16gb-mtp-vision.sh
```

Then talk to `http://127.0.0.1:8899` (OpenAI-compatible, `/v1/chat/completions`, tools work with `--jinja`, images through the
chat UI or `image_url` content parts).

| script | files | context | VRAM after load (nvidia-smi, RTX 5060 Ti 16GB, no display attached) | measured on the RTX 5060 Ti 16GB |
| --- | --- | --- | --- | --- |
| `serve-16gb-mtp-vision.sh` | merged file + mmproj | 262144 | 15,070 MiB (15,326 after an image request) | 67.3 tok/s median over a code, a prose and a bash prompt (code 71.0, prose 59.5, bash 67.3), greedy output byte-identical to the same build without the head; 39.5 tok/s with 38,774 tokens of context and 21.0 with 119,489 |
| `serve-12gb-mtp.sh` | merged file | 131072 | 10,634 MiB | for 12 GB cards such as the RTX 5070 |
| `serve-12gb.sh` | `Ternary-Bonsai-2-27B-PTQ1_0.gguf` (original) | 262144 | 11,694 MiB | llama-bench tg128 54.5 tok/s fresh, 29.5 tok/s at 64K depth, pp512 1,061.9 tok/s |
| `serve-8gb.sh` | `Ternary-Bonsai-2-27B-PTQ1_0.gguf` (original) | 98304 | 8,002 MiB on an RTX 3060 | for 8 GB cards such as the RTX 5060; on a card that also drives a display run it with `CTX=65536` |

On a 16 GB card that also drives a display, start the 16 GB script with `CTX=131072` (11,486 MiB with the head and the
mmproj, 11,748 after an image request).

Every script takes `MODEL`, `HOST`, `PORT` and `CTX` (and `MMPROJ` for the vision script) from the environment and passes
extra arguments through to `llama-server`. Set `MODEL=/path/to/file.gguf` if your models live elsewhere.

All scripts pass `--reasoning-effort medium`. The GGUF chat template defaults `reasoning_effort` to `xhigh`, which adds a
"think carefully" system line; with it the model can spend a whole 4,096-token answer budget thinking and return nothing.
Thinking still counts against the client `max_tokens`; use 8,192 or more for code, or pass `--reasoning-effort default` to
get the template's behaviour back.

The merged file is the original PTQ1_0 file with the `blk.64` MTP tensors of Qwen3.8-27B appended (byte-exact,
reversible); the tools that make it are in github.com/sudoingX/bonsai2-small-gpu (`graft/`).

## Files

- `bin/llama-server`, `bin/llama-bench`, `bin/llama-cli`: the fork's binaries, RPATH set to `lib/`.
- `lib/`: libggml*, libllama, libmtmd and the CUDA 12.8 runtime (libcudart, libcublas, libcublasLt).
- `LICENSES/`: llama.cpp MIT (also the PrismML fork's license), NVIDIA CUDA EULA for the redistributed runtime.
- `SHA256SUMS`: `sha256sum -c SHA256SUMS` from this directory.

## Licenses

llama.cpp and the PrismML fork are MIT licensed (LICENSES/llama.cpp-LICENSE.txt). The CUDA runtime and cuBLAS
libraries in `lib/` are NVIDIA's, redistributed under the CUDA Toolkit EULA, Attachment A
(LICENSES/NVIDIA-CUDA-EULA.txt). Ternary Bonsai 2 27B is PrismML's, Apache 2.0; the MTP head is from Qwen3.8-27B,
Apache 2.0. Neither model file is in this archive.
