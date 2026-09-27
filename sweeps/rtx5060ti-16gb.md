# Ternary Bonsai 2 27B + grafted Qwen 3.8 MTP head, RTX 5060 Ti 16GB

All numbers from one RTX 5060 Ti 16GB (Blackwell, sm_120, 16,311 MiB, 180 W limit), driver 595.91.07, CUDA 12.8.93, gcc
13.3.0, Sep 26 and 27 2026. `--parallel 1` unless stated, thinking off, q4_0 K/V cache, flash attention on. Client tok/s
are `probe.py` (streamed content deltas, time to first token excluded), medians of 3 runs per prompt unless stated.

Binaries, both built from source with `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120` (sm_120a) and
`cmake --build build -j 32`:

- stock: PrismML llama.cpp fork at `prism-b10685-7dffb15` (build 10685). The tag is the full name; `git checkout prism-b10685`
  fails.
- ours: github.com/sudoingX/llama.cpp `bonsai2` at `285542d98` plus a one-line fix to the PTQ1_0 mat-vec kernel (a
  `ggml_cuda_pdl_sync()` before it reads its activations; commit `dc359ed46`, test `64bf73608`, reported as build 10740).
  Without the fix this branch decodes a run of `!` on RTX 50 cards with the head off and is not reproducible with it on
  (`../kernel/blackwell.md`). Called "the fixed build" below; the q4_0 V dequant fix of Sep 27 comes on top of it in v1.1.

Files: `Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf` (7,012,820,512 bytes, the fat file with the head),
`Ternary-Bonsai-2-27B-PTQ1_0.gguf` (5,946,648,928 bytes), `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` (629,246,976 bytes), all
sha256-checked against the Hugging Face listings.

## Identity and reproducibility (the fixed build, fat file, 131072, three fresh server sessions per arm)

| arm | code / prose / bash, same bytes in all three sessions |
| --- | --- |
| head off, `GGML_CUDA_BATCH_INVARIANT=1` | `2ff5fe93` / `d3e4b03c` / `3b15237f` |
| head n-max 1, `GGML_CUDA_BATCH_INVARIANT=1` | identical to head off |
| head n-max 2, `GGML_CUDA_BATCH_INVARIANT=1` | identical to head off |
| head off, no knob | `9948638b` / `95ebe1d1` / `982cfd55`, logprobs equal to 4 decimals across sessions |

With the knob the head is lossless on this card at n-max 1 and 2.

## llama-bench r=3, original file (`-ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -p 512 -n 128`)

| build | pp512 tok/s | tg128 tok/s |
| --- | ---: | ---: |
| stock | 470.74 +/- 7.12 | 42.93 +/- 0.23 |
| ours | 1023.45 +/- 30.08 | 52.85 +/- 0.09 |


## probe.py fresh, fat file, 131072

| arm | code | prose | bash | median | VRAM after the probe |
| --- | ---: | ---: | ---: | ---: | ---: |
| stock, head off | 42.8 | 43.0 | 42.6 | 42.8 | 8,796 MiB |
| ours, head off | 52.4 | 52.4 | 52.2 | 52.3 | 8,794 MiB |
| ours, head n-max 1 | 72.6 | 58.5 | 67.8 | 67.8 | 10,680 MiB |
| ours, head n-max 2 | 80.0 | 56.9 | 67.3 | 67.3 | 10,830 MiB |

Our arms with `GGML_CUDA_BATCH_INVARIANT=1`.

## The head at depth (`deep_probe.py`, fat file, 131072, median of 2 runs)

| depth (tokens) | stock, head off | ours, head off | ours, head n-max 1 | n-max 1 acceptance |
| ---: | ---: | ---: | ---: | --- |
| 18,176 | 24.62 | 27.93 | 34.99 | 0.693, 0.628 |
| 38,774 | 16.67 | 18.17 | 23.47 | 0.689, 0.627 |
| 76,693 | 10.48 | 11.07 | 14.96 | 0.753, 0.665 |
| 114,575 | 7.64 | 7.92 | 10.76 | 0.713, 0.684 |
| 119,489 | 7.39 | 7.65 | 10.22 | 0.693, 0.643 |

Prefill of the filled prompt: 119,501 tokens in 189.5 s on ours (630 tok/s), 327.4 s on stock (365 tok/s). Decode at depth
falls behind an RTX 3060 12GB, ours from about 16K tokens and stock from about 32K (llama-bench tg128 at 64K depth: stock
11.77 against 14.27, ours 12.31 against 17.77 tok/s on the 3060); the cost that grows is attention over the K/V cache,
the same on both builds, and the head adds 34% at 119K. The K/V section below finds why, and v1.1 fixes it.

## Several requests at once (the fixed build, Sep 27)

`llama-server -ngl 99 -fa on -c 32768*np -np <np> -ctk q4_0 -ctv q4_0 --jinja`, np simultaneous streaming requests, 256
tokens each, thinking off, four different prompts, 2 rounds; aggregate = all generated tokens over the span from the
first token of any request to the last token of the last. Original file for the head-off arms, the fat file with
`GGML_CUDA_BATCH_INVARIANT=1` for the head arm.

| arm | one request | np 2 aggregate (round 1 / 2) | np 4 aggregate (round 1 / 2) | per request at np 4 | VRAM after load, np 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| stock, head off | 42.16 | 48.24 / 48.95 | 58.72 / 59.31 | 14.62 / 14.77 (median) | 9,144 MiB |
| ours, head off | 51.78 | 79.82 / 81.86 | 101.26 / 105.49 | 26.06 / 26.27 (median) | 9,104 MiB |
| ours, head n-max 1 | 75.26 | 86.52 / 90.85 | **121.87 / 123.58** | 30.77 to 35.18 | 11,370 MiB |

The head runs with several slots: every slot drafts, the drafts go into the shared batch, and against the same fat file
with the head off (np 2 79.14 / 81.39, np 4 107.39 / 106.90) it adds 9% to 16% in aggregate. On an RTX 3060 12GB the
same client gave ours 56.33 / 65.91 and stock 32.41 / 32.51 at np 2 / 4.

## K/V cache type at depth (the fixed build, Sep 27)

llama-bench tg128, original file, head off, `-ngl 99 -fa 1 -p 512 -n 128 -r 2`:

| K/V | fresh | 16K | 32K | 64K |
| --- | ---: | ---: | ---: | ---: |
| f16 | 55.55 | 47.62 | 43.22 | 35.49 |
| q8_0 | 54.91 | 45.22 | 39.33 | 30.54 |
| q4_0 | 53.07 | 29.02 | 20.17 | 12.47 |

q4_0 fell far behind q8_0 with the same kernel, grid and occupancy: the sm_120 build of the flash-attention vector kernel
kept the q4_0 V dequant's scratch value on the stack inside the loop over the KV cache (256 bytes of stack and 128
generic stores per step on this card, none on the RTX 30 and 40 builds of the same source, none in the q8_0 kernel).
PrismML's release `prism-b10743-adfffbe` gives the same stock curve (43.36 / 25.78 / 18.64 / 11.78 against b10685's
42.81 / 25.97 / 18.68 / 11.80 in the same session). Details: `../kernel/blackwell.md`.

## v1.1: the q4_0 V dequant fix (Sep 27)

The dequant builds the value in a register and takes each byte with a shift. Same arithmetic: greedy text byte-identical
before and after, with the head and without, and `test-backend-ops -o FLASH_ATTN_EXT` 362/362 on this card, including new
head-size-256 decode cases.

llama-bench tg128, q4_0 K/V, head off, the same session:

| | fresh | 16K | 32K | 64K |
| --- | ---: | ---: | ---: | ---: |
| before | 52.82 | 28.97 | 20.16 | 12.41 |
| after | 53.59 | **43.98** | **38.28** | **29.40** |

The head at depth (`deep_probe.py`, fat file, 131072, n-max 1, the knob, median of 2 runs):

| depth (tokens) | before | after | RTX 3060 12GB, same method |
| ---: | ---: | ---: | ---: |
| 38,774 | 23.71 | **39.00** | 27.47 |
| 119,489 | 10.43 | **22.03** | 13.64 |

Prefill unchanged. q4_0 now decodes within 4% of q8_0 at half the K/V bytes, so the full 262K window with the head and
the vision tower keeps its speed on 16 GB.

## v1.1, the cuda12.8 sm120 bundle (Sep 27)

The release tarball `bonsai2-small-gpu-linux-x64-cuda12.8-sm120-ff41412` (`bonsai2` v1.1, build 10757), extracted into a
clean directory and run under `env -i` with its own scripts, the way a new owner runs it. Stock is PrismML's
`prism-b10743-adfffbe` built from source, same card, same session.

| check | result |
| --- | --- |
| `serve-16gb-mtp-vision.sh` (fat file, mmproj, 262144, head n-max 1, the knob) | loads in 4 s, 15,070 MiB, 15,326 after an image request |
| identity, head off vs n-max 1, and three head sessions | byte-identical (`2ff5fe93` / `d3e4b03c` / `3b15237f`, the same bytes as every earlier build) |
| probe, head n-max 1 | 67.3 tok/s median (code 71.0, prose 59.5, bash 67.3) |
| probe, head off, same config | 53.4 tok/s median |
| probe, prism-b10743, head off, same config | 42.0 tok/s median |
| llama-bench tg128 / pp512, original file | 54.49 / 1,061.88 tok/s fresh (r=3), 29.49 / 595.47 at 64K depth (r=2) |
| the head with 38,774 / 119,489 tokens of context | 39.52 / 20.99 tok/s (runs 39.76 and 39.28, 19.95 and 22.04) |
| four requests at once, 32768 each | head n-max 1 127.08 / 129.23 tok/s aggregate, 31.66 to 36.20 per request; prism-b10743 head off 58.34 / 58.92 |
| the shop-page image | 1,048 prompt tokens at 629 tok/s, answer at 69.61 tok/s, 53 of 60 drafts accepted, all six facts right |

## What fits in 16GB (largest window that loads and completes a 1K-token request)

| configuration | window | VRAM after load | after the request | after an image request |
| --- | ---: | ---: | ---: | ---: |
| original, head off | 262144 | 11,694 MiB | 11,746 MiB | |
| fat file, head n-max 1 | 262144 | 14,218 MiB | 14,276 MiB | |
| fat file, head n-max 1, mmproj Q8_0 | 262144 | 15,070 MiB | 15,128 MiB | 15,332 MiB |

The head, vision and the full 262,144 window fit together with 979 MiB to spare.

## Vision at 262144 (the shop-page test, fat file + mmproj, fresh server per run, 3 runs)

| arm | image prefill (1,048 tokens) | decode tok/s | drafts accepted | page facts |
| --- | --- | ---: | --- | --- |
| head n-max 1 | 1.82 to 1.83 s | 67.61 to 67.87 | 53 of 60 | 6/6 every run |
| head off | 1.74 to 1.78 s | 48.41 to 50.11 | | 6/6 every run |

The answer text is the same with and without the head.

## Install from zero, following the public links

Clone ours 24.6 s, configure 5.0 s, build 149.2 s at `-j 32`, fat file download 66.5 s, first `/health` 4.1 s, first token
0.353 s after the request: 250.0 s of steps on the Hugging Face card path. Friction on the way, all four fixed in the v1.1
docs: the stock tag needs its full name, the card's serve line defaulted to port 8080, the hub scripts needed `build/bin`
on `PATH`, and neither doc said RTX 50 cards need CUDA 12.8 or newer.
