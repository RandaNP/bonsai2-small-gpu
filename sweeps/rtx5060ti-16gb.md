# Ternary Bonsai 2 27B + grafted Qwen 3.8 MTP head, RTX 5060 Ti 16GB

All numbers from one RTX 5060 Ti 16GB (Blackwell, sm_120, 16,311 MiB, 180 W limit), driver 595.91.07, CUDA 12.8.93, gcc
13.3.0, Sep 26 to 28 2026. `--parallel 1` unless stated, thinking off, q4_0 K/V cache, flash attention on. Client tok/s
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

## v1.1 to the full window, up to 8 at once, four agents, board power (Sep 28)

The same bundle (`b10757-ff414120c`) against stock `prism-b10743-adfffbe` built from source, same card and driver. The
16GB line for depth and power: fat file + mmproj, `-c 262144 -np 1 -ngl 99 -fa on -ctk q4_0 -ctv q4_0 --jinja
--reasoning-effort medium`; head n-max 1 = `serve-16gb-mtp-vision.sh` as shipped (the knob, draft-mtp n-max 1), head off =
the same `llama-server`, file and knob without the spec flags, stock = `prism-b10743`, no knob, no spec flags.

### Depth to the full window

`deep_probe.py`: a llama.cpp docs corpus (116,053 characters, the PrismML fork's README and its build, new-model and
multimodal docs) repeated to the target length, "write a 250 word summary", 300 tokens, thinking off. Median client
tok/s of 2 runs: run 1 prefills the whole prompt from nothing, run 2 sends it again from the cache. Depth = the server's
`/tokenize` count (its `prompt_n` adds 12 template tokens).

| depth (tokens) | stock, head off | v1.1, head off | v1.1, head n-max 1 | n-max 1 acceptance (run 1, run 2) | prefill, run 1, head n-max 1 |
| ---: | ---: | ---: | ---: | --- | ---: |
| 18,176 | 24.53 | 43.30 | 48.37 | 0.633, 0.693 | 931.3 tok/s |
| 38,774 | 16.63 | 35.63 | 39.74 | 0.759, 0.652 | 855.0 tok/s |
| 76,693 | 10.46 | 27.02 | 29.01 | 0.703, 0.728 | 724.1 tok/s |
| 114,575 | 7.63 | 21.75 | 22.36 | 0.738, 0.646 | 626.3 tok/s |
| 119,489 | 7.39 | 21.27 | 21.37 | 0.703, 0.624 | 615.0 tok/s |
| 149,999 | | | 18.19 | 0.674, 0.615 | 556.9 tok/s |
| 199,998 | | | 14.93 | 0.684, 0.684 | 476.9 tok/s |
| 250,000 | | | 12.34 | 0.713, 0.605 | 416.8 tok/s |
| 261,000 | **3.73** | **12.35** | **12.07** | 0.652, 0.699 | 405.8 tok/s |

Head off and stock ran the base depths and the deepest point only. At 261,000 tokens (`prompt_n` 261,012, plus 300 out,
inside 262,144 with 832 to spare) v1.1 is 3.31x stock with the head off (12.35 / 3.73) and 3.24x with the head; the gap
grows with depth from 1.77x at 18,176 tokens. Head off and stock prefill the 261,000 tokens at the same 427.7 tok/s
(610.2 s), the head arm at 405.8 tok/s (643.2 s).

The head's gain shrinks with depth: +11.7% at 18,176 tokens, +7.4% at 76,693, +0.5% at 119,489 and -2.3% at 261,000,
while its acceptance stays between 0.605 and 0.759. Past about 120K tokens the head-off arm is as fast and holds
2,532 MiB less: head n-max 1 15,070 MiB after load and 15,128 after every depth, head off 12,546 / 12,596, stock
12,586 / 12,598. The window is allocated at load, so the full-window request adds nothing.

### Several at once, to 8

`-np N -c 32768*N`, q4_0 K/V, no mmproj, N streaming requests of 256 tokens, four short prompts (cycled past 4),
temperature 0, thinking off, 2 rounds, after a 1-request reference on the same server. Head n-max 1 = the fat file with
the knob; head off and stock = the original file, no knob. Aggregate = all generated tokens over the span from the first
token of any request to the last.

| np | arm | 1-request reference | aggregate, round 1 / 2 | mean | per request, both rounds | VRAM after load |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | v1.1, head n-max 1 | 75.05 | 87.76 / 92.68 | 90.22 | 45.14 to 52.60 | 9,108 MiB |
| 2 | v1.1, head off | 54.20 | 83.05 / 85.46 | 84.25 | 41.51 to 42.57 | 7,396 MiB |
| 2 | stock, head off | 42.81 | 48.97 / 49.79 | 49.38 | 24.46 to 24.80 | 7,436 MiB |
| 4 | v1.1, head n-max 1 | 77.33 | 124.45 / 124.95 | **124.70** | 31.00 to 36.96 | 11,370 MiB |
| 4 | v1.1, head off | 52.53 | 111.93 / 112.93 | 112.43 | 27.88 to 28.12 | 9,104 MiB |
| 4 | stock, head off | 42.75 | 58.45 / 59.40 | 58.92 | 14.61 to 14.82 | 9,144 MiB |
| 6 | v1.1, head n-max 1 | 73.50 | 136.93 / 145.11 | **141.02** | 24.01 to 27.90 | 13,632 MiB |
| 8 | v1.1, head n-max 1 | does not load | | | | out of memory |
| 8 | v1.1, head off | 52.34 | 141.27 / 141.09 | **141.18** | 17.57 to 17.60 | 12,518 MiB |
| 8 | stock, head off | 42.11 | 133.94 / 140.65 | 137.30 | 16.79 to 17.52 | 12,558 MiB |

np 8 with the head (`-c 262144`) fails while building the MTP draft context (a 130 MiB compute buffer, `cudaMalloc
failed: out of memory`); np 6 with the head loads at `-c 196608`. At 8 streams the totals meet (1.03x) because both
builds leave the mat-vec path: stock sends PTQ1_0 through its mat-vec up to 7 tokens per step, v1.1 through its own up
to 4, and at 8 both run the same MMQ tile path (read from the source, not profiled). At np 2 and 4 with the head off
v1.1 is 1.71x and 1.91x stock.

### Four agents, each with its own 60K document

`-np 4 -c 262144` (65,536 tokens per slot), head n-max 1 (the concurrency head arm). Each slot gets its own document of
about 60,000 tokens (cuts of four parts of the PrismML source tree: docs, the server, `src`, the CUDA backend), prefilled
alone at 768.5 to 772.0 tok/s; then all four decode 256 tokens at once from their cached documents, temperature 0, 2
rounds: **41.10 / 43.21 tok/s aggregate**, 10.24 to 11.11 tok/s per agent, acceptance 0.660 to 0.735, 14,732 MiB after
load. Four short prompts on the same arm do 124.70 tok/s; four 60K documents keep a third of it.

### Board power

`nvidia-smi` every 500 ms, board power only, mean over the samples at 80% utilization or more. Fresh decode = `probe.py`
on the 16GB line (3 prompts x 3 runs, 400 tokens).

| state | board power | speed | tok/s per W | kWh per 1M tokens |
| --- | ---: | ---: | ---: | ---: |
| idle, no server | 10.11 W | | | |
| the 16GB line loaded, no request | 20.31 W | | | |
| decode, fresh, head n-max 1 | 144.50 W | 67.4 tok/s | 0.466 | 0.60 |
| decode, fresh, head off | 143.70 W | 54.2 tok/s | 0.377 | 0.74 |
| decode at 261,000 tokens, head n-max 1 (run 1 / 2) | 156.13 / 154.58 W | 11.91 / 12.24 tok/s | 0.076 / 0.079 | 3.64 / 3.51 |
| decode at 261,000 tokens, head off (run 1 / 2) | 179.34 / 177.80 W | 12.34 / 12.36 tok/s | 0.069 / 0.070 | 4.04 / 4.00 |
| decode at 261,000 tokens, stock (run 1 / 2) | 95.12 / 92.73 W | 3.73 / 3.73 tok/s | 0.039 / 0.040 | 7.08 / 6.91 |
| 6 at once, head n-max 1 | 124.83 W | 141.02 tok/s | 1.130 | 0.25 |
| 8 at once, head off | 129.71 W | 141.18 tok/s | 1.088 | 0.26 |
| 8 at once, stock | 127.90 W | 137.30 tok/s | 1.074 | 0.26 |
| four agents at 60K each, head n-max 1 (round 1 / 2) | 140.66 / 141.95 W | 41.10 / 43.21 tok/s | 0.292 / 0.304 | 0.95 / 0.91 |

Prefills of 150,000 tokens and more run at the 180 W limit (178.09 to 178.98 W). At 261,000 tokens the head-off decode
draws more than the head decode; measured, cause not investigated.

Against the Sep 27 window on this card: fresh 67.4 tok/s with the head (was 67.3) and 54.2 without (53.4), 39.74 and
21.37 tok/s at 38,774 and 119,489 tokens (39.52 and 20.99), four at once 124.45 / 124.95 with the head (127.08 /
129.23) and 58.45 / 59.40 on stock (58.34 / 58.92), 15,070 MiB after loading the 16GB line (same). Not run: head off and
stock at 149,999 to 250,000 tokens, np 8 with the head at a smaller window, the four agents with the head off or on
stock, a second fresh prefill at any depth.

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
