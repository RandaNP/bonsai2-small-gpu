# Ternary Bonsai 2 27B, RTX 3000 Ada Generation Laptop GPU 8GB

All numbers from one NVIDIA RTX 3000 Ada Generation Laptop GPU (8 GB, sm_89) on Windows 11 +
WSL2 (Ubuntu 22.04.5, kernel 5.15.167.4-microsoft-standard-WSL2, i9-13900H, 20 threads),
driver 595.95, CUDA UMD 13.2, one session, 2026-09-30. The GPU drives no display
(`nvidia-smi` idle: 0 MiB used). Numbers only from runs done on this machine.

This card is **not** the RTX 3060 12GB and not the 3060 Ti 8GB. It is an Ada (sm_89)
laptop part: 8 GB on a 128-bit bus at 8001 MHz max (about 256 GB/s peak), a 50 W power cap
(`nvidia-smi power.max_limit = 50.00 W`) and no display attached. Same VRAM as the 3060 Ti
row (`rtx3060ti-8gb.md`), but its stock decode is already at the card's practical bandwidth
wall, which changes what the PTQ1_0 kernel is worth here.

## Binaries

- **bundle**: `bonsai2-small-gpu-linux-x64-cuda12.4-sm86-sm89-285542d` (the `bonsai2`
  branch of `sudoingX/llama.cpp`, `285542d98`, build 10738): `bdc23b56b` + the ten
  `pr-ptq1-mmv` commits including the PTQ1_0 mat-vec kernel and
  `GGML_CUDA_BATCH_INVARIANT`. Confirmed by the `mul_mat_vec_ptq1_0_pt` SASS and the sm_86
  **and** sm_89 cubins in `lib/libggml-cuda.so.0`. cuda12.4 runtime.
- **prism**: the source build at `/home/randon_m/bonsai/llama.cpp`, branch `prism` at
  `bdc23b56b` (build 10728) — the base, **without** the kernel (`ggml/src/ggml-cuda/mmvq-ptq1_0.cuh`
  is absent). This is the "stock" arm.

## Files

- `Ternary-Bonsai-2-27B-PTQ1_0.gguf`, 5,946,648,928 bytes — base, no MTP.
- `Ternary-Bonsai-2-27B-PTQ1_0-MTP.gguf`, 6,297,658,656 bytes, sha256
  `09bce6c2eb6f862a5ac8b654c3b4d53f7600e915a4030986af09647a275a1514` — the **lean** MTP
  graft (`blk.64.*` only, no duplicate embedding; the fork reads the trunk's own table via
  the Hadamard fix). Built from the unsloth Qwen3.8-27B donor.
- `Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf`, 7,012,820,512 bytes, sha256
  `83a396ee218c36e5ed88205eccb940a71549d9a72a3d020cc2713f94a78f70f0` — the **fat** MTP file
  shipped by `bonsai2-small-gpu` (`graft/recipe.txt` step 3); carries a second, Q4_K
  `blk.64.nextn.embed_tokens.weight` (682 MiB) for forks without the Hadamard fix.

## A/B on the same card: bundle vs prism (llama-bench)

Same model (base file), same flags, same session, only the binary differs.

```
llama-bench -m Ternary-Bonsai-2-27B-PTQ1_0.gguf -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -p 512 -n 128 -r 3 -d 0,16384,65536
```

| test | prism `bdc23b56b` | bundle `285542d98` | delta |
| --- | ---: | ---: | ---: |
| pp512 | 499.08 ± 9.83 | 498.29 ± 9.45 | -0.2% |
| tg128 | 26.97 ± 0.64 | **27.77 ± 1.41** | **+3.0%** |
| pp512 @ d16384 | 432.18 ± 21.45 | 433.08 ± 19.79 | +0.2% |
| tg128 @ d16384 | 22.49 ± 1.21 | 22.57 ± 1.22 | +0.4% |
| pp512 @ d65536 | 260.33 ± 4.31 | 267.86 ± 5.50 | +2.9% |
| tg128 @ d65536 | 14.18 ± 0.29 | 14.21 ± 0.26 | +0.2% |

**The PTQ1_0 mat-vec kernel is worth about +3% here, not the +53% it is worth on the RTX
3060 12GB.** That is the expected result, not a broken build. The kernel removes idle lanes
from a mat-vec that is compute-bound on Ampere; this card is already memory-bound. The
stock decode of both cards is the same number (26.3 tok/s on the 3060 12GB, 27.0 here), but
the 3060 has 360 GB/s to spend and this card has about 256 GB/s, so once the kernel stops
bottlenecking on lanes, this card has far less bandwidth to convert into tokens:

| card | peak BW | stock tg128 | at peak | kernel tg128 | at peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX 3060 12GB (sm_86) | 360 GB/s | 26.3 | 43% | 40.5 | 67% |
| RTX 3000 Ada Laptop (sm_89) | ~256 GB/s | 27.0 | 63% | 27.8 | 65% |

## The real win on this card: lean MTP at a 64K window

Same bundle binary, `-ngl 99 -fa on -np 1 -ctk q4_0 -ctv q4_0`, `graft/probe.py` client
tok/s (thinking off, 400 max tokens, TTFT excluded, 3 prompts x 3 runs; mean / median of the
nine runs). `GGML_CUDA_BATCH_INVARIANT=1` unless noted; `draft-mtp` n-max as listed.

### 65536 window

| arm | file | spec | inv | probe mean | probe median | acceptance | VRAM peak |
| --- | --- | --- | --- | ---: | ---: | --- | ---: |
| base | base | off | on | 25.8 | 25.9 | — | 7,293 MiB |
| lean | lean | n-max 1 | on | 30.3 | 31.0 | 1.77–1.85 | 7,919 MiB |
| **lean** | lean | **n-max 2** | **on** | **31.9** | **33.5** | **2.29–2.46** | 7,937 MiB |
| lean | lean | n-max 2 | off | 32.1 | 32.0 | 2.29–2.51 | 7,937 MiB |
| fat | fat | n-max 2 | on | 12.2 | 11.9 | 2.31–2.55 | 7,939 MiB |

Per-prompt medians, lean n-max 2 invariant: code 37.8, prose 24.7, bash 33.5 against base
25.9 / 25.9 / 25.9. **+24% mean, +29% median, and lossless.**

The acceptance is high (mean accepted length 2.30, i.e. two drafted tokens accepted 0.73 and
0.57 of the time), so the draft head matches this ternary trunk well. `n-max 2` beats
`n-max 1` by about 5%. Invariance is free at this window (31.9 vs 32.1 without it), and it
is what makes the greedy output identical:

```
graft/tools/verify_identity.py run ... off   # base server
graft/tools/verify_identity.py run ... on    # lean MTP, n-max 2, GGML_CUDA_BATCH_INVARIANT=1
graft/tools/verify_identity.py compare ... off on
code   IDENTICAL (1001 bytes, off vs on)
prose  IDENTICAL (825 bytes, off vs on)
bash   IDENTICAL (1073 bytes, off vs on)
```

Server `print_timing` (decode only, excludes prefill) in the same session: off code 25.06,
prose 24.92, bash 25.06 tok/s; on code 38.21, prose 29.55, bash 32.34 tok/s (+33% mean).

### Stretch window: 73728

The base decode is flat from 64K to 96K, so a window above 64K is only a question of where
MTP stops paying. A context scan (lean, n-max 2, invariant, short probe) is **not
monotonic**: fast at 65536–73728 and at 81920; slow at 77824 (12.6) and from 86016 up
(18–20). The full 3x3 probe re-ranks the candidates:

| window | probe mean | probe median | code / prose / bash |
| ---: | ---: | ---: | --- |
| 65536 | 31.9 | 33.5 | 37.8 / 24.7 / 33.5 |
| 69632 | 30.2 | 29.7 | 37.1 / 23.3 / 29.7 |
| **73728** | **30.7** | **30.9** | 37.4 / 24.5 / 30.9 |
| 81920 | 26.3 | 25.6 | 32.0 / 21.6 / 25.6 |
| 98304 | 12.1 | 12.6 | 14.9 / 9.0 / 12.6 |

73728 (18 × 4096) is the pick above 64K: **+19% median over the 25.9 base**, stable across
three cold loads (medians 30.9 / 30.6 / 30.6), lossless at that window (the same three
byte-identical greedy completions), and a 1,500-token generation completed with no OOM.
81920 is not worth it (25.6, level with base) despite a fast code-prompt reading. Because
77824 and 86016 sit next to fast windows and are reproducibly slow, scan the window you
intend to use before trusting a number off this table.

### 98304 window — MTP does not survive it

| arm | file | spec | inv | probe mean | probe median | VRAM peak |
| --- | --- | --- | --- | ---: | ---: | ---: |
| base | base | off | off | 25.5 | 25.5 | 7,937 MiB |
| base | base | off | on | 26.2 | 26.2 | 7,937 MiB |
| lean | lean | n-max 1 | on | 14.9 | 15.3 | 7,939 MiB |
| lean | lean | n-max 2 | on | 12.1 | 12.6 | 7,939 MiB |
| lean | lean | n-max 2 | off | 11.9 | 11.5 | 7,939 MiB |

The base decode is unchanged at 98304 (25.5, same as 65536), but MTP is **less than half**
speed even with high acceptance. It is not the invariance switch (11.5 without it), not the
file (lean), and not memory pressure the flags can relieve: at 98304, quantizing the draft
K/V (`-ctkd q4_0 -ctvd q4_0`) gave 14.3 and shrinking the batch (`-b 256 -ub 64`) gave 7.8
against 15.4 baseline. Contexts between 64K and 98K are not monotonic either (see the
stretch section), so this reads as an allocation-size effect on this driver (WSL2, 595.95)
rather than the kernel. The stretch recommendation is 73728, not 81920 or 98304.

**Practical rule for an 8 GB Ada: serve a 64K window with the lean MTP head (n-max 2,
`GGML_CUDA_BATCH_INVARIANT=1`) at about 33 tok/s; a 73728 window is the stretch at about 31
tok/s; a 96K window is 25.5 without the head.** This mirrors the 3060 Ti row's finding that
window length is the dominant cost, not any kernel.

## The fat MTP file does not fit an 8 GB card

The `bonsai2-small-gpu` MTP file carries a duplicate Q4_K embedding table
(`blk.64.nextn.embed_tokens.weight`, 682 MiB) that only exists for forks without the
Hadamard fix. At 65536 the load logs read:

| file | CUDA0 model buffer | probe mean | probe median |
| --- | ---: | ---: | ---: |
| lean | 5,730.08 MiB | 31.9 | 33.5 |
| fat | 6,412.11 MiB | 12.2 | 11.9 |

The extra 682 MiB on the GPU pushes the 8 GB card past capacity, decode spills to system
memory and the same-session speed collapses by 2.7x, reproducibly (11.9 and 12.0 on two
runs). The lean graft — the same head, letting the MTP graph read the trunk's own embedding
table via the Hadamard fix — is both smaller and 2.7x faster here. This is the row's main
practical result.

## Served VRAM

`nvidia-smi --query-gpu=memory.used --format=csv,noheader`, after load and during the probe:

| file | 65536 | 73728 | 98304 |
| --- | ---: | ---: | ---: |
| base | 7,273 / 7,293 MiB | — | 7,917 / 7,937 MiB |
| lean MTP | 7,921 / 7,937 MiB | 7,937 / 7,939 MiB | 7,919 / 7,939 MiB |
| fat MTP | 7,931 / 7,939 MiB | — | — |

(carried = after load / during probe). 73728 loads and peaks at the same ~7,937 MiB as 64K;
98304 fits with about 250 MiB to spare and loads cleanly.

## Reproducibility

- The bundle-vs-prism `llama-bench` A/B and the probe table were each run once in this
  session; the probe medians agree with the existing `../bonsai/bench_mtp.py` runs
  (MTP 31.08 vs no-MTP 24.70, +25.9%) to within 2%.
- The fat MTP collapse was measured twice (11.9 and 12.0 median) and is reported as one of
  the findings, not an outlier.
- The 73728 stretch was measured three times with a cold load each time: medians 30.9 /
  30.6 / 30.6, and its identity check is exact. The window scan around it (77824 and 86016
  slow, 81920 fast-then-mediocre) was each run at least twice.
- The identity check is exact (three byte-identical greedy completions) at both 65536 and
  73728.

## Notes

- **This is a WSL2 run.** Ubuntu 22.04.5 under `microsoft-standard-WSL2`, driver 595.95
  exposed from Windows, CUDA UMD 13.2. The bundle is the cuda12.4 build and needs no toolkit.
- The bundle was built with `GGML_CUDA_FA=ON` and CUDA graphs; `-fa on` and `-fa 1` both map
  to `fa = 1` in `llama-bench`.
- No `--no-mmap`, no power-limit change, no overclock. The 50 W cap is the laptop's.
- The PTQ1_0 mat-vec kernel barely helps at depth either (`tg128 @ d65536` +0.2%): the kernel
  targets the compute-bound small-batch path, and this card is bandwidth-bound at every
  depth.
- Use `GGML_CUDA_BATCH_INVARIANT=1` with the head: it is free at 64K and it is what makes the
  speculative decode lossless.
