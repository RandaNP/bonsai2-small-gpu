#!/bin/bash
# 16GB cards with the MTP head and the vision tower (merged file from ../graft/ plus the mmproj): the full 262K window, 15.1 GB after load and 15.3 GB with an image in the prompt, n-max 1 = the lossless speed pick.
GGML_CUDA_BATCH_INVARIANT=1 "${LLAMA_SERVER:-llama-server}" -m "${1:-Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf}" --mmproj "${2:-Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf}" -c 262144 --spec-type draft-mtp --spec-draft-n-max 1 -ngl 99 -fa on -np 1 -ctk q4_0 -ctv q4_0 --jinja --reasoning-effort medium --temp 1.0 --top-p 0.95 --top-k 20 --host 127.0.0.1 --port 8899
