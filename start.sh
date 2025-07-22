#!/bin/bash
set -e

export HF_HOME="/workspace/hf"
# export HF_TOKEN="<your-token-here>"
export TRANSFORMERS_OFFLINE=0 


vllm serve dst19/jess-voice-merged \
--max-model-len 2048 \
--gpu-memory-utilization 0.6 \
--host 0.0.0.0 \
--port 9191 \
--max-num-batched-tokens 4096 \
--max-num-seqs 24 \
--enable-chunked-prefill \
--disable-log-requests \
--block-size 16 \
--enable-prefix-caching &


uvicorn main_v2:app \
--host 0.0.0.0 \
--port 9090 \
--workers 12 \
--loop uvloop \
--http httptools \
--log-level warning \
--access-log

# trtllm-serve serve dst19/jess-voice-merged \
# --backend pytorch \
# --host 0.0.0.0 \
# --port 9090 \
# --max_batch_size 8 \
# --max_num_tokens 8192 \
# --max_seq_len 2048 \
# --max_beam_width 1 \
# --kv_cache_free_gpu_memory_fraction 0.8 \
# --trust_remote_code

