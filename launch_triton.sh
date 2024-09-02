git clone https://github.com/triton-inference-server/tensorrtllm_backend.git

cp -r all_models/inflight_batcher_llm/* triton_model_repo/
cp /path/to/medusa-vicuna-1.3/trtllm/engine triton_model_repo/tensorrt_llm/1

# Create above Dockerfile in this directory

docker build -t tritonserver_custom:latest .

docker run --rm -it --net host --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
-v /path/to/tensorrtllm_backend/repo:/tensorrtllm_backend \
-v /path/to/medusa/vicuna-7b-v1.3/tokenizer:/tokenizer \
tritonserver_custom:latest