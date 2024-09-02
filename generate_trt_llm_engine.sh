git clone https://github.com/NVIDIA/TensorRT-LLM.git

cd TensorRT-LLM/examples/medusa

virtualenv env
source env/bin/activate

pip install -r requirements.txt

sudo apt-get update && sudo apt-get install git-lfs
git lfs install

git clone https://huggingface.co/lmsys/vicuna-7b-v1.3

# Convert and Build Medusa decoding support for vicuna-13b-v1.3 with 4-way tensor parallelism.
python convert_checkpoint.py --model_dir ./vicuna-7b-v1.3 \
                            --medusa_model_dir /path/to/trained/medusa-heads-model-directory \
                            --output_dir ./tllm_checkpoint_1gpu_medusa \
                            --dtype float16 \
                            --num_medusa_heads 4 \
                            --tp_size 4 \
                            --workers 4

# Build TRT-LLM Engine
trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_medusa \
             --output_dir ./tmp/medusa/7B/trt_engines/fp16/1-gpu/ \
             --gemm_plugin float16 \
             --speculative_decoding_mode medusa \
             --max_batch_size 4

# Sample Medusa decoding using vicuna-7b-v1.3 model with 1 GPU
# For straming mode, add --streaming
python ../run.py --engine_dir ./tmp/medusa/7B/trt_engines/fp16/1-gpu/ \
                 --tokenizer_dir ./vicuna-7b-v1.3/ \
                 --max_output_len=100 \
                 --medusa_choices="[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]" \
                 --temperature 1.0 \
                 --input_text "Once upon"