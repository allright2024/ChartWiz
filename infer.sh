CUDA_VISIBLE_DEVICES=1 python -m llava.eval.model_vqa_lora \
    --model-path /home/work/ai-hub/Test_LLaVA/checkpoints/llava-v1.5-7b-qwen2-lora \
    --model-base Qwen/Qwen2-7B-Instruct \
    --question-file /home/work/ai-hub/data/test/json_data/validation-qa-nogpt-20.json \
    --image-folder /home/work/ai-hub/data/test/img_data \
    --answers-file ./inference_result/ans_1000k.json \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode qwen_2
