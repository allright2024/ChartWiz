CUDA_VISIBLE_DEVICES=1 python -m llava.eval.model_vqa_lora \
    --model-path /home/work/ai-hub/deplot_llava_zero/checkpoints/llava-v1.5-100k-7b-lora \
    --model-base maywell/Synatra-7B-v0.3-dpo \
    --question-file /home/work/ai-hub/data/test/json_data/validation-qa-nogpt-20.json \
    --image-folder /home/work/ai-hub/data/test/img_data \
    --answers-file ./inference_result/ans_1000k.json \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode vicuna_v1
