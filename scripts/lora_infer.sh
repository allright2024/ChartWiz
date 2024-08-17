CUDA_VISIBLE_DEVICES=1 python -m llava.eval.model_vqa_lora \
    --model-path /home/work/ai-hub/Test_LLaVA/checkpoints/llava_synatra_7b_mlp5x_summary_20k \
    --model-base /home/work/ai-hub/pretrained_model/maywell/Synatra-7B-v0.3-dpo \
    --question-file /home/work/ai-hub/data/test/json_data/validation-summary-300.json \
    --image-folder /home/work/ai-hub/data/test/img_data \
    --answers-file ./inference_result/5x_summary_20k_model_output.json \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode vicuna_v1

