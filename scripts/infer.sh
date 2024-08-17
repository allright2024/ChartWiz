CUDA_VISIBLE_DEVICES=1 python -m llava.eval.model_vqa \
    --model-path /home/work/ai-hub/Test_LLaVA/checkpoints/llava_v1.5_synatra_1.3b \
    --question-file /home/work/ai-hub/data/test/json_data/validation-summary-300.json \
    --image-folder /home/work/ai-hub/data/test/img_data \
    --answers-file ./inference_result/1.3b_summary_20k_model_output.json \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode synatra_mini

