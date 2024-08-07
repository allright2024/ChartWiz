CUDA_VISIBLE_DEVICES=1 python -m llava.eval.model_vqa_lora \
    --model-path /home/work/ai-hub/Test_LLaVA/checkpoints/llava_v1.5_7b_synatra_summ_QA_new_deplot \
    --model-base /home/work/ai-hub/pretrained_model/maywell/Synatra-7B-v0.3-dpo \
    --question-file /home/work/ai-hub/data/test/json_data/validation-qa-nogpt-50_v2.json \
    --image-folder /home/work/ai-hub/data/test/img_data \
    --answers-file ./inference_result/ans_50_with_summ_QA_new_deplot.json \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode qwen_2
