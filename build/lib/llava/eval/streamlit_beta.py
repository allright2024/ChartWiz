import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, Pix2StructForConditionalGeneration, AutoProcessor
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model import *
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import math

import streamlit as st

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="/home/work/ai-hub/Test_LLaVA/checkpoints/llava_v1.5_7b_synatra_summ_QA")
parser.add_argument("--model-base", type=str, default='/home/work/ai-hub/pretrained_model/maywell/Synatra-7B-v0.3-dpo')
parser.add_argument("--conv-mode", type=str, default="v1")
parser.add_argument("--num-chunks", type=int, default=1)
parser.add_argument("--chunk-idx", type=int, default=0)
parser.add_argument("--temperature", type=float, default=0)
parser.add_argument("--top_p", type=float, default=None)
parser.add_argument("--num_beams", type=int, default=1)
args = parser.parse_args()

st.markdown("""
<style>
.stButton > button {
    display: block;
    margin-left: auto;
    margin-right: 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):
    # kwargs = {"device_map": device_map}
    kwargs = {}
    kwargs['torch_dtype'] = torch.bfloat16

    # Load LLaVA model
    if model_base is None:
        raise ValueError('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
    if model_base is not None:
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            model_base,
            model_max_length=2048,
            padding_side="right", 
            use_fast=False)
        print('Loading LLaVA from base model...')
        if "qwen" in model_base.lower():
            model = LlavaQwen2ForCausalLM.from_pretrained(
                model_base, 
                low_cpu_mem_usage=False,  # 끄기 
                config=lora_cfg_pretrained, 
                **kwargs)
        else:
            model = LlavaMistralForCausalLM.from_pretrained(
                model_base, 
                low_cpu_mem_usage=False,  # 끄기 
                config=lora_cfg_pretrained, 
                **kwargs)
        
        print("Model from pretrained")
        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        print('Loading additional LLaVA weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        else:
            # this is probably from HF Hub
            from huggingface_hub import hf_hub_download
            def load_from_hf(repo_id, filename, subfolder=None):
                cache_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    subfolder=subfolder)
                return torch.load(cache_file, map_location='cpu')
            non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)
        model.to(device)


        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')

    image_processor = None

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.bfloat16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len

model_path = os.path.expanduser(args.model_path)
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

st.title("Chart QA")

uploaded_image = st.file_uploader("차트 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="업로드한 차트 이미지", use_column_width=True)
    user_text = st.text_input("차트에 대한 질문을 입력하세요")

if st.button("출력"):
    if uploaded_image is not None and user_text:
        processed_image = image_processor(images=[image], text="Generate underlying data table of the figure below:", return_tensors="pt",  max_patches=512)
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], "<image>" + "\n" + user_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        processed_image = processed_image.to('cuda')
        inputs = {}
        inputs["flattened_patches"] = torch.tensor(processed_image["flattened_patches"]).unsqueeze(0)
        inputs["attention_mask"] = torch.tensor(processed_image["attention_mask"]).unsqueeze(0)
        attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device='cuda')
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask = attention_mask,
                pad_token_id = tokenizer.pad_token_id,
                eos_token_id = tokenizer.eos_token_id,
                images=inputs,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1636,
                use_cache=True)
            
        results = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        st.write(results)
    elif uploaded_image is not None:
        st.warnings("instruction을 입력하세요")
    elif uploaded_image is None and user_text:
        st.warnings("이미지를 업로드 하세요.")
    else:
        st.warnings("텍스트를 입력하고 이미지를 업로드하세요.")