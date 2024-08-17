import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import warnings
import shutil
import shelve
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, Pix2StructForConditionalGeneration, AutoProcessor
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model import *
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import math
import time 
import hashlib


import streamlit as st

def get_file_hash(file):
    file.seek(0)
    file_bytes = file.read()
    file.seek(0)  # 파일 포인터를 처음으로 되돌립니다.
    return hashlib.md5(file_bytes).hexdigest()

st.set_page_config(page_title="ChartWiz", layout="wide")

col1, col2, col3 = st.columns([5, 5, 2])

with col2:
    st.image("/home/work/ai-hub/Test_LLaVA/llava/eval/chartwiz_logo.png", width=350)

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="/home/work/ai-hub/Test_LLaVA/checkpoints/llava_synatra_description_summary_table_QA_deplot")
parser.add_argument("--model-base", type=str, default='/home/work/ai-hub/pretrained_model/maywell/Synatra-7B-v0.3-dpo')
parser.add_argument("--conv-mode", type=str, default="v1")
parser.add_argument("--num-chunks", type=int, default=1)
parser.add_argument("--chunk-idx", type=int, default=0)
parser.add_argument("--temperature", type=float, default=0)
parser.add_argument("--top_p", type=float, default=None)
parser.add_argument("--num_beams", type=int, default=1)
args = parser.parse_args()

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


def get_response(uploaded_file, user_text):
    # print(user_text)
    processed_image = image_processor(images=[image], text="Generate underlying data table of the figure below:", return_tensors="pt",  max_patches=512)
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], "<image>" + "\n" + user_text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        
    input_ids = input_ids.to(device='cuda', non_blocking=True)
    # input_ids = torch.tensor(input_ids).unsqueeze(0)
    input_ids = input_ids.clone().detach().unsqueeze(0)
    processed_image = processed_image.to('cuda')
    inputs = {}
    inputs["flattened_patches"] = processed_image["flattened_patches"].clone().detach().unsqueeze(0)
    inputs["attention_mask"] = processed_image["attention_mask"].clone().detach().unsqueeze(0)
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
    results = results.replace("\n", "<br/>")
    time.sleep(1)
    return results

def stream_response(text):
    # 이 함수는 스트리밍 응답을 처리합니다.
    with st.chat_message("assistant"):
        st.markdown(text)

# 페이지 레이아웃 설정

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.image_received = False
    


if "image_hash" not in st.session_state:
    st.session_state["image_hash"] = None

if "pending_response" not in st.session_state:
    st.session_state["pending_response"] = False


# 중앙에 제목 배치

# 화면을 두 개의 열로 나눔 (왼쪽: 이미지, 오른쪽: 채팅)
col1, col2 = st.columns([3, 5])

# 파일 업로드 처리
with col1:
    uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"], key="image_uploader")

    if uploaded_file is not None:
        current_image_hash = get_file_hash(uploaded_file)

        # 새로운 이미지가 업로드되었을 때 대화 기록 초기화
        if st.session_state["image_hash"] != current_image_hash:
            st.session_state["messages"] = []
            st.session_state["image_hash"] = current_image_hash
            st.session_state["pending_response"] = False  # 응답 대기 상태 초기화

        image = Image.open(uploaded_file)
        st.image(image, caption='업로드된 이미지', use_column_width=True)
    else:
        st.session_state["image_hash"] = None

with col2:
    # 채팅 기록을 표시
    if 'messages' not in st.session_state:
        st.session_state.messages = []
                    # <img src='https://cdn-icons-png.flaticon.com/512/1077/1077012.png' style='vertical-align: middle; width: 30px; height: 30px;' />
                    # <img src='https://cdn-icons-png.flaticon.com/512/6873/6873405.png' style='vertical-align: middle; width: 30px; height: 30px;' />

    def display_chat():
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style='text-align: right; margin-bottom: 30px; margin-right: 50px;'>
                    <div style='background-color: #f0f0f0; border-radius: 10px; padding: 10px; display: inline-block;'>
                        <span style='font-size:20px'>{msg['content']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='text-align: left; margin-bottom: 30px; margin-left: 50px; width: 70%'>
                    <div style = 'color: ; border-radius: 10px; padding: 10px; display: inline-block;'>
                        <span style='font-size: 20px'>{msg['content']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # 채팅 기록 표시
    display_chat()

    # 채팅 기록이 있을 때 입력 필드 아래로 스크롤하기
    st.write("###")


    # 입력 필드와 전송 버튼을 담을 컨테이너
    with st.container():
        col3, col1, col2 = st.columns([1, 15, 2])
    
    with col1:
        prompt = st.text_input("질문을 입력하세요:", key="input", placeholder="여기에 입력하세요...", label_visibility="collapsed")
    
    with col2:
        if st.button("전송") and not st.session_state["pending_response"]:
            if prompt:
                st.session_state.messages.append({'role': 'user', 'content': prompt})
                st.session_state["pending_response"] = True  # 응답 대기 상태 설정

                response = get_response(uploaded_file, prompt)
                
                st.session_state.messages.append({'role': 'assistant', 'content': response})
                st.session_state["pending_response"] = False  # 응답 생성 완료
                st.experimental_rerun()
            else:
                st.write("질문을 입력해 주세요.")