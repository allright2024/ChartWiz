# ChartWiz

LLaVA with Deplot Encoder for ChartQA 

LLaVA 모델에서 기존의 CLIP 대신 Deplot의 Encoder를 연결한 모델입니다.

### 환경 세팅
1. Anaconda3로 conda 가상환경 생성
2. python 버전 : 3.10.12
3. cuda 버전 : 12.2
4. cudnn 버전 : 8.9.2
(nvidia docker file : https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-09.html 여기서 docker파일을 이용하면 cuda, cudnn버전을 맞출 필요없이 환경 세팅이 가능합니다.)

### 라이브러리 설치
1. pip install -e .
2. pip install -e ".[train]"
3. pip install flash-attn --no-build-isolation

### 데이터셋 구성 방법
AI-hub 차트 이미지-텍스트 데이터 쌍 데이터 준비

- train
    - img_data
        - img_etc_mix
            - img_files(.jpg)
        - img_etc_radial
        - img_horizontal bar_100per accumulation
        - img_horizontal bar_accumulation
        - img_horizontal bar_standard
        - img_line_standard
        - img_pie_standard
        - img_vertical bar_100per accumulation
        - img_vertical bar_accumulation
        - img_vertical bar_standard
    - text_data
        - text_etc_mix
            - json_files(.json)
        - text_etc_radial
        - text_horizontal bar_100per accumulation
        - text_horizontal bar_accumulation
        - text_horizontal bar_standard
        - text_line_standard
        - text_pie_standard
        - text_vertical bar_100per accumulation
        - text_vertical bar_accumulation
        - text_vertical bar_standard
- test
    - img_data
        - img_etc_mix
            - img_files(.jpg)
        - img_etc_radial
        - img_horizontal bar_100per accumulation
        - img_horizontal bar_accumulation
        - img_horizontal bar_standard
        - img_line_standard
        - img_pie_standard
        - img_vertical bar_100per accumulation
        - img_vertical bar_accumulation
        - img_vertical bar_standard
    - text_data
        - text_etc_mix
            - json_files(.json)
        - text_etc_radial
        - text_horizontal bar_100per accumulation
        - text_horizontal bar_accumulation
        - text_horizontal bar_standard
        - text_line_standard
        - text_pie_standard
        - text_vertical bar_100per accumulation
        - text_vertical bar_accumulation
        - text_vertical bar_standard

### Pretrain 데이터 준비
LLaVA에서 Pretrain 단계는 Projection Layer를 학습하는 단계이며 데이터를 처리할 때도 이미지에 대한 Instruction 없이 Image 자체에 대하여 학습을 합니다.  
그래서 차트 이미지에 대하여 일반적인 학습을 하기 위해서는 차트 상세 설명 데이터 혹은 차트 테이블 데이터로 학습을 진행해야 하는데, 저희는 테이블 데이터로 학습하기로 정했습니다.  
테이블 데이터를 구축하기 위해서는 AI-hub 데이터에서 annotations로 부터 정보를 추출해야합니다.  
코드는 preprocess폴더의 make_table_data.py에 있습니다.