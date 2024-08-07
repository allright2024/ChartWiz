import streamlit as st
from PIL import Image

# 제목 설정
st.title("이미지 및 텍스트 입력 받기")

# 이미지 업로드
uploaded_image = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

# 텍스트 입력
user_text = st.text_input("텍스트를 입력하세요")

# 버튼 추가
if st.button("출력"):
    # 이미지와 텍스트 출력
    if uploaded_image is not None:
        # 이미지 처리
        image = Image.open(uploaded_image)
        st.image(image, caption='업로드한 이미지', use_column_width=True)
    else:
        st.warning("이미지를 업로드하세요.")
    
    if user_text:
        st.write("입력한 텍스트:", user_text)
    else:
        st.warning("텍스트를 입력하세요.")
