import streamlit as st
from PIL import Image

st.header("Structure Diagram", divider="red")

# 파일 경로 설정 (streamlit 스크립트 파일이 있는 위치 기준)
image_path = 'figures/structure_diagram.png'

# Load image
image = Image.open(image_path)

# Display image to fit most of the width
st.image(image, caption='Structure Diagram', use_column_width=True)