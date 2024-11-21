import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="NIPA-GOOGLE-2팀",
    page_icon="🏠",
    layout="wide",
)

# --- HEADER SECTION ---

with st.container():
    st.subheader("NIPA_Google_3기")
    st.title("중고서적 판매를 위한 등급분류 프로젝트 📖")

st.markdown(
    "<h1 style='text-align:right; font-style: italic; font-size:20px; border-bottom:1px solid red;'>2팀: 김예원, 김형수, 윤지영, 이혜민, 홍유경</h1>",
    unsafe_allow_html=True
)

# st.markdown(
#     """
#     <style>
#         div[data-testid="column"]:nth-of-type(1)
#         {
#             border:1px solid red;
#         } 

#         div[data-testid="column"]:nth-of-type(2)
#         {
#             border:1px solid blue;
#             text-align: end;
#         } 
#     </style>
#     """,unsafe_allow_html=True
# )

# col1, col2 = st.columns(2)

# with col1:
#     """
#     ### 이름, 이름, 이름, 이름, 이름
#     """
# with col2:
#     """
#     ### 이름, 이름, 이름, 이름, 이름
    
#     """
#     st.button("➡️")

# with st.align('right'):
#     st.write("이름, 이름, 이름, 이름, 이름")

st.sidebar.success("Select a page above")

# --- BODY SECTION ---
st.markdown(
    """
    ### AI/Computer Vision 기반 중고 서적 상태 분석 및 챗봇 서비스
    **목표:**
    - 알라딘의 중고 서적 매입 시스템을 개선하여, 사용자 경험을 극대화하고, 자동화된 서적 상태 평가 시스템을 도입해 효율성을 높임.

    **주요 기능:**
    - AI 이미지 분석: Object Detection/Instance Segmentation 모델을 활용한 중고 서적의 상태 자동 판정.
    - 알라딘 API 연동: ISBN 코드 입력을 받아 책 정보 제공.
    - AI 챗봇 서비스: Gemini 챗봇 API를 통한 실시간 사용자 응대.

    **타겟:**
    - 알라딘 (기업), 특히 중고 서적 매입 시스템을 개선하고자 하는 알라딘의 사업 부서.
    
"""
)

st.markdown(
    "<h1 style='text-align:right; font-style: normal; font-size:100px; '>📚</h1>",
    unsafe_allow_html=True
)


st.markdown(
    """
    <style>
    /* Make the main content and body take up full height */
    html, body, .css-1gkbhjr, .css-1d391kg {
        height: 100%;
        min-height: 100%;
    }

    /* Adjust the main content area (right side) */
    .css-1outpf7 {
        height: 100vh;  /* Use full viewport height */
        padding: 10px;  /* Adjust as needed */
    }

    /* Remove any bottom margins or padding */
    .css-1outpf7 > div {
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;  /* Ensures footer or bottom content sticks */
    }

    /* Optional: Adjust the sidebar height */
    .css-1d391kg {
        height: 100vh;  /* Full height for sidebar */
    }

    </style>
    """,
    unsafe_allow_html=True
)


# Custom CSS to reduce the sidebar size and margins
# st.markdown(
#     """
#     <style>
#         /* Make the sidebar narrower */
#         .css-1d391kg {
#             width: 10px;  /* Adjust the width as needed */
#         }

#         /* Reduce the padding/margins of the main page */
#         .css-1lcbmhc {
#             padding: 1rem 1rem 1rem 1rem; /* Adjust the padding as needed */
#         }
#     </style>
#     """,
#     unsafe_allow_html=True
# )


# Inject custom CSS to make the layout full-height

# st.write("# 중고서적 판매를 위한 등급분류 서비스 👋")