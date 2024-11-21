import streamlit as st
from PIL import Image

# 페이지 제목
st.title("Introduction")

# --- Load Asset --- 
img_aladin_app = Image.open("figures/aladin_app.png")
img_car = Image.open("figures/car.png")
img_output = Image.open("figures/output.png")

# Introduction 아래에 밑줄 추가
st.markdown(
    """
    <hr style="border: 1px solid #ff0000;" />  <!-- 빨간색 밑줄 -->
    """, 
    unsafe_allow_html=True
)

# 레이아웃에 빈 컬럼 추가하여 간격을 넓히기
col1, empty_col1, col2, empty_col2, col3 = st.columns([1, 0.1, 1, 0.1, 1])  # 사이에 빈 공간(0.5)을 두고 3개의 컬럼 생성


# 첫 번째 섹션 - 기존 시스템의 비효율성 
with col1:
    st.image(img_aladin_app)
    st.markdown(
        """
        <div style="text-align: center;">
            <h2 style="font-size: 26px;">기존 시스템의 비효율성</h2>
        </div>
        """, unsafe_allow_html=True
    )
    st.write("""
    - 현재 알라딘 중고 서점은 서적 상태에 대해 직접 방문하거나 택배로 발송한 이후에 매입 가격 결정
    - 판매자의 입장에서 번거롭고 불편함
    - 추가 문의 사항은 즉각적인 응대가 불가능하며, 이메일을 통해서만 소통 가능
    """)

# 두 번째 섹션 - AI 기술의 확장 가능성
with col2:
    st.image(img_car)
    st.markdown(
        """
        <div style="text-align: center;">
            <h2 style="font-size: 27px;">AI 기술의 확장 가능성</h2>
        </div>
        """, unsafe_allow_html=True
    )
    st.write("""
    - 손상의 종류를 포함한 데미지를 정확히 식별할 수 있다면, 이 기술은 다른 플랫폼에서의 높은 확장성 
    - AI를 통한 이미지 데이터 분석은 서적 외에도 전자 제품, 가구 등 다양한 중고 물품에도 적용 가능 
    - 쏘카의 AI 기반 손상 탐지 기술 타 플랫폼 사례 
    """)

# 세 번째 섹션 - 중고책 시장의 성장
with col3:
    st.image(img_output)
    st.markdown(
        """
        <div style="text-align: center;">
            <h2 style="font-size: 30px;">중고책 시장의 성장</h2>
        </div>
        """, unsafe_allow_html=True
    )
    st.write("""
    - 도서 정가제로 인한 신규 도서 가격 부담으로 인해 사람들이 중고책에 대한 관심이 증가
    - 최근 한강 작가의 노벨 문학상 수상 이후 중고서적 거래에 대한 수요 증가
    - 환경 보호를 위한 지속 가능성의 인식이 높아지며, 친환경적인 소비로서 주목되는 중고책 소비
    """)

# 페이지 양쪽에 여백 추가
st.markdown(
    """
    <style>
    .block-container {
        padding-left: 80px;  /* 양쪽 여백을 더 넓힘 */
        padding-right: 80px;
    }

    h2 {
        font-size: 30px;  /* 헤더 폰트 크기 */
    }
    </style>
    """,
    unsafe_allow_html=True
)
