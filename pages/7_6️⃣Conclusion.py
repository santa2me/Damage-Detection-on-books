import streamlit as st

# 페이지 제목
st.title("Conclusion")

# Conclusion 아래에 빨간 밑줄 추가
st.markdown(
    """
    <hr style="border: 1px solid #ff0000;" />  
    """, 
    unsafe_allow_html=True
)

# 4개의 동일한 컬럼 생성 (여백 포함)
col1, empty_col1, col2, empty_col2, col3, empty_col3, col4 = st.columns([1, 0.1, 1, 0.1, 1, 0.1, 1])

# 공통 스타일을 위한 변수
common_style = """
    <div style="text-align: center; line-height: 1.5; margin-bottom: 15px;">
        <h2 style="font-size: 26px; margin: 0; padding: 0;">{}</h2>
    </div>
"""

# 첫 번째 섹션 - 데이터셋 확장 및 모델 개선
with col1:
    st.markdown(common_style.format("데이터셋 확장"), unsafe_allow_html=True)
    st.write("""
    - 현재 1000장의 이미지로 트레이닝해서 mAP가 높지 않음. 이후 데이터셋을 추가하여 성능 향상 목표.
    - 추가적인 데이터 전처리 및 증강 기법 도입으로 YOLO 모델 최적화 필요.
    """)

# 두 번째 섹션 - 프리미엄 알라딘 API 도입
with col2:
    st.markdown(common_style.format("프리미엄 API"), unsafe_allow_html=True)
    st.write("""
    - 실시간 판매 시세 정보를 위해 알라딘 프리미엄 API 도입 필요.
    - 실시간 매입 가능 여부 확인 및 정확한 가격 추정 가능.
    """)

# 세 번째 섹션 - 모델 선택과 실시간 서비스의 한계
with col3:
    st.markdown(common_style.format("향후 도입 계획"), unsafe_allow_html=True)
    st.write("""
    - 실시간 서비스 구현을 위해 YOLO 모델 사용 중이나 Faster R-CNN이 더 높은 mAP 성능을 보임.
    - 향후 더 나은 서버 환경에서 Faster R-CNN 도입 예정.
    """)

# 네 번째 섹션 - 데이터베이스 연동
with col4:
    st.markdown(common_style.format("DB 연동"), unsafe_allow_html=True)
    st.write("""
    - 현재 데이터베이스 미연동 상태로, 향후 연동 시 다양한 추가 정보 제공 가능.
    - 서적 상태와 가격 정보를 실시간으로 받아 신뢰성 및 편의성 향상 기대.
    """)

# 페이지 전체 스타일 조정
st.markdown(
    """
    <style>
    .block-container {
        padding-left: 80px;
        padding-right: 80px;
    }
    h2 {
        font-size: 30px;
        margin: 0;
        padding: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


