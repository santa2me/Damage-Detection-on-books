import google.generativeai as genai 
import streamlit as st
from PIL import Image
import numpy as np
import requests
import xml.etree.ElementTree as ET
import json
from datetime import datetime
from ultralytics import YOLO  # YOLOv8 로드
from ultralytics import settings
from dotenv import load_dotenv
import os
from typing import Optional  # Optional 추가
from typing import Tuple

# 환경 변수 로드
load_dotenv()
my_api_key = os.getenv('GENAI_API_KEY')
ttbkey = os.getenv('ALADIN_API_KEY')

# Gemini API 및 알라딘 API 설정
genai.configure(api_key=my_api_key)

# Default book data in case API call fails or no results are found
DEFAULT_BOOK_DATA = {
    'title': '소년이 온다',
    'author': '한강',
    'pubDate': 'N/A',
    'description': 'N/A.',
    'isbn': '9788936434120',
    'isbn13': 'N/A',
    'priceSales': 'N/A',
    'priceStandard': 'N/A',
    'publisher': '창비',
    'cover': 'N/A',
    'salesPoint': 'N/A',
    'customerReviewRank': 'N/A'
}

# 알라딘 API에서 도서 정보를 가져오는 함수
def get_book_data_by_isbn(isbn: str) -> Optional[dict]:
    url = f"http://www.aladin.co.kr/ttb/api/ItemLookUp.aspx?ttbkey={ttbkey}&itemIdType=ISBN&ItemId={isbn}&output=xml&Version=20131101&OptResult=usedList"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # HTTP 에러 발생 시 예외 처리

        root = ET.fromstring(response.content)
        namespace = {'ns': 'http://www.aladin.co.kr/ttb/apiguide.aspx'}
        items = root.findall('.//ns:item', namespace)

        if not items:
            st.write("도서 정보를 찾을 수 없습니다. 기본값을 사용합니다.")
            return DEFAULT_BOOK_DATA

        item = items[0]  # 첫 번째 결과 사용
        book_data = {
            'title': item.find('ns:title', namespace).text or 'N/A',
            'author': item.find('ns:author', namespace).text or 'N/A',
            'pubDate': item.find('ns:pubDate', namespace).text or 'N/A',
            'description': item.find('ns:description', namespace).text or 'N/A',
            'isbn': item.find('ns:isbn', namespace).text or 'N/A',
            'isbn13': item.find('ns:isbn13', namespace).text or 'N/A',
            'priceSales': item.find('ns:priceSales', namespace).text or 'N/A',
            'priceStandard': item.find('ns:priceStandard', namespace).text or 'N/A',
            'publisher': item.find('ns:publisher', namespace).text or 'N/A',
            'cover': item.find('ns:cover', namespace).text or 'N/A',
            'salesPoint': item.find('ns:salesPoint', namespace).text or 'N/A',
            'customerReviewRank': item.find('ns:customerReviewRank', namespace).text or 'N/A'
        }
        return book_data
    except requests.RequestException as e:
        st.write(f"API 요청 중 오류가 발생했습니다: {e}. 기본값을 사용합니다.")
        return None

# Gemini API를 사용해 책 설명을 생성하는 함수
def generate_book_description(book_data: dict) -> str:
    prompt = (
        f"책 제목: {book_data['title']}\n"
        f"저자: {book_data['author']}\n"
        # f"출판일: {book_data['pubDate']}\n"
        # f"책 설명: {book_data['description']}\n"
        # f"판매 가격: {book_data['priceSales']}원\n"
        f"\n책 제목과 저자를 이용해 검색하여 책에대한 최신 정보를 알려주세요."
    )

    model = genai.GenerativeModel('gemini-1.5-flash-002')  # Gemini 모델 사용
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(prompt)

    return response.text

# 도서 정보와 설명을 JSON 파일로 저장하는 함수
# def save_description_to_json(book_data: dict, book_description: str) -> None:
#     current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#     file_name = f"book_description_{current_time}.json"

#     book_data["generated_description"] = book_description

#     with open(file_name, "w", encoding="utf-8") as file:
#         json.dump(book_data, file, ensure_ascii=False, indent=4)
#     print(f"설명과 책 정보가 {file_name} 파일에 저장되었습니다.")

# YOLOv8 모델 로드 함수
@st.cache_resource
def load_yolo_model():
    return YOLO("small_best.pt")

yolo_model = load_yolo_model()

# front = Image.open("figures/front_cover.png")
# back = Image.open("figures/back_cover.png")
# spine = Image.open("figures/spine.png")
# page_edges = Image.open("figures/page_edges.png")
# 샘플 이미지 경로 설정
SAMPLE_IMAGES = {
    "front": "figures/front_cover.png",
    "back": "figures/back_cover.png",
    "spine": "figures/spine.png",
    "page_edges": "figures/page_edges.png"
}

# 객체 탐지 및 시각화 함수
def detect_and_annotate(image: Image) -> [Image, bool]:
    image_np = np.array(image)  # PIL 이미지를 NumPy 배열로 변환
    results = yolo_model(image_np)  # 탐지 수행

    detected = len(results[0].boxes) > 0  # 탐지 여부
    annotated_image = results[0].plot()  # 탐지 결과 시각화
    return Image.fromarray(annotated_image[..., ::-1]), detected  # RGB 변환


# YOLOv8 객체 탐지 함수
def yoloinf(image: Image) -> Image:
    image = np.array(image)
    results = yolo_model(image)
    annotated_img = results[0].plot()  # 바운딩 박스 그리기
    return Image.fromarray(annotated_img[..., ::-1])  # RGB 포맷으로 변환하여 반환


# 품질 평가 로직 추가
def generate_quality_evaluation(detection_results: list) -> str:
    detected_count = sum(int(result) for result in detection_results)
    if detected_count >= 3:
        return "중: 4장의 이미지 중 3장에서 손상이 탐지되었습니다."

    prompt = (
        f"탐지된 이미지 수: {detected_count}/4\n"
        "책의 품질 등급을 평가해 주세요:\n"
        "- 최상: 새것에 가까운 책\n"
        "- 상: 약간의 사용감이 있으나 깨끗한 상태\n"
        "- 중: 변색 및 약간의 손상 있음\n"
        "- 매입불가: 심한 손상 또는 오염\n"
    )
    model = genai.GenerativeModel('gemini-pro')
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(prompt)

    return response.text

# JSON으로 저장
# def save_description_to_json(book_data: dict, book_description: str) -> None:
#     current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#     file_name = f"book_description_{current_time}.json"
#     book_data["generated_description"] = book_description

#     with open(file_name, "w", encoding="utf-8") as file:
#         json.dump(book_data, file, ensure_ascii=False, indent=4)
#     st.write(f"{file_name}에 저장되었습니다.")

# main() 함수로 코드를 구조화

# Streamlit 앱
def main():
    st.title("Gemini-Bot")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📚 책 정보 조회"):
            st.session_state["show_isbn_input"] = True
            st.session_state["show_upload"] = False


    with col2:
        if st.button("⭐ 판매 등급 판정"):
            st.session_state["show_upload"] = True
            st.session_state["show_isbn_input"] = False

    if st.session_state.get("show_isbn_input", False):
        isbn = st.text_input("ISBN을 입력하세요")
        if isbn:
            book_data = get_book_data_by_isbn(isbn)
            if book_data:
                book_description = generate_book_description(book_data)
                st.write("생성된 설명:", book_description)
                #save_description_to_json(book_data, book_description)

    if st.session_state.get("show_upload", False):
        st.write("아래 샘플 사진을 참고해 이미지를 업로드하세요:")

        # 샘플 이미지를 2개씩 한 줄에 표시
        cols = st.columns(2)
        with cols[0]:
            st.image(SAMPLE_IMAGES["front"], caption="앞표지 예시", width=200)
            st.image(SAMPLE_IMAGES["spine"], caption="책등 예시", width=200)
        with cols[1]:
            st.image(SAMPLE_IMAGES["back"], caption="뒷표지 예시", width=200)
            st.image(SAMPLE_IMAGES["page_edges"], caption="책배 예시", width=200)

        # 사용자 이미지 업로드
        front_cover = st.file_uploader("앞표지", type=["jpg", "png", "jpeg"])
        back_cover = st.file_uploader("뒷표지", type=["jpg", "png", "jpeg"])
        spine = st.file_uploader("책등", type=["jpg", "png", "jpeg"])
        page_edges = st.file_uploader("책배", type=["jpg", "png", "jpeg"])

        if front_cover and back_cover and spine and page_edges:
            st.success("이미지가 모두 업로드되었습니다.")

            # 각 이미지 탐지 및 시각화 수행
            detected_front, front_found = detect_and_annotate(Image.open(front_cover))
            detected_back, back_found = detect_and_annotate(Image.open(back_cover))
            detected_spine, spine_found = detect_and_annotate(Image.open(spine))
            detected_page_edges, edges_found = detect_and_annotate(Image.open(page_edges))

            # 탐지된 이미지 표시
            st.image(detected_front, caption="앞표지 탐지 결과")
            st.image(detected_back, caption="뒷표지 탐지 결과")
            st.image(detected_spine, caption="책등 탐지 결과")
            st.image(detected_page_edges, caption="책배 탐지 결과")

            # 탐지 결과 저장 (boolean 값만 포함)
            detection_results = [front_found, back_found, spine_found, edges_found]

            # 품질 평가 수행
            st.write("Gemini 품질 평가 요청 중...")
            quality_result = generate_quality_evaluation(detection_results)
            st.write("품질 평가 결과:", quality_result)
        else:
            st.warning("4장의 이미지를 모두 업로드해 주세요.")


# main 함수 실행
if __name__ == "__main__":
    main()
