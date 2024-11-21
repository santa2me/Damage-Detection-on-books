
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
from typing import Optional, Tuple  # Optional 추가 20241021
# from typing import Tuple 20241021
from PIL import Image # 20241021 
import re # 20241021 

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

def clean_response_text(response) -> str:
    """
    입력된 텍스트에서 마크다운 포맷(** 또는 __ 등) 제거.
    문자열이 아닌 경우 빈 문자열 반환.
    """
    if not isinstance(response, str):
        # response가 None이거나 문자열이 아니면 빈 문자열 반환
        return ""

    # ** 또는 __로 감싸진 텍스트를 제거
    cleaned_text = re.sub(r"\*\*(.*?)\*\*", r"\1", response)  # **bold** 처리
    cleaned_text = re.sub(r"__(.*?)__", r"\1", cleaned_text)  # __italic__ 처리
    return cleaned_text

# Gemini API를 사용해 책 설명을 생성하는 함수
def generate_book_description(book_data: dict) -> str:
    prompt = (
        f"책 제목: {book_data['title']}\n"
        f"저자: {book_data['author']}\n"
        # f"출판일: {book_data['pubDate']}\n"
        # f"책 설명: {book_data['description']}\n"
        # f"판매 가격: {book_data['priceSales']}원\n"
        f"위 정보를 바탕으로 이 책에 대해 간단한 설명을 작성해 주세요\n" #책 제목과 저자를 이용해 검색하여 책에대한 최신 정보를 알려주세요.
    )

    model = genai.GenerativeModel('gemini-1.5-flash')  # Gemini 모델 사용 gemini-1.5-flash-002
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(prompt)

    # 응답이 None일 경우 대비
    raw_response = response.text if response and hasattr(response, 'text') else ""

    # 마크다운 제거 후 반환
    cleaned_response = clean_response_text(raw_response)
    return cleaned_response

    # return response.text

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


# if SAMPLE_IMAGES:
#     print("images are in dict")

#20241021 YOLO 탐지 결과 처리 함수

def process_yolo_results(results) -> str:
    detected_classes = [r['name'] for r in results[0].boxes.data.cpu().numpy()]
    wear_count = detected_classes.count('wear')
    wet_count = detected_classes.count('wet')
    ripped_count = detected_classes.count('ripped')

    if wet_count > 0:
        return "매입불가"
    elif ripped_count > 0:
        return call_gpt_for_rip_length(results)  # ChatGPT 호출하여 길이 측정
    elif wear_count >= 2:
        return "상"
    elif wear_count == 1:
        return "최상"
    else:
        return "품질 정보 없음"

#20241021 # ChatGPT-4o를 호출하여 찢어진 부위 길이 측정
def call_gpt_for_rip_length(results) -> str:
    ripped_coordinates = [
        box['coordinates'] for box in results[0].boxes.data.cpu().numpy() if box['name'] == 'ripped'
    ]

#20241021 # GPT-4o 모델에 찢어진 길이 요청
    prompt = (
        f"YOLO 모델이 탐지한 찢어진 부위의 좌표는 다음과 같습니다:\n"
        f"{ripped_coordinates}\n"
        "이 좌표를 이용해 찢어진 길이를 cm 단위로 계산해 주세요."
    )

    chat_session = genai.GenerativeModel('gpt-4o').start_chat(history=[])
    response = chat_session.send_message(prompt)

#20241021 # 결과에서 길이 추출 및 반환
    length = response.text.split()[0]  # 예: "3cm" -> "3" 추출
    return f"찢어진 길이: {length}cm"
        


# 20241021 객체 탐지 및 시각화 함수
def detect_and_annotate(image: Image.Image) -> Tuple[Image.Image, bool,Optional[float]]:
    image_np = np.array(image)  # PIL 이미지를 NumPy 배열로 변환
    results = yolo_model(image_np)  # 탐지 수행

    detected = len(results[0].boxes) > 0  # 탐지 여부
    annotated_image = results[0].plot()  # 탐지 결과 시각화
# 찢어진 부분이 있을 경우 길이 계산
    rip_length = None
    contains_wet = False  # wet 존재 여부 초기화


    for box in results[0].boxes.data.cpu().numpy():
        # Bounding box의 너비 또는 높이를 길이로 사용 (예시 계산)
        # 20241021 추가
        class_index = int(box[5])  # 클래스 인덱스는 보통 5번째 위치에 있음
        class_name = yolo_model.names[class_index]  # 클래스 이름으로 변환

        if class_name == 'ripped':
            # Bounding box의 좌표 추출 및 길이 계산
            x1, y1, x2, y2 = box[:4]
            rip_length = round(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 10, 2)  # cm 변환
        elif class_name == 'wet':
            contains_wet = True  # wet이 발견되면 True 설정

            # x1, y1, x2, y2 = box[:4]  # 좌표 추출
            # rip_length = round(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 10, 2)  # cm로 변환
    # 텍스트 정보 추출: 탐지된 객체의 클래스 이름 모음
    detected_texts = [yolo_model.names[int(box[5])] for box in results[0].boxes.data.cpu().numpy()]

    #return Image.fromarray(annotated_image[..., ::-1]), detected, rip_length  # RGB 변환
    return Image.fromarray(annotated_image[..., ::-1]), detected, rip_length, detected_texts  # RGB로 변환하여 반환



# 20241021 수정 YOLOv8 객체 탐지 함수
def yoloinf(image: Image.Image) -> Tuple[str, Image.Image]:
    try: 
        image_np = np.array(image)
        results = yolo_model(image_np)
    # annotated_img = results[0].plot()  # 바운딩 박스 그리기
    # return Image.fromarray(annotated_img[..., ::-1])  # RGB 포맷으로 변환하여 반환
        detected_classes = [yolo_model.names[int(cls)] for cls in results[0].boxes.cls]
        if "ripped" in detected_classes:
            quality = "매입불가"
        elif detected_classes.count("wear") >= 2:
            quality = "상"
        else:
            quality = "최상"

        annotated_img = results[0].plot()
        return quality, Image.fromarray(annotated_img[..., ::-1])
    except Exception as e:
        st.error(f"탐지 중 오류가 발생했습니다: {e}")
        return "오류", image

# 품질 평가 로직 추가
def generate_quality_evaluation(detection_results: list, rip_length: Optional[float] = None, detected_texts: dict = None) -> str:
    #detected_count = sum(int(result) for result in detection_results)
    detected_count = sum(1 for result in detection_results if result != "최상")
    #detected_count = sum(1 for result in detection_results if result != "최상")
    
    # # 탐지된 텍스트에서 'ripped', 'wet', 'wear' 개수 세기
    ripped_count = sum(
        texts.count("front_ripped") + texts.count("side_ripped")
        for texts in detected_texts.values()
    )
    wet_count = sum(
        texts.count("wet") for texts in detected_texts.values()
    )
    wear_count = sum(
        texts.count("wear") for texts in detected_texts.values()
    )

    # 조건 1: wet이 하나라도 감지되면 매입불가
    if wet_count >= 1:
        return "매입불가: 젖은 책이라 매입불가입니다."
    # 조건 2: ripped가 2개 이상 감지되면 매입불가    
    if ripped_count >= 3:
        return "매입불가: 탐지된 찢어진 부위가 2cm 이상입니다."
     # 조건 3: wear만 감지된 경우
    if wear_count > 0 and ripped_count == 0 and wet_count == 0:
        if wear_count == 1:
            return "최상: wear 손상이 1개만 감지되었습니다."
        elif wear_count == 2:
            return "상: wear 손상이 2개 감지되었습니다."

    prompt = (
        f"탐지된 이미지 수: {detected_count}/4\n"
        "책의 품질 등급을 평가해 주세요:\n"
        "- 최상: 새것에 가까운 책\n"
        "- 상: 약간의 사용감이 있으나 깨끗한 상태\n"
        "- 중: 변색 및 약간의 손상 있음\n"
        "- 매입불가: 심한 손상 또는 오염\n"
    )
# 20241021 시작
    if rip_length:
        prompt += f"\n찢어진 길이: {rip_length}cm\n"
    
    # 탐지된 텍스트 정보를 포함
    if detected_texts:
        prompt += "\n탐지된 텍스트 정보:\n"
        for key, texts in detected_texts.items():
            prompt += f"{key}: {', '.join(texts)}\n"
 
    prompt += "\n 이 정보를 바탕으로 책의 품질을 평가해 주세요. 혹시 손상된 부분이 있다면 어떤 부분이 손상되었는지 자세하게 알려주세요."
# 20241021 끗

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
    st.title("Aladin-Bot")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📚 책 정보 조회"):
            st.session_state["show_isbn_input"] = True

    with col2:
        if st.button("⭐ 판매 등급 판정"):
            st.session_state["show_upload"] = True

    if st.session_state.get("show_isbn_input", False):
        isbn = st.text_input("ISBN을 입력하세요")
        if isbn:
            book_data = get_book_data_by_isbn(isbn)
            if book_data:
                book_description = generate_book_description(book_data)
                st.write("생성된 설명:", book_description)
                #save_description_to_json(book_data, book_description)

    if st.session_state.get("show_upload"):
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


            # 각 이미지 탐지 및 시각화 수행 20241021

            
            detected_front, front_found, front_rip, front_texts = detect_and_annotate(Image.open(front_cover))
            detected_back, back_found, back_rip, back_texts = detect_and_annotate(Image.open(back_cover))
            detected_spine, spine_found, spine_rip, spine_texts = detect_and_annotate(Image.open(spine))
            detected_page_edges, edges_found, edges_rip, edges_texts = detect_and_annotate(Image.open(page_edges))

            #detected_front, front_found, front_rip = detect_and_annotate(Image.open(front_cover))
            # detected_front, front_found, front_texts = detect_and_annotate(Image.open(front_cover))
            # st.image(detected_front, caption="앞표지 탐지 결과")
            # detected_back, back_found, back_rip = detect_and_annotate(Image.open(back_cover))
            # detected_spine, spine_found, spine_rip = detect_and_annotate(Image.open(spine))
            # detected_page_edges, edges_found, edges_rip = detect_and_annotate(Image.open(page_edges))

            # 탐지된 이미지 표시
            st.image(detected_front, caption="앞표지 탐지 결과")
            st.image(detected_back, caption="뒷표지 탐지 결과")
            st.image(detected_spine, caption="책등 탐지 결과")
            st.image(detected_page_edges, caption="책배 탐지 결과")
                # 탐지된 텍스트 정보 출력
            # if front_texts:
            #     st.write("앞표지에서 탐지된 객체들:")
            #     st.write(", ".join(front_texts))  # 탐지된 객체 이름들을 쉼표로 구분하여 출력
            # else:
            #     st.write("앞표지에서 탐지된 객체가 없습니다.")
            # st.write("앞표지 탐지 결과:", ", ".join(front_texts))
            # st.write("뒷표지 탐지 결과:", ", ".join(back_texts))
            # st.write("책등 탐지 결과:", ", ".join(spine_texts))
            # st.write("책배 탐지 결과:", ", ".join(edges_texts))
            # 2024102 주석 처리 탐지 결과 저장 (boolean 값만 포함)
           # detection_results = [front_found, back_found, spine_found, edges_found]
            # 20241021 # 찢어진 길이 중 가장 긴 값 선택
            rip_length = max(
                filter(None, [front_rip, back_rip, spine_rip, edges_rip]), default=None
            )
            # 탐지된 텍스트 정보를 딕셔너리로 정리
            detected_texts = {
                "앞표지": front_texts,
                "뒷표지": back_texts,
                "책등": spine_texts,
                "책배": edges_texts,
            }

            
            # 품질 평가 수행
            detection_results = [front_found, back_found,spine_found, edges_found] 
            st.write("Gemini 품질 평가 요청 중...")
            quality_result = generate_quality_evaluation(detection_results, rip_length, detected_texts)
            st.write("품질 평가 결과:", quality_result)
        else:
            st.warning("4장의 이미지를 모두 업로드해 주세요.")



# main 함수 실행
if __name__ == "__main__":
    main()

