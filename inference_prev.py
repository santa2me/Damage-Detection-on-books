## Inference page

## diff dataset with same models eval(with yolo)
# Initial dataset probelms >> modified num of classes (side_wear, front_ripped .. etc)

## what, how image processing/other tech(increased dataset size, augumentation) improved models
# diff augumentation or image processing with same models eval (if can find)


##Finalize dataset >> WHY?
##with the same dataset(best) >> diff models(in yolo)            
#                             >> Detectron
#                             >> Faster RCNN                >>>> Show the predicted same image
#                             >> MASK RCNN


## What model selected for demo? WHY?


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_csv_comparison(csv_files, file_labels):
    """
    Parameters:
    csv_files (list): CSV 파일 경로 리스트
    plot_title (str): Plot 제목 
    file_labels (list): 각 파일에 해당하는 레이블 리스트
    
    Returns:
    None: Streamlit에 차트를 렌더링
    
    """

    col1, col2, col3, col4, col5= st.columns([0.5,0.5, 3, 0.5, 0.5])

    with col3:
    # Load the CSV files
        data = [pd.read_csv(file) for file in csv_files]

        # Create a single chart combining the data from the files
        fig, ax = plt.subplots()

        # Plot the metrics for each CSV file on the same chart
        for idx, df in enumerate(data):
            ax.plot(df['epoch'], df['metrics/mAP50(B)'], label=file_labels[idx])
        # for idx, df in enumerate(data):
        #     ax.plot(df['epoch'], df['metrics/mAP50(B)'], label=f'File {idx+1} - mAP50(B)')
            # ax.plot(df['epoch'], df['metrics/mAP50-95(B)'], label=f'File {idx+1} - mAP50-95(B)')
            # ax.plot(df['epoch'], df['metrics/mAP50(M)'], label=f'File {idx+1} - mAP50(M)')
            # ax.plot(df['epoch'], df['metrics/mAP50-95(M)'], label=f'File {idx+1} - mAP50-95(M)')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('metrics/mAP50(B)')
        # ax.legend()

        # Add a plot title
        # ax.set_title(plot_title)

        ax.legend()

        # Show the plot in Streamlit
        st.pyplot(fig)
        
        

# Example usage:
# csv_files = ['csv_files/results.csv', 'results1.csv', 'results2.csv', 'results3.csv']
# plot_csv_comparison(csv_files)

# Vertical_Histogram definition
def create_vertical_histogram(labels, values):
    """
    Parameters:
    labels (list): 히스토그램의 x축 레이블 리스트
    values (list): 히스토그램의 각 항목에 해당하는 값 리스트
    """

    col1, col2, col3, col4, col5= st.columns([0.5,0.5, 3, 0.5, 0.5])

    with col3:
        fig, ax = plt.subplots()

        ax.bar(labels, values)

        # ax.set_title(title)
        ax.set_xlabel('Version')
        ax.set_ylabel('mAP')

        st.pyplot(fig)


# Horizontal Histogram Definition
def create_horizontal_histogram(labels, values):
    """
    Parameters:
    labels (list): 히스토그램의 y축 레이블 리스트
    values (list): 히스토그램의 각 항목에 해당하는 값 리스트
    """

    col1, col2, col3, col4, col5= st.columns([0.5,0.5, 3, 0.5, 0.5])

    with col3:
        fig, ax = plt.subplots()

        ax.barh(labels, values) 

        # ax.set_title(title)
        ax.set_xlabel('Value')
        ax.set_ylabel('Version')

        st.pyplot(fig)

# main() function

def main():
    st.header("Review", divider="red")

    st.markdown(
    "<h1 style='text-align:center; font-style: normal; font-size:30px; '>Object Detection  vs  Instance Segmentation</h1>",
    unsafe_allow_html=True)
    st.divider()

    st.subheader("• 초기 방향성")

    cols = st.columns(2)
    with cols[0]:
        st.image("figures/OD_anno.png", caption="Bounding box", width=400)
    with cols[1]:
        st.image("figures/IS_anno.png", caption="Polygon", width=400)


    # 첫 번째 섹션: Object Detection vs Instance Segmentation
    # Histogram
    labels = ['YOLOv6', 'YOLOv7', 'YOLOv11']
    values = [0.131, 0.126, 0.137] # 각 버전의 성능 값

    # 두 번째 섹션: 5 Class vs 9 Class
    csv_five_nine = ['csv_files/1011_aug2_results.csv', 'csv_files/dup2_not_aug_results_small.csv']
    file_labels_five_nine = ['5 Class Model', '9 Class Model']

    # 세 번째 섹션: aug VS not aug
    csv_aug_diff = ['csv_files/dup2_not_aug_results_small.csv', 'csv_files/dup2_aug_results_small.csv']
    file_labels_aug_diff = ['No Augmentation', 'With Augmentation']

    # 네 번째 섹션: nano VS small vs large
    csv_ins_version = ['csv_files/dup2_aug_results_nano.csv', 'csv_files/dup2_aug_results_small.csv', 'csv_files/dup2_aug_results_large.csv']
    file_labels_ins_version = ['Nano Model', 'Small Model', 'Large Model']


    # 다섯 번째 섹션: Faster R-CNN VS Detectron VS Mask R-CNN
    ## Test image + mAP + Loss func Graph

    st.markdown(
    "<h1 style='text-align:center; font-style: normal; font-size:20px; '>Comparison of Different YOLO Versions (Object Detection)</h1>",
    unsafe_allow_html=True)
    # st.subheader('Comparison of Different YOLO Versions (Object Detection)')
    create_vertical_histogram(labels, values)
    st.divider()

    st.markdown(
    "<h1 style='text-align:center; font-style: normal; font-size:20px; '>5 Class vs 9 Class Comparison</h1>",
    unsafe_allow_html=True)
    plot_csv_comparison(csv_five_nine, file_labels_five_nine)
    st.divider()

    st.markdown(
    "<h1 style='text-align:center; font-style: normal; font-size:20px; '>Augmentation vs Non-Augmentation</h1>",
    unsafe_allow_html=True)
    plot_csv_comparison(csv_aug_diff, file_labels_aug_diff)
    st.divider()

    st.markdown(
    "<h1 style='text-align:center; font-style: normal; font-size:20px; '>Model Version Comparison: Nano vs Small vs Large</h1>",
    unsafe_allow_html=True)
    plot_csv_comparison(csv_ins_version, file_labels_ins_version)
        #4-1 번째 섹션: Loss function in YOLO
    loss = Image.open('figures/loss.png')
    st.image(loss, caption='Loss Graph', use_column_width=True)

        # 마지막 부분: 이미지 두 개씩 한 줄에 표시
    st.markdown("<h1 style='text-align:center; font-size:20px;'>Additional Results</h1>", unsafe_allow_html=True)
    cols = st.columns(2)

    with cols[0]:
        img1 = Image.open('figures/Faster_r_cnn_back.png')
        st.image(img1, use_column_width=True)
        st.markdown(
            "<h1 style='text-align:center; font-size:20px;margin: 0; padding: 0;'>Faster_r_cnn 탐지 결과</h1>",
            unsafe_allow_html=True,
        )

    with cols[1]:
        img2 = Image.open('figures/Mask_r_cnn_side.png')
        st.image(img2, use_column_width=True)
        st.markdown(
            "<h1 style='text-align:center; font-size:20px;margin: 0; padding: 0;'>Mask_r_cnn 탐지 결과</h1>",
            unsafe_allow_html=True,
        )



    # main 함수 실행
if __name__ == "__main__":
    main()
