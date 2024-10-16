import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# 디렉토리 경로 설정
base_dir = '/Users/hong-eun-yeong/Codes/train'
png_dir = 'eyes_png'
txt_dir = 'total_txt'

all_images = []
all_labels = []

if os.path.exists(base_dir):
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    #print("\nbase_dir 내의 디렉토리 목록:")
    # for d in subdirs:
    #     print(f"- {d}")
    
    for subdir in subdirs:
        png_path = os.path.join(base_dir, subdir, png_dir)
        txt_path = os.path.join(base_dir, subdir, txt_dir)
        txt_files = [f for f in os.listdir(txt_path) if f.endswith('.txt')]
        #print(len(txt_files))
        #print(txt_path)
        
        # 각 txt 파일에 대해 반복
        for txt_file in txt_files:
            txt_full_path = os.path.join(txt_path, txt_file)
            all_labels.append(txt_full_path)


#undenormalize

label_contents = []
extracted_values = []

#for label_path in all_labels:
for label_path in all_labels[0:1]:
    print(label_path)
    try:
        with open(label_path, 'r') as file:
            content = file.read().strip()
            values = content.split()
            if len(values) >= 7:
                # 4, 5, 6, 7번째 값 추출 (인덱스는 3부터 시작)
                extracted = [float(values[i]) for i in range(3, 7)]
                
                # 값 변환
                extracted[0] *= 640  # 4번째 값에 640 곱하기
                extracted[1] *= 480  # 5번째 값에 480 곱하기
                extracted[2] *= 640  # 6번째 값에 640 곱하기
                extracted[3] *= 480  # 7번째 값에 480 곱하기
                
                extracted_values.append(extracted)
            else:
                print(f"파일에 충분한 값이 없습니다: {label_path}")
    except IOError as e:
        print(f"파일을 읽는 중 오류 발생: {label_path}")
        print(f"오류 내용: {str(e)}")
    except ValueError as e:
        print(f"값을 변환하는 중 오류 발생: {label_path}")
        print(f"오류 내용: {str(e)}")

print(f"처리된 라벨 파일 수: {len(extracted_values)}")
if extracted_values:
    print("첫 번째 추출된 값 (denormalize 변환 후):")
    print('denormalize 변환 후',extracted_values[0])


# landmarks: [right_x, right_y, left_x, left_y] 순서로 되어 있어야 합니다.
left_x, left_y = extracted_values[0][2], extracted_values[0][3]    # 왼쪽 눈 좌표
right_x, right_y = extracted_values[0][0], extracted_values[0][1]  # 오른쪽 눈 좌표

# 왼쪽 눈 crop 영역의 시작점 기준으로 픽셀 좌표를 변환
crop_left_x = left_x - 16.0
crop_left_y = left_y - 16.0

# 오른쪽 눈 crop 영역의 시작점 기준으로 픽셀 좌표를 변환
crop_right_x = right_x -16.0
crop_right_y = right_y -16.0

print('crop 시작 위치',crop_left_x, crop_left_y, crop_right_x, crop_right_y) #crop 좌표


def convert_to_pixel_coordinates(left_crop_coords, right_crop_coords, landmarks):
    # landmarks: [right_x, right_y, left_x, left_y] 순서로 되어 있어야 합니다.
    
    right_x, right_y = landmarks[0], landmarks[1]  # 오른쪽 눈 좌표
    left_x, left_y = landmarks[2], landmarks[3]    # 왼쪽 눈 좌표
    # 원본 이미지에서의 픽셀 좌표 계산 (float 유지)
    pixel_right_x = right_x 
    pixel_right_y = right_y 
    pixel_left_x = left_x 
    pixel_left_y = left_y
    print('denormalize 변환 후',pixel_right_x, pixel_right_y, pixel_left_x, pixel_left_y)
    # 왼쪽 눈 crop 영역의 시작점 기준으로 픽셀 좌표를 변환
    print('left_crop_coords:', left_crop_coords[0], left_crop_coords[1])
    new_left_x = pixel_left_x - left_crop_coords[0]
    new_left_y = pixel_left_y - left_crop_coords[1]
    
    # 오른쪽 눈 crop 영역의 시작점 기준으로 픽셀 좌표를 변환
    new_right_x = pixel_right_x - right_crop_coords[0]
    new_right_y = pixel_right_y - right_crop_coords[1]
    print('new_left_x, new_left_y, new_right_x, new_right_y:', new_left_x, new_left_y, new_right_x, new_right_y)
    return [new_left_x, new_left_y, new_right_x, new_right_y]

pixel_landmarks = convert_to_pixel_coordinates((crop_left_x, crop_left_y), (crop_right_x, crop_right_y), extracted_values[0])
new_normalized_landmarks = [coord / 32.0 for coord in pixel_landmarks]

print('new_normalized_landmarks:', new_normalized_landmarks)


