import cv2
import numpy as np
import random
import os
import matplotlib.pyplot as plt


def random_crop_around_label(image, right_x, right_y, left_x, left_y, crop_size=32, max_offset=8):
    # 이미지 크기 확인
    height, width = image.shape[:2]
 
    # 레이블 주변의 랜덤한 지점 선택 (최대 offset 만큼)
    center_right_x = int(right_x + random.randint(-max_offset, max_offset))
    center_right_y = int(right_y + random.randint(-max_offset, max_offset))
    center_left_x = int(left_x + random.randint(-max_offset, max_offset))
    center_left_y = int(left_y + random.randint(-max_offset, max_offset))
    
    # crop 영역의 좌상단 좌표 계산
    right_left = max(0, center_right_x - crop_size // 2)
    right_top = max(0, center_right_y - crop_size // 2)
    left_left = max(0, center_left_x - crop_size // 2)
    left_top = max(0, center_left_y - crop_size // 2)
    

    # crop 영역이 이미지 경계를 벗어나지 않도록 조정
    right_left = min(right_left, width - crop_size)
    right_top = min(right_top, height - crop_size)
    left_left = min(left_left, width - crop_size)
    left_top = min(left_top, height - crop_size)
    
    # 이미지 crop
    right_cropped_image = image[right_top:right_top+crop_size, right_left:right_left+crop_size]
    left_cropped_image = image[left_top:left_top+crop_size, left_left:left_left+crop_size]
    
    # 새로운 레이블 좌표 계산
    new_right_x = right_x - right_left
    new_right_y = right_y - right_top
    new_left_x = left_x - left_left
    new_left_y = left_y - left_top
    
    # 레이블 좌표 정규화 (0~1 범위로)
    normalized_right_x = new_right_x / crop_size
    normalized_right_y = new_right_y / crop_size
    normalized_left_x = new_left_x / crop_size
    normalized_left_y = new_left_y / crop_size
    return right_cropped_image, left_cropped_image, (normalized_right_x, normalized_right_y, normalized_left_x, normalized_left_y)

# 디렉토리 경로 설정
base_dir = '/Users/hong-eun-yeong/Codes/train'
png_dir = 'eyes_png'
txt_dir = 'total_txt'
blind_dir = 'blind_png'

all_images = []
all_labels = []

if os.path.exists(base_dir):
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for subdir in subdirs:
        #png_path = os.path.join(base_dir, subdir, png_dir)
        txt_path = os.path.join(base_dir, subdir, txt_dir)
        blind_path = os.path.join(base_dir, subdir, blind_dir)
        
        txt_files = [f for f in os.listdir(txt_path) if f.endswith('.txt')]
        
        for txt_file in txt_files:
            txt_full_path = os.path.join(txt_path, txt_file)
            all_labels.append(txt_full_path)
            
            # 해당 txt 파일에 대응하는 이미지 파일 경로 생성
            base_name = os.path.splitext(txt_file)[0]
            blind_img_path = os.path.join(blind_path, f"{base_name}.png")
            #left_eye_path = os.path.join(png_path, f"{base_name}_L.png")
            #right_eye_path = os.path.join(png_path, f"{base_name}_R.png")
            
            # if os.path.exists(blind_img_path) and os.path.exists(left_eye_path) and os.path.exists(right_eye_path):
            #     all_images.append((blind_img_path))

# 예시로 첫 번째 이미지와 레이블 처리
# if all_images and all_labels:
#     blind_img_path= all_images[1000]
#     txt_path = all_labels[1000]
    
#     original_image = cv2.imread(blind_img_path)
    
#     with open(txt_path, 'r') as file:
#         content = file.read().strip()
#         values = content.split()
#         if len(values) >= 7:
#             # 4, 5, 6, 7번째 값 추출 (인덱스는 3부터 시작)
#             extracted = [float(values[i]) for i in range(4, 8)]
            
#             # 값 변환
#             extracted[0] *= 640  # 4번째 값에 640 곱하기
#             extracted[1] *= 480  # 5번째 값에 480 곱하기
#             extracted[2] *= 640  # 6번째 값에 640 곱하기
#             extracted[3] *= 480  # 7번째 값에 480 곱하기
            
                
#     original_right_x, original_right_y = extracted[0], extracted[1]  # 640x480 이미지의 중앙점
#     original_left_x, original_left_y = extracted[2], extracted[3]  # 640x480 이미지의 중앙점

#     # 랜덤 crop 및 새 레이블 계산
#     right_cropped_image, left_cropped_image, new_normalized_label = random_crop_around_label(original_image, original_right_x, original_right_y, original_left_x, original_left_y)
  
  
#     plt.figure(figsize=(15, 5))
#     plt.subplot(1, 3, 1)
#     plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
#     plt.title("원본 이미지")
#     plt.plot(original_right_x, original_right_y, 'ro')
#     plt.plot(original_left_x, original_left_y, 'bo')

#     plt.subplot(1, 3, 2)
#     plt.imshow(cv2.cvtColor(right_cropped_image, cv2.COLOR_BGR2RGB))
#     plt.title("오른쪽 눈 Crop 이미지")
#     plt.plot(new_normalized_label[0]*32, new_normalized_label[1]*32, 'ro')

#     plt.subplot(1, 3, 3)
#     plt.imshow(cv2.cvtColor(left_cropped_image, cv2.COLOR_BGR2RGB))
#     plt.title("왼쪽 눈 Crop 이미지")
#     plt.plot(new_normalized_label[2]*32, new_normalized_label[3]*32, 'bo')

#     plt.show()

#     print(f"새로운 정규화된 레이블 좌표: {new_normalized_label}")
# else:
#     print("처리할 이미지나 레이블이 없습니다.")