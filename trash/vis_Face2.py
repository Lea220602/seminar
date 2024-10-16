import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# 디렉토리 경로 설정
base_dir = '/Users/hong-eun-yeong/Codes/train/'
png_dir = 'blind_png2'
txt_dir = 'total_txt'

all_images = []
all_labels = []

if os.path.exists(base_dir):
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    subdirs = ['SanDiego_cam0_center_gray_002_F_asian_ng_231005_1012_sunny']
    print("\nbase_dir 내의 디렉토리 목록:")
    for d in subdirs:
        print(f"- {d}")
    
    for subdir in subdirs:
        png_path = os.path.join(base_dir, subdir, png_dir)
        print('ㅇㅇ',(png_path) )
        txt_path = os.path.join(base_dir, subdir, txt_dir)
        
        if os.path.exists(png_path) and os.path.exists(txt_path):
            png_files = [f for f in os.listdir(png_path) if f.endswith('.png')]
            print('ㅇㅇ',png_files)
            for png_file in png_files:
                png_full_path = os.path.join(png_path, png_file)
                txt_full_path = os.path.join(txt_path, png_file.replace('.png', '.txt'))
                if os.path.exists(txt_full_path):
                    all_images.append(png_full_path)
                    all_labels.append(txt_full_path)
     
    print(f"\n총 {len(all_images)}개의 PNG 파일과 대응하는 텍스트 파일을 찾았습니다.")
else:
    print(f"\n{base_dir} 디렉토리가 존재하지 않습니다.")

num_samples = min(1, len(all_images))
selected_indices = random.sample(range(len(all_images)), num_samples)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1행 3열

# 크롭 좌표
left_crop_coords = (345, 274)  # 왼쪽 상단 좌표
right_crop_coords = (305, 276)  # 오른쪽 상단 좌표

print("\n시각화된 파일 경로:")
for i, idx in enumerate(selected_indices):
    image_path = all_images[idx]
    txt_path = all_labels[idx]

    print(f"{i+1}. 이미지: {image_path}")
    print(f"   레이블: {txt_path}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 원본 이미지에 랜드마크 표시
    try:
        with open(txt_path, 'r') as f:
            labels = [line.strip().split() for line in f.readlines()]
    except FileNotFoundError:
        print(f"텍스트 파일을 찾을 수 없습니다: {txt_path}")
        continue

    for label in labels:
        if len(label) >= 2:
            L_x = float(label[4]) * 640
            L_y = float(label[5]) * 480
            R_x = float(label[6]) * 640
            R_y = float(label[7]) * 480
            cv2.circle(image, (int(L_x), int(L_y)), radius=1, color=(255, 0, 0), thickness=1)
            cv2.circle(image, (int(R_x), int(R_y)), radius=1, color=(255, 0, 0), thickness=1)

    # 원본 이미지 표시
    axes[0].imshow(image)
    axes[0].axis('off')
    axes[0].set_title('Original Image with Landmarks')

    # 왼쪽 눈 이미지 크롭 및 확대
    left_eye_cropped = image[left_crop_coords[1]:left_crop_coords[1]+32, 
                              left_crop_coords[0]:left_crop_coords[0]+32]
    axes[1].imshow(left_eye_cropped)
    axes[1].axis('off')
    axes[1].set_title('Left Eye (Cropped)')

    # 오른쪽 눈 이미지 크롭 및 확대
    right_eye_cropped = image[right_crop_coords[1]:right_crop_coords[1]+32, 
                               right_crop_coords[0]:right_crop_coords[0]+32]
    axes[2].imshow(right_eye_cropped)
    axes[2].axis('off')
    axes[2].set_title('Right Eye (Cropped)')

plt.tight_layout()
plt.show()
