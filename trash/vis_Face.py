import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# 디렉토리 경로 설정
base_dir = '/Users/hong-eun-yeong/Codes/train'
png_dir = 'blind_png2'
txt_dir = 'total_txt'

all_images = []
all_labels = []

if os.path.exists(base_dir):
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    subdirs = ['/Users/hong-eun-yeong/Codes/train/Detroit_cam0_apillar_gray_006_M_asian_ng_240425_0616_sunny']
    print("\nbase_dir 내의 디렉토리 목록:")
    for d in subdirs:
        print(f"- {d}")
    
    for subdir in subdirs:
        png_path = os.path.join(base_dir, subdir, png_dir)
        txt_path = os.path.join(base_dir, subdir, txt_dir)
        
        if os.path.exists(png_path) and os.path.exists(txt_path):
            png_files = [f for f in os.listdir(png_path) if f.endswith('.png')]
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

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

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

    axes[i].imshow(image)
    axes[i].axis('off')
    axes[i].set_title(f'Image {i+1}')

plt.tight_layout()
plt.show()

