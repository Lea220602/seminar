import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# 디렉토리 경로 설정
base_dir = '/Users/hong-eun-yeong/Codes/train'
png_dir = 'eyes_png'
txt_dir = 'eyes_txt'

all_images = []
all_labels = []

if os.path.exists(base_dir):
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
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

num_samples = min(10, len(all_images))
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
    
    # 이미지를 32x32로 리사이즈
    image_resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)

    try:
        with open(txt_path, 'r') as f:
            labels = [line.strip().split() for line in f.readlines()]
    except FileNotFoundError:
        print(f"텍스트 파일을 찾을 수 없습니다: {txt_path}")
        continue

    for label in labels:
        if len(label) >= 2:
            x = float(label[0]) * 32
            y = float(label[1]) * 32
            cv2.circle(image_resized, (int(x), int(y)), radius=0, color=(255, 0, 0), thickness=1)

    axes[i].imshow(image_resized)
    axes[i].axis('off')
    axes[i].set_title(f'Image {i+1}')

plt.tight_layout()
plt.show()

