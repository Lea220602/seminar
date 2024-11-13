import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# 디렉토리 경로 설정
base_dir = '/workspace/data/v4.2.2_pupil/Right_eye/'
output_dir = '/workspace/data/v4.2.2_pupil/Right_eye_visualization_results'
os.makedirs(output_dir, exist_ok=True)

all_images = []
all_labels = []

if os.path.exists(base_dir):
    # 직접 base_dir에서 이미지 파일들을 찾음
    image_files = [f for f in os.listdir(base_dir) if f.endswith('.png')]
    
    for img_file in image_files:
        img_full_path = os.path.join(base_dir, img_file)
        txt_full_path = os.path.join(base_dir, img_file.replace('.png', '.txt'))
        
        if os.path.exists(txt_full_path):
            all_images.append(img_full_path)
            all_labels.append(txt_full_path)
    
    print(f"\n총 {len(all_images)}개의 이미지 파일과 대응하는 텍스트 파일을 찾았습니다.")
else:
    print(f"\n{base_dir} 디렉토리가 존재하지 않습니다.")

# 100개의 랜덤 샘플 선택
num_samples = min(100, len(all_images))
selected_indices = random.sample(range(len(all_images)), num_samples)

# 20개씩 5행으로 구성
rows, cols = 5, 20
fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
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
            content = f.read().strip()
            values = content.split()
            if len(values) == 2:  # 정규화된 x, y 좌표만 있음
                x = float(values[0]) * 64  # 정규화된 좌표를 픽셀 좌표로 변환
                y = float(values[1]) * 64
                
                # 점 찍기
                cv2.circle(image, (int(x), int(y)), 1, (255, 0, 0), -1)
    except FileNotFoundError:
        print(f"텍스트 파일을 찾을 수 없습니다: {txt_path}")
        continue

    axes[i].imshow(image)
    axes[i].axis('off')
    axes[i].set_title(f'Image {i+1}', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'visualization_100_samples.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n시각화 결과가 {output_dir} 디렉토리에 저장되었습니다.")

