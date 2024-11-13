import cv2
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import re
# 랜덤하게 shift해서 32*32
from slugify import slugify
# 전역 변수로 눈 인덱스 정의
RIGHT_EYE_INDICES = [0, 1, 9, 10, 11, 12, 13]
LEFT_EYE_INDICES = [7, 8, 2, 3, 4, 5, 6]

def random_crop_around_label(image, coords, min_crop_size=32, max_crop_size=64):
    height, width = image.shape[:2]
    if len(coords) != 14:
        print(f"Error: Expected 14 coordinates, but got {len(coords)}")
        return None, None, None, None, None

    # 6번과 13번 인덱스를 중심 좌표로 사용
    left_center_x, left_center_y = coords[6]
    right_center_x, right_center_y = coords[13]

    # 눈 크기 계산 (기존 방식 유지)
    right_eye_coords = [coords[i] for i in RIGHT_EYE_INDICES]
    left_eye_coords = [coords[i] for i in LEFT_EYE_INDICES]
    
    right_min_x, right_max_x = min(coord[0] for coord in right_eye_coords), max(coord[0] for coord in right_eye_coords)
    right_min_y, right_max_y = min(coord[1] for coord in right_eye_coords), max(coord[1] for coord in right_eye_coords)
    left_min_x, left_max_x = min(coord[0] for coord in left_eye_coords), max(coord[0] for coord in left_eye_coords)
    left_min_y, left_max_y = min(coord[1] for coord in left_eye_coords), max(coord[1] for coord in left_eye_coords)

    right_eye_size = max(right_max_x - right_min_x, right_max_y - right_min_y)
    left_eye_size = max(left_max_x - left_min_x, left_max_y - left_min_y)
    crop_size = min(max(min_crop_size, int(max(right_eye_size, left_eye_size) * 1.5)), max_crop_size)

    # 중심 좌표를 기준으로 크롭 영역 계산
    right_left = max(0, min(int(right_center_x - crop_size / 2), width - crop_size))
    right_top = max(0, min(int(right_center_y - crop_size / 2), height - crop_size))
    left_left = max(0, min(int(left_center_x - crop_size / 2), width - crop_size))
    left_top = max(0, min(int(left_center_y - crop_size / 2), height - crop_size))

    try:
        right_cropped_image = image[right_top:right_top+crop_size, right_left:right_left+crop_size]
        left_cropped_image = image[left_top:left_top+crop_size, left_left:left_left+crop_size]
        
        if right_cropped_image.size == 0 or left_cropped_image.size == 0:
            print(f"Error: Cropped image size is 0. Right: {right_cropped_image.shape}, Left: {left_cropped_image.shape}")
            return None, None, None, None, None
    except Exception as e:
        print(f"Error during image cropping: {str(e)}")
        print(f"Crop parameters - Right: ({right_left}, {right_top}, {crop_size}), Left: ({left_left}, {left_top}, {crop_size})")
        return None, None, None, None, None

    new_normalized_label = {}
    for i, (x, y) in enumerate(coords):
        if i in RIGHT_EYE_INDICES:
            new_x, new_y = (x - right_left) / crop_size, (y - right_top) / crop_size
        elif i in LEFT_EYE_INDICES:
            new_x, new_y = (x - left_left) / crop_size, (y - left_top) / crop_size
        new_normalized_label[i*2] = max(0, min(1, new_x))
        new_normalized_label[i*2+1] = max(0, min(1, new_y))

    return right_cropped_image, left_cropped_image, new_normalized_label, (right_left, right_top, crop_size), (left_left, left_top, crop_size)

def visualize_results(original_image, left_cropped_image, right_cropped_image, new_normalized_label, coords, left_crop_area, right_crop_area, base_name):
    try:
        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        
        axs[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title('Original Image', fontsize=16)
        
        for i, (x, y) in enumerate(coords):
            color = 'r' if i in RIGHT_EYE_INDICES else 'g'  # 'r'은 red, 'g'는 green
            axs[0, 0].plot(x, y, color + 'o', markersize=8)  # 동적으로 색상과 마커를 설정
            axs[0, 0].text(x, y, str(i), color='white', fontsize=12, fontweight='bold',
                           bbox=dict(facecolor=color, edgecolor='none', alpha=0.7))
        
        right_rect = plt.Rectangle((right_crop_area[0], right_crop_area[1]), right_crop_area[2], right_crop_area[2], fill=False, edgecolor='red', linewidth=2)
        left_rect = plt.Rectangle((left_crop_area[0], left_crop_area[1]), left_crop_area[2], left_crop_area[2], fill=False, edgecolor='lime', linewidth=2)
        axs[0, 0].add_patch(right_rect)
        axs[0, 0].add_patch(left_rect)
        axs[0, 0].axis('off')
        
        axs[0, 1].imshow(cv2.cvtColor(right_cropped_image, cv2.COLOR_BGR2RGB))
        axs[0, 1].set_title('Right Eye Crop', fontsize=16)
        for i in RIGHT_EYE_INDICES:
            x = new_normalized_label[i*2] * right_crop_area[2]
            y = new_normalized_label[i*2+1] * right_crop_area[2]
            axs[0, 1].plot(x, y, 'ro', markersize=8)
            axs[0, 1].text(x, y, str(i), color='white', fontsize=12, fontweight='bold',
                           bbox=dict(facecolor='red', edgecolor='none', alpha=0.7))
        axs[0, 1].axis('off')
        
        axs[1, 0].imshow(cv2.cvtColor(left_cropped_image, cv2.COLOR_BGR2RGB))
        axs[1, 0].set_title('Left Eye Crop', fontsize=16)
        for i in LEFT_EYE_INDICES:
            x = new_normalized_label[i*2] * left_crop_area[2]
            y = new_normalized_label[i*2+1] * left_crop_area[2]
            axs[1, 0].plot(x, y, 'go', markersize=8)
            axs[1, 0].text(x, y, str(i), color='white', fontsize=12, fontweight='bold',
                           bbox=dict(facecolor='lime', edgecolor='none', alpha=0.7))
        axs[1, 0].axis('off')
        
        right_x = [new_normalized_label[i*2] for i in RIGHT_EYE_INDICES]
        right_y = [new_normalized_label[i*2+1] for i in RIGHT_EYE_INDICES]
        left_x = [new_normalized_label[i*2] for i in LEFT_EYE_INDICES]
        left_y = [new_normalized_label[i*2+1] for i in LEFT_EYE_INDICES]
        
        axs[1, 1].scatter(right_x, right_y, c='red', s=100)
        axs[1, 1].scatter(left_x, left_y, c='lime', s=100)
        
        for i in RIGHT_EYE_INDICES + LEFT_EYE_INDICES:
            x = new_normalized_label[i*2]
            y = new_normalized_label[i*2+1]
            color = 'r' if i in RIGHT_EYE_INDICES else 'g'
            axs[1, 1].text(x, y, str(i), color='white', fontsize=12, fontweight='bold',
                           bbox=dict(facecolor=color, edgecolor='none', alpha=0.7))
        axs[1, 1].set_xlim(0, 1)
        axs[1, 1].set_ylim(0, 1)
        axs[1, 1].set_title('Normalized Coordinates Distribution', fontsize=16)
        axs[1, 1].set_xlabel('X', fontsize=14)
        axs[1, 1].set_ylabel('Y', fontsize=14)
        
        # 눈 중심 좌표 표시
        left_center_x, left_center_y = new_normalized_label[12], new_normalized_label[13]  # 6번 인덱스
        right_center_x, right_center_y = new_normalized_label[26], new_normalized_label[27]  # 13번 인덱스

        axs[0, 1].plot(right_center_x * right_crop_area[2], right_center_y * right_crop_area[2], 'bo', markersize=10)
        axs[1, 0].plot(left_center_x * left_crop_area[2], left_center_y * left_crop_area[2], 'bo', markersize=10)

        axs[1, 1].plot(right_center_x, right_center_y, 'bo', markersize=10)
        axs[1, 1].plot(left_center_x, left_center_y, 'bo', markersize=10)

        #plt.tight_layout()
        #output_path = os.path.join(output_base_dir, base_name + "_visualization.png")
        #plt.savefig(output_path, dpi=300, bbox_inches='tight')
        #plt.close()
        #print("Visualization saved: " + output_path)
    except Exception as e:
        print("Error in visualize_results: " + str(e))
        print("Error occurred at line: " + str(e.__traceback__.tb_lineno))

#aimmo_folder = ['aimmo_0-4_1']
base_folder = "/workspace/data/project1_rev_test" 


# 새로운 폴더 로 설정
output_base_dir = "/workspace/data/aimmo/project1_rev_test/eyes" 
right_output_dir = os.path.join(output_base_dir, 'Right')
left_output_dir = os.path.join(output_base_dir, 'Left')
right_label_dir = os.path.join(output_base_dir, 'Right')
left_label_dir = os.path.join(output_base_dir, 'Left')
original_with_points_dir = os.path.join(output_base_dir, 'Original_with_points')
output_dirs = [right_output_dir, left_output_dir, right_label_dir, left_label_dir, original_with_points_dir]

for dir_path in output_dirs:
    os.makedirs(dir_path, exist_ok=True)

# 허용할 이미지 확장자 목록
allowed_extensions = ('.png', '.jpg', '.jpeg')
all_images = []
all_labels = []


folder_path = os.path.join(base_folder)
for filename in os.listdir(folder_path):
        # 파일 확장자 확인
    if filename.lower().endswith(allowed_extensions):
        image_path = os.path.join(folder_path, filename)
        label_path = os.path.join(folder_path, filename.split('.')[0] + '.txt')
        all_images.append(image_path)
        all_labels.append(label_path)


# 메인 처리 루프
for image_path in all_images[:]:
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    txt_path = os.path.join(os.path.dirname(image_path), f"{base_name}.txt")
    original_image = cv2.imread(image_path)
    width, height = original_image.shape[1], original_image.shape[0]
    coords = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:]:
            values = line.split()
            for i in range(0, 28, 2):
                x = float(values[i]) * width
                y = float(values[i+1]) * height
                coords.append((x, y))

    right_cropped_image, left_cropped_image, new_normalized_label, right_crop_area, left_crop_area = random_crop_around_label(original_image, coords)
    if right_cropped_image is None or left_cropped_image is None:
        print(f"Skipping image {base_name} due to cropping error")
        continue

    try:
        safe_base_name = re.sub(r'[^a-zA-Z0-9_-]', '_', base_name)
        #visualize_results(original_image, left_cropped_image, right_cropped_image, new_normalized_label, coords, left_crop_area, right_crop_area, safe_base_name)
    except Exception as e:
        print(f"Error: 이미지 {base_name} 시각화 중 오류 발생 - {str(e)}")
        continue
    cv2.imwrite(os.path.join(right_output_dir, f"{safe_base_name}_R.png"), right_cropped_image)
    cv2.imwrite(os.path.join(left_output_dir, f"{safe_base_name}_L.png"), left_cropped_image)
    
    with open(os.path.join(right_label_dir, f"{safe_base_name}_R.txt"), 'w') as f:
        # 오른쪽 눈 랜드마크만 저장 (RIGHT_EYE_INDICES)
        for i in RIGHT_EYE_INDICES:
            x = new_normalized_label[i*2]
            y = new_normalized_label[i*2+1]
            f.write(f"{x} {y}\n")

    with open(os.path.join(left_label_dir, f"{safe_base_name}_L.txt"), 'w') as f:
        # 왼쪽 눈 랜드마크만 저장 (LEFT_EYE_INDICES)
        for i in LEFT_EYE_INDICES:
            x = new_normalized_label[i*2]
            y = new_normalized_label[i*2+1]
            f.write(f"{x} {y}\n")

print("Processing complete!")































