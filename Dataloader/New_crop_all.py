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
    
    # 새로운 레이블 좌표 계산 (정규화 전)
    new_right_x = right_x - right_left
    new_right_y = right_y - right_top
    new_left_x = left_x - left_left
    new_left_y = left_y - left_top
    
    # 레이블 좌표 정규화 (0~1 범위로)
    normalized_right_x = new_right_x / crop_size
    normalized_right_y = new_right_y / crop_size
    normalized_left_x = new_left_x / crop_size
    normalized_left_y = new_left_y / crop_size
    
    return right_cropped_image, left_cropped_image, (normalized_right_x, normalized_right_y, normalized_left_x, normalized_left_y), (new_right_x, new_right_y, new_left_x, new_left_y), (right_left, right_top, left_left, left_top)

# 디렉토리 경로 설정
base_dir = '/Users/hong-eun-yeong/Codes/train'
png_dir = 'eyes_png'
txt_dir = 'total_txt'
blind_dir = 'blind_png'
# 새로운 폴더 경로 설정
output_base_dir = '/Users/hong-eun-yeong/Codes/output'
right_output_dir = os.path.join(output_base_dir, 'Right')
left_output_dir = os.path.join(output_base_dir, 'Left')
right_label_dir = os.path.join(output_base_dir, 'right_label')
left_label_dir = os.path.join(output_base_dir, 'left_label')
original_with_points_dir = os.path.join(output_base_dir, 'Original_with_points')

# 새로운 폴더 생성
for dir_path in [right_output_dir, left_output_dir, right_label_dir, left_label_dir]:
    os.makedirs(dir_path, exist_ok=True)

os.makedirs(original_with_points_dir, exist_ok=True)

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
            
            if os.path.exists(blind_img_path):
                all_images.append((blind_img_path))

# 예시로 첫 번째 이미지와 레이블 처리
for i, blind_img_path in enumerate(all_images):
    
    txt_path = all_labels[i]
    
    original_image = cv2.imread(blind_img_path)
    
    with open(txt_path, 'r') as file:
        content = file.read().strip()
        values = content.split()
        if len(values) >= 7:
            extracted = [float(values[i]) for i in range(4, 8)]
            extracted = [extracted[0] * 640, extracted[1] * 480, extracted[2] * 640, extracted[3] * 480]
    
    original_right_x, original_right_y, original_left_x, original_left_y = extracted
    
    # 원본 이미지에 점 찍기
    original_with_points = original_image.copy()
    cv2.circle(original_with_points, (int(original_right_x), int(original_right_y)), 5, (0, 0, 255), -1)
    cv2.circle(original_with_points, (int(original_left_x), int(original_left_y)), 5, (255, 0, 0), -1)
    
    # 원본 이미지에 점 찍은 것 저장
    base_name = os.path.splitext(os.path.basename(blind_img_path))[0]
    cv2.imwrite(os.path.join(original_with_points_dir, f"{base_name}_with_points.png"), original_with_points)
    
    right_cropped_image, left_cropped_image, new_normalized_label, new_label, crop_coords = random_crop_around_label(
        original_image, original_right_x, original_right_y, original_left_x, original_left_y
    )
    
    # new_normalized_label에 음수 값이 있는지 확인
    if any(val < 0 for val in new_normalized_label):
        #base_name = os.path.splitext(os.path.basename(blind_img_path))[0]
        # print(f"에러: 파일 {base_name}에서 음수 정규화 레이블이 발견되었습니다.")
        # print(f"원본 좌표: right_x={original_right_x}, right_y={original_right_y}, left_x={original_left_x}, left_y={original_left_y}")
        # print(f"크롭 좌표: right_left={crop_coords[0]}, right_top={crop_coords[1]}, left_left={crop_coords[2]}, left_top={crop_coords[3]}")
        # print(f"새 좌표 (정규화 전): right_x={new_label[0]}, right_y={new_label[1]}, left_x={new_label[2]}, left_y={new_label[3]}")
        # print(f"정규화된 레이블 값: {new_normalized_label}")
        
        # 문제가 있는 이미지 저장
        # cv2.imwrite(os.path.join(output_base_dir, f"error_{base_name}_original.png"), original_image)
        # cv2.imwrite(os.path.join(output_base_dir, f"error_{base_name}_right_crop.png"), right_cropped_image)
        # cv2.imwrite(os.path.join(output_base_dir, f"error_{base_name}_left_crop.png"), left_cropped_image)
        
        continue  # 이 이미지의 처리를 건너뛰고 다음 이미지로 넘어갑니다.
    
    # 크롭된 이미지 저장
    cv2.imwrite(os.path.join(right_output_dir, f"{base_name}_R.png"), right_cropped_image)
    cv2.imwrite(os.path.join(left_output_dir, f"{base_name}_L.png"), left_cropped_image)
    
    # 새로운 레이블 저장
    with open(os.path.join(right_label_dir, f"{base_name}_R.txt"), 'w') as f:
        f.write(f"{new_normalized_label[0]} {new_normalized_label[1]}")
    with open(os.path.join(left_label_dir, f"{base_name}_L.txt"), 'w') as f:
        f.write(f"{new_normalized_label[2]} {new_normalized_label[3]}")
    
    # 진행 상황 출력
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1} images")

print("Processing complete!")
