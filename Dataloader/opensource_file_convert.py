import cv2
import numpy as np
import random
import os
import matplotlib.pyplot as plt
# 랜덤하게 shift해서 32*32

def random_crop_around_label(image, right_x, right_y, left_x, left_y, crop_size=64, max_offset=8):
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

def visualize_results(original_image, right_cropped_image, left_cropped_image, new_normalized_label, base_name):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # 원본 이미지 표시
    axs[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    # 오른쪽 눈 크롭 이미지 표시
    axs[1].imshow(cv2.cvtColor(right_cropped_image, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Right Eye Crop')
    axs[1].plot(new_normalized_label[0] * 64, new_normalized_label[1] * 64, 'ro')
    axs[1].axis('off')
    
    # 왼쪽 눈 크롭 이미지 표시
    axs[2].imshow(cv2.cvtColor(left_cropped_image, cv2.COLOR_BGR2RGB))
    axs[2].set_title('Left Eye Crop')
    axs[2].plot(new_normalized_label[2] * 64, new_normalized_label[3] * 64, 'ro')
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_base_dir, f"{base_name}_visualization.png"))
    plt.close()

aimmo_folder = ['aimmo_0-4_1', 'aimmo_f_1', 'aimmo_f_2', 'aimmo_i_3', 'aimmo_i_3', 'aimmo_j-z']
base_folder = '/workspace/data/v4.2.1/train'


# 새로운 폴더 경로 설정
output_base_dir = '/workspace/data/aimmo'
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


for folder in aimmo_folder:
    folder_path = os.path.join(base_folder, folder)
    for filename in os.listdir(folder_path):
        # 파일 확장자 확인
        if filename.lower().endswith(allowed_extensions):
            image_path = os.path.join(folder_path, filename)
            label_path = os.path.join(folder_path, filename.split('.')[0] + '.txt')
            all_images.append(image_path)
# 예시로 첫 번째 이미지와 레이블 처리
# all_images가 리스트인 경우, 각 이미지 경로에 대해 반복 처리

for image_path in all_images[:]:
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    txt_path = os.path.join(os.path.dirname(image_path), f"{base_name}.txt")
    original_image = cv2.imread(image_path)
    width , height = original_image.shape[1], original_image.shape[0]
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:16]:
            values = line.split()
            x,y,w,h = map(float, values[1:5])
            x1,y1 = float(values[5])*width, float(values[6])*height
            x2,y2 = float(values[8])*width, float(values[9])*height
            x3,y3 = float(values[11])*width, float(values[12])*height
            x4,y4 = float(values[14])*width, float(values[15])*height
            x5,y5 = float(values[17])*width, float(values[18])*height
        original_right_x, original_right_y, original_left_x, original_left_y = x1,y1,x2,y2
    
    # 원본 이미지에 점 찍기
    original_with_points = original_image.copy()
    cv2.circle(original_with_points, (int(original_right_x), int(original_right_y)), 1, (0, 0, 255), -1)
    cv2.circle(original_with_points, (int(original_left_x), int(original_left_y)), 1, (255, 0, 0), -1)
    
    # 원본 이미지에 점 찍은 것 저장
    #cv2.imwrite(os.path.join(original_with_points_dir, f"{base_name}_with_points.png"), original_with_points)
    

    right_cropped_image, left_cropped_image, new_normalized_label, new_label, crop_coords = random_crop_around_label(
        original_image, original_right_x, original_right_y, original_left_x, original_left_y
    )
    
    # new_normalized_label에 음수 값이 있는지 확인
    if any(val < 0 for val in new_normalized_label):

        continue  # 이 이미지의 처리를 건너뛰고 다음 이미지로 넘어갑니다.
    #visualize_results(original_image, right_cropped_image, left_cropped_image, new_normalized_label, base_name)

    # 크롭된 이미지 저장
    cv2.imwrite(os.path.join(right_output_dir, f"{base_name}_R.png"), right_cropped_image)
    cv2.imwrite(os.path.join(left_output_dir, f"{base_name}_L.png"), left_cropped_image)
    
    # 새로운 레이블 저장
    with open(os.path.join(right_label_dir, f"{base_name}_R.txt"), 'w') as f:
        f.write(f"{new_normalized_label[0]} {new_normalized_label[1]}")
    with open(os.path.join(left_label_dir, f"{base_name}_L.txt"), 'w') as f:
        f.write(f"{new_normalized_label[2]} {new_normalized_label[3]}")
    

print("Processing complete!")

