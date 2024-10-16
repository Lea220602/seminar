import cv2
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt


def random_crop_around_label(image, right_x, right_y, left_x, left_y, crop_size=32, max_offset=8):
    # 이미지 크기 확인
    height, width = image.shape[:2]
    print('height, width:', height, width)
    print('right_x, right_y:', right_x, right_y)
    print('left_x, left_y:', left_x, left_y)
    # 레이블 주변의 랜덤한 지점 선택 (최대 offset 만큼)
    center_right_x = int(right_x + random.randint(-max_offset, max_offset))
    center_right_y = int(right_y + random.randint(-max_offset, max_offset))
    print('center_right_x, center_right_y:', center_right_x, center_right_y)
    center_left_x = int(left_x + random.randint(-max_offset, max_offset))
    center_left_y = int(left_y + random.randint(-max_offset, max_offset))
    print('center_left_x, center_left_y:', center_left_x, center_left_y)
    
    # crop 영역의 좌상단 좌표 계산
    right_left = max(0, center_right_x - crop_size // 2)
    right_top = max(0, center_right_y - crop_size // 2)
    left_left = max(0, center_left_x - crop_size // 2)
    left_top = max(0, center_left_y - crop_size // 2)
    
    print('Right_left, Right_top:', right_left, right_top)
    print('Left_left, Left_top:', left_left, left_top)
    
    # crop 영역이 이미지 경계를 벗어나지 않도록 조정
    right_left = min(right_left, width - crop_size)
    right_top = min(right_top, height - crop_size)
    left_left = min(left_left, width - crop_size)
    left_top = min(left_top, height - crop_size)
    
    # 이미지 crop
    right_cropped_image = image[right_top:right_top+crop_size, right_left:right_left+crop_size]
    print('right_x, right_y:', right_x, right_y)
    print('left_x, left_y:', left_x, left_y)
    left_cropped_image = image[left_top:left_top+crop_size, left_left:left_left+crop_size]
    
    # 새로운 레이블 좌표 계산
    new_right_x = right_x - right_left
    new_right_y = right_y - right_top
    print('new_right_x, new_right_y:', new_right_x, new_right_y)
    new_left_x = left_x - left_left
    new_left_y = left_y - left_top
    print('new_left_x, new_left_y:', new_left_x, new_left_y)
    
    # 레이블 좌표 정규화 (0~1 범위로)
    normalized_right_x = new_right_x / crop_size
    normalized_right_y = new_right_y / crop_size
    print('normalized_right_x, normalized_right_y:', normalized_right_x, normalized_right_y)
    normalized_left_x = new_left_x / crop_size
    normalized_left_y = new_left_y / crop_size
    print('normalized_left_x, normalized_left_y:', normalized_left_x, normalized_left_y)
    return right_cropped_image, left_cropped_image, (normalized_right_x, normalized_right_y, normalized_left_x, normalized_left_y)

# 예시 사용
ori_path = '/Users/hong-eun-yeong/Codes/train/SanDiego_cam0_center_gray_002_F_asian_ng_231005_1012_sunny/blind_png/SanDiego_cam0_center_gray_002_F_asian_ng_231005_1012_sunny_22366.png'
img_L_path = '/Users/hong-eun-yeong/Codes/train/SanDiego_cam0_center_gray_002_F_asian_ng_231005_1012_sunny/eyes_png/SanDiego_cam0_center_gray_002_F_asian_ng_231005_1012_sunny_26056_L.png'
img_R_path = '/Users/hong-eun-yeong/Codes/train/SanDiego_cam0_center_gray_002_F_asian_ng_231005_1012_sunny/eyes_png/SanDiego_cam0_center_gray_002_F_asian_ng_231005_1012_sunny_26056_R.png'
txt_path = '/Users/hong-eun-yeong/Codes/train/SanDiego_cam0_center_gray_002_F_asian_ng_231005_1012_sunny/total_txt/SanDiego_cam0_center_gray_002_F_asian_ng_231005_1012_sunny_22366.txt'
original_image = cv2.imread(ori_path)


with open(txt_path, 'r') as file:
    content = file.read().strip()
    values = content.split()
    if len(values) >= 7:
        # 4, 5, 6, 7번째 값 추출 (인덱스는 3부터 시작)
        extracted = [float(values[i]) for i in range(4, 8)]
        
        # 값 변환
        extracted[0] *= 640  # 4번째 값에 640 곱하기
        extracted[1] *= 480  # 5번째 값에 480 곱하기
        extracted[2] *= 640  # 6번째 값에 640 곱하기
        extracted[3] *= 480  # 7번째 값에 480 곱하기
        
                
# 원본 레이블 좌표 (예시)
original_right_x, original_right_y = extracted[0], extracted[1]  # 640x480 이미지의 중앙점
original_left_x, original_left_y = extracted[2], extracted[3]  # 640x480 이미지의 중앙점

# 랜덤 crop 및 새 레이블 계산
right_cropped_image, left_cropped_image, new_normalized_label = random_crop_around_label(original_image, original_right_x, original_right_y, original_left_x, original_left_y)

# 시각화
plt.figure(figsize=(15, 5))

# 원본 이미지
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title("원본 이미지")
plt.plot(original_right_x, original_right_y, 'ro', label='오른쪽 눈')
plt.plot(original_left_x, original_left_y, 'bo', label='왼쪽 눈')
plt.legend()

# 오른쪽 눈 크롭 이미지
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(right_cropped_image, cv2.COLOR_BGR2RGB))
plt.title("오른쪽 눈 Crop 이미지")
plt.plot(new_normalized_label[0]*32, new_normalized_label[1]*32, 'ro')
plt.text(0, -2, f'좌표: ({new_normalized_label[0]:.2f}, {new_normalized_label[1]:.2f})', color='red')

# 왼쪽 눈 크롭 이미지
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(left_cropped_image, cv2.COLOR_BGR2RGB))
plt.title("왼쪽 눈 Crop 이미지")
plt.plot(new_normalized_label[2]*32, new_normalized_label[3]*32, 'bo')
plt.text(0, -2, f'좌표: ({new_normalized_label[2]:.2f}, {new_normalized_label[3]:.2f})', color='blue')

plt.tight_layout()
plt.show()

# 정규화된 좌표 출력
print("새로운 정규화된 좌표:")
print(f"오른쪽 눈: ({new_normalized_label[0]:.2f}, {new_normalized_label[1]:.2f})")
print(f"왼쪽 눈: ({new_normalized_label[2]:.2f}, {new_normalized_label[3]:.2f})")
