import cv2
import numpy as np
def find_crop_coordinates(original_image_path, left_eye_image_path, right_eye_image_path):
    # 원본 이미지 로드
    original_image = cv2.imread(original_image_path)

    # 왼쪽 눈 이미지 로드
    left_eye_image = cv2.imread(left_eye_image_path)

    # 오른쪽 눈 이미지 로드
    right_eye_image = cv2.imread(right_eye_image_path)

    # 왼쪽 눈에서의 위치 찾기
    left_match = cv2.matchTemplate(original_image, left_eye_image, cv2.TM_CCOEFF_NORMED)
    left_threshold = 1.0  # 매칭 임계값 설정
    left_locations = np.where(left_match >= left_threshold)

    if left_locations[0].size > 0:
        # 첫 번째 매칭된 좌표 가져오기
        left_y, left_x = left_locations[0][0], left_locations[1][0]
        print(f"Left eye crop coordinates: (x: {left_x}, y: {left_y})")
    else:
        print("Left eye not found!")

    # 오른쪽 눈에서의 위치 찾기
    right_match = cv2.matchTemplate(original_image, right_eye_image, cv2.TM_CCOEFF_NORMED)
    right_locations = np.where(right_match >= left_threshold)

    if right_locations[0].size > 0:
        # 첫 번째 매칭된 좌표 가져오기
        right_y, right_x = right_locations[0][0], right_locations[1][0]
        print(f"Right eye crop coordinates: (x: {right_x}, y: {right_y})")
    else:
        print("Right eye not found!")

# 경로 설정
original_image_path = '/Users/hong-eun-yeong/Codes/train/SanDiego_cam0_center_gray_002_F_asian_ng_231005_1012_sunny/blind_png/SanDiego_cam0_center_gray_002_F_asian_ng_231005_1012_sunny_22366.png'  # 원본 이미지 경로
left_eye_image_path =  '/Users/hong-eun-yeong/Codes/train/SanDiego_cam0_center_gray_002_F_asian_ng_231005_1012_sunny/eyes_png/SanDiego_cam0_center_gray_002_F_asian_ng_231005_1012_sunny_22366_L.png'
right_eye_image_path =  '/Users/hong-eun-yeong/Codes/train/SanDiego_cam0_center_gray_002_F_asian_ng_231005_1012_sunny/eyes_png/SanDiego_cam0_center_gray_002_F_asian_ng_231005_1012_sunny_22366_R.png'

# 좌표 찾기
find_crop_coordinates(original_image_path, left_eye_image_path, right_eye_image_path)
