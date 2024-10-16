import cv2
import numpy as np

def convert_to_pixel_coordinates(original_width, original_height, left_crop_coords, right_crop_coords, landmarks):
    # landmarks: [right_x, right_y, left_x, left_y] 순서로 되어 있어야 합니다.
    
    right_x, right_y = landmarks[0], landmarks[1]  # 오른쪽 눈 좌표
    left_x, left_y = landmarks[2], landmarks[3]    # 왼쪽 눈 좌표
    print(right_x, right_y, left_x, left_y)
    # 원본 이미지에서의 픽셀 좌표 계산 (float 유지)
    pixel_right_x = right_x * original_width
    pixel_right_y = right_y * original_height
    pixel_left_x = left_x * original_width
    pixel_left_y = left_y * original_height
    print('pixel_right_x, pixel_right_y, pixel_left_x, pixel_left_y:', pixel_right_x, pixel_right_y, pixel_left_x, pixel_left_y)
    # 왼쪽 눈 crop 영역의 시작점 기준으로 픽셀 좌표를 변환
    print(left_crop_coords[0], left_crop_coords[1])
    new_left_x = pixel_left_x - left_crop_coords[0]
    new_left_y = pixel_left_y - left_crop_coords[1]
    
    # 오른쪽 눈 crop 영역의 시작점 기준으로 픽셀 좌표를 변환
    new_right_x = pixel_right_x - right_crop_coords[0]
    new_right_y = pixel_right_y - right_crop_coords[1]
    
    return [new_left_x, new_left_y, new_right_x, new_right_y]

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
        left_y, left_x = left_locations[0][0], left_locations[1][0]
        print(f"Left eye crop coordinates: (x: {left_x}, y: {left_y})")
    else:
        print("Left eye not found!")
        left_x, left_y = None, None

    # 오른쪽 눈에서의 위치 찾기
    right_match = cv2.matchTemplate(original_image, right_eye_image, cv2.TM_CCOEFF_NORMED)
    right_locations = np.where(right_match >= left_threshold)

    if right_locations[0].size > 0:
        right_y, right_x = right_locations[0][0], right_locations[1][0]
        print(f"Right eye crop coordinates: (x: {right_x}, y: {right_y})")
    else:
        print("Right eye not found!")
        right_x, right_y = None, None
    print('left_x, left_y, right_x, right_y:', left_x, left_y, right_x, right_y)
    return (left_x, left_y), (right_x, right_y)

# 원본 이미지 크기
original_width = 640.0
original_height = 480.0

# 경로 설정
original_image_path = '/Users/hong-eun-yeong/Codes/train/SanDiego_cam0_center_gray_002_F_asian_ng_231005_1012_sunny/blind_png/SanDiego_cam0_center_gray_002_F_asian_ng_231005_1012_sunny_22366.png'
left_eye_image_path = '/Users/hong-eun-yeong/Codes/train/SanDiego_cam0_center_gray_002_F_asian_ng_231005_1012_sunny/eyes_png/SanDiego_cam0_center_gray_002_F_asian_ng_231005_1012_sunny_22366_L.png'
right_eye_image_path = '/Users/hong-eun-yeong/Codes/train/SanDiego_cam0_center_gray_002_F_asian_ng_231005_1012_sunny/eyes_png/SanDiego_cam0_center_gray_002_F_asian_ng_231005_1012_sunny_22366_R.png'

# 좌표 찾기
left_crop_coords, right_crop_coords = find_crop_coordinates(original_image_path, left_eye_image_path, right_eye_image_path)

# 각 눈에 대한 crop 영역의 시작 좌표 (예시 값)
if left_crop_coords and right_crop_coords:
    left_crop_x, left_crop_y = left_crop_coords
    right_crop_x, right_crop_y = right_crop_coords

    # 정규화된 좌표
    landmarks = [0.5015625, 0.6083333, 0.5640625, 0.6041667]

    # 픽셀 좌표로 변환
    pixel_landmarks = convert_to_pixel_coordinates(original_width, original_height, (left_crop_x, left_crop_y), (right_crop_x, right_crop_y), landmarks)

    # 32x32 이미지로 정규화
    new_normalized_landmarks = [coord / 32.0 for coord in pixel_landmarks]

    print("New pixel landmarks:", pixel_landmarks)
    print("New normalized landmarks for 32x32 crop:", new_normalized_landmarks)

