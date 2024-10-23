import cv2
import mediapipe as mp
import os
import math

# MediaPipe Face Mesh 및 Iris 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_iris = mp.solutions.face_mesh

# Iris 관련 랜드마크 인덱스
LEFT_EYE_CORNER = [33, 133]  # 왼쪽 눈 끝 인덱스
RIGHT_EYE_CORNER = [362, 263]  # 오른쪽 눈 끝 인덱스
LEFT_IRIS = [474, 475, 476, 477]  # 왼쪽 눈 동공
RIGHT_IRIS = [469, 470, 471, 472]  # 오른쪽 눈 동공

# 입력 이미지 디렉토리 설정
input_dir = "/workspace/data/v4.2.1/train/aimmo_f_1"
# 출력 디렉토리 생성
output_dir = "/workspace/data/aimmo/mediapipe_results" 
os.makedirs(output_dir, exist_ok=True)

max_images = 10
image_count = 0

# MediaPipe Face Mesh 초기화
with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
    # 각 이미지 처리
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # 이미지 읽기
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            height, width, _ = image.shape

            # BGR을 RGB로 변환
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Face Mesh 처리
            results = face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 왼쪽과 오른쪽 눈 끝 시각화
                    left_eye_corner = [face_landmarks.landmark[i] for i in LEFT_EYE_CORNER]
                    right_eye_corner = [face_landmarks.landmark[i] for i in RIGHT_EYE_CORNER]

                    # 왼쪽 눈 끝 그리기
                    for idx, landmark in enumerate(left_eye_corner):
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)  # 녹색으로 눈 끝 표시

                    # 오른쪽 눈 끝 그리기
                    for idx, landmark in enumerate(right_eye_corner):
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)  # 녹색으로 눈 끝 표시

                    # 왼쪽 눈 동공 시각화
                    left_iris = [face_landmarks.landmark[i] for i in LEFT_IRIS]
                    left_iris_center_x = sum([p.x for p in left_iris]) / len(left_iris)
                    left_iris_center_y = sum([p.y for p in left_iris]) / len(left_iris)
                    left_iris_center = (int(left_iris_center_x * width), int(left_iris_center_y * height))

                    # 동공 가장자리와 원 시각화
                    for landmark in left_iris:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)  # 빨간색으로 동공 경계점

                    # 동공 중심 시각화
                    cv2.circle(image, left_iris_center, 3, (255, 0, 0), -1)  # 파란색으로 동공 중앙 표시

                    # 오른쪽 눈 동공 시각화
                    right_iris = [face_landmarks.landmark[i] for i in RIGHT_IRIS]
                    right_iris_center_x = sum([p.x for p in right_iris]) / len(right_iris)
                    right_iris_center_y = sum([p.y for p in right_iris]) / len(right_iris)
                    right_iris_center = (int(right_iris_center_x * width), int(right_iris_center_y * height))

                    # 동공 가장자리와 원 시각화
                    for landmark in right_iris:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)  # 빨간색으로 동공 경계점

                    # 동공 중심 시각화
                    cv2.circle(image, right_iris_center, 3, (255, 0, 0), -1)  # 파란색으로 동공 중앙 표시

            # 결과 이미지 저장
            output_path = os.path.join(output_dir, f"labeled_{filename}")
            cv2.imwrite(output_path, image)
                    # 이미지 처리 카운터 증가
            image_count += 1
            
            # 10개의 이미지 처리 후 루프 종료
            if image_count >= max_images:
                break

