import cv2
import numpy as np
import os
import albumentations as A
# 이미지 선명도 증가

def sharpen_image(image):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(image, -1, kernel)

aug_pipeline = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.5),
    A.GaussNoise(p=0.3),
])

def process_and_augment_images(input_dir, label_dir, output_dir, output_label_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            image_path = os.path.join(input_dir, filename)
            label_path = os.path.join(label_dir, filename.replace('.png', '.txt'))

            image = cv2.imread(image_path)
            if image is None:
                print(f"이미지를 읽을 수 없습니다: {image_path}")
                continue

            # 이미지 선명화
            sharpened = sharpen_image(image)

            # Augmentation 적용
            augmented = aug_pipeline(image=sharpened)['image']

            # 새 이미지 저장
            output_image_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_image_path, augmented)

            # 레이블 파일 복사
            if os.path.exists(label_path):
                output_label_path = os.path.join(output_label_dir, filename.replace('.png', '.txt'))
                with open(label_path, 'r') as f_in, open(output_label_path, 'w') as f_out:
                    f_out.write(f_in.read())
            else:
                print(f"레이블 파일을 찾을 수 없습니다: {label_path}")

    print(f"{input_dir}의 모든 이미지 처리 완료")

# 디렉토리 설정
base_dir = '/Users/hong-eun-yeong/Codes/1_crop_output'
right_input_dir = os.path.join(base_dir, 'Right')
left_input_dir = os.path.join(base_dir, 'Left')
right_label_dir = os.path.join(base_dir, 'right_label')
left_label_dir = os.path.join(base_dir, 'left_label')

# 새로운 출력 디렉토리 설정
output_base_dir = '/Users/hong-eun-yeong/Codes/0_augmented_output'
right_output_dir = os.path.join(output_base_dir, 'Right')
left_output_dir = os.path.join(output_base_dir, 'Left')
right_output_label_dir = os.path.join(output_base_dir, 'right_label')
left_output_label_dir = os.path.join(output_base_dir, 'left_label')

# 이미지 처리 및 augmentation
print("오른쪽 이미지 처리 시작")
process_and_augment_images(right_input_dir, right_label_dir, right_output_dir, right_output_label_dir)
print("왼쪽 이미지 처리 시작")
process_and_augment_images(left_input_dir, left_label_dir, left_output_dir, left_output_label_dir)

print("이미지 처리 및 augmentation 완료!")