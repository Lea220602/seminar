import cv2
import numpy as np
import os
# 너무 어두운 이미지 제외
def evaluate_image_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    std_dev = np.std(gray)
    return mean_brightness, std_dev

def process_images(input_dir, label_dir, output_dir, output_label_dir, brightness_threshold=5, contrast_threshold=3):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    processed_count = 0
    total_count = 0

    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            total_count += 1
            image_path = os.path.join(input_dir, filename)
            label_path = os.path.join(label_dir, filename.replace('.png', '.txt'))

            image = cv2.imread(image_path)
            if image is None:
                print(f"이미지를 읽을 수 없습니다: {image_path}")
                continue

            brightness, contrast = evaluate_image_quality(image)
            #print(f"처리 중: {filename}, 밝기: {brightness:.2f}, 대비: {contrast:.2f}")

            if brightness <= brightness_threshold or contrast <= contrast_threshold:
                # 이미지 이동
                output_image_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_image_path, image)
                os.remove(image_path)  # 원본 이미지 삭제
                
                # 레이블 이동
                if os.path.exists(label_path):
                    output_label_path = os.path.join(output_label_dir, filename.replace('.png', '.txt'))
                    with open(label_path, 'r') as f_in, open(output_label_path, 'w') as f_out:
                        f_out.write(f_in.read())
                    os.remove(label_path)  # 원본 레이블 파일 삭제
                else:
                    print(f"레이블 파일을 찾을 수 없습니다: {label_path}")
                
                processed_count += 1
                #print(f"이동됨: {filename}")
            #else:
                #print(f"유지됨: {filename}")

    print(f"총 {total_count}개 중 {processed_count}개의 이미지가 처리되었습니다.")

# 디렉토리 설정
base_dir = '/Users/hong-eun-yeong/Codes/1_crop_output'
right_input_dir = os.path.join(base_dir, 'Right')
left_input_dir = os.path.join(base_dir, 'Left')
right_label_dir = os.path.join(base_dir, 'right_label')
left_label_dir = os.path.join(base_dir, 'left_label')

# 새로운 출력 디렉토리 설정
output_base_dir = '/Users/hong-eun-yeong/Codes/2_filtered_output'
right_output_dir = os.path.join(output_base_dir, 'Right')
left_output_dir = os.path.join(output_base_dir, 'Left')
right_output_label_dir = os.path.join(output_base_dir, 'right_label')
left_output_label_dir = os.path.join(output_base_dir, 'left_label')

# 모든 필요한 디렉토리 생성
for directory in [output_base_dir, right_output_dir, left_output_dir, right_output_label_dir, left_output_label_dir]:
    os.makedirs(directory, exist_ok=True)

# 이미지 처리
print("오른쪽 이미지 처리 시작")
process_images(right_input_dir, right_label_dir, right_output_dir, right_output_label_dir)
print("왼쪽 이미지 처리 시작")
process_images(left_input_dir, left_label_dir, left_output_dir, left_output_label_dir)

print("이미지 필터링 완료!")
