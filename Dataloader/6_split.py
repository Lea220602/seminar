import os
import random
import shutil

def create_test_dataset(source_dir, test_dir, test_ratio=0.2):
    # 소스 디렉토리의 모든 파일 목록 가져오기
    all_files = os.listdir(source_dir)
    
    # 이미지 파일만 선택 (jpg 또는 png)
    image_files = [f for f in all_files if f.endswith('.jpg') or f.endswith('.png')]
    
    # 테스트 세트로 선택할 이미지 수 계산
    num_test = int(len(image_files) * test_ratio)
    
    # 무작위로 테스트 이미지 선택
    test_images = random.sample(image_files, num_test)
    
    # 테스트 디렉토리 생성
    os.makedirs(test_dir, exist_ok=True)
    
    # 선택된 이미지와 해당 레이블 파일을 테스트 디렉토리로 이동
    for image in test_images:
        # 이미지 파일 이동
        src_image = os.path.join(source_dir, image)
        dst_image = os.path.join(test_dir, image)
        shutil.move(src_image, dst_image)
        
        # 레이블 파일 이동 (확장자를 .txt로 변경)
        label = os.path.splitext(image)[0] + '.txt'
        src_label = os.path.join(source_dir, label)
        dst_label = os.path.join(test_dir, label)
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)
        else:
            print(f"Warning: Label file not found for {image}")
    
    print(f"Moved {len(test_images)} images and their labels to {test_dir}")

# 사용 예
source_directory = '/Users/hong-eun-yeong/Codes/combined_dataset'
test_directory = '/Users/hong-eun-yeong/Codes/test_dataset'
create_test_dataset(source_directory, test_directory, test_ratio=0.2)