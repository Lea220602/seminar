import os
import shutil

# 원본 디렉토리 설정
source_dirs = [
    ('/Users/hong-eun-yeong/Codes/0_augmented_output', 'aug_'),
    ('/Users/hong-eun-yeong/Codes/1_crop_output', 'crop_')
]

# 대상 디렉토리 설정
target_dir = '/Users/hong-eun-yeong/Codes/combined_dataset'

# 대상 디렉토리 생성
os.makedirs(target_dir, exist_ok=True)

# 이미지와 레이블 파일 이동 함수
def move_files(src_img_dir, src_label_dir, target_dir, prefix):
    for filename in os.listdir(src_img_dir):
        if filename.endswith('.png'):
            src_img_path = os.path.join(src_img_dir, filename)
            new_img_filename = f"{prefix}{filename}"
            target_img_path = os.path.join(target_dir, new_img_filename)
            shutil.copy2(src_img_path, target_img_path)

            # 대응하는 레이블 파일 이동
            label_filename = filename.replace('.png', '.txt')
            new_label_filename = f"{prefix}{label_filename}"
            src_label_path = os.path.join(src_label_dir, label_filename)
            target_label_path = os.path.join(target_dir, new_label_filename)
            if os.path.exists(src_label_path):
                shutil.copy2(src_label_path, target_label_path)
            else:
                print(f"경고: 레이블 파일을 찾을 수 없습니다: {src_label_path}")

# 각 소스 디렉토리에서 파일 이동
for source_dir, prefix in source_dirs:
    for side in ['left', 'right']:
        src_img_dir = os.path.join(source_dir, side)
        src_label_dir = os.path.join(source_dir, f'{side}_label')
        
        move_files(src_img_dir, src_label_dir, target_dir, prefix)

print("파일 재구성 완료!")