import cv2
import numpy as np
import os
import shutil
# 오른쪽 레이블이 너무 코너에 있는 경우 제외
# 디렉토리 설정
base_dir = '/Users/hong-eun-yeong/Codes/train'
png_dir = 'eyes_png'
txt_dir = 'total_txt'
blind_dir = 'blind_png'
new_base_dir = '/Users/hong-eun-yeong/Codes/0_filtered_output'
new_image_dir = os.path.join(new_base_dir, '0_filtered_images')
new_label_dir = os.path.join(new_base_dir, '0_filtered_labels')

# 새로운 폴더 생성
os.makedirs(new_image_dir, exist_ok=True)
os.makedirs(new_label_dir, exist_ok=True)

# 이미지와 레이블 파일 처리 함수
def process_files(base_dir, subdirs):
    for subdir in subdirs:
        txt_path = os.path.join(base_dir, subdir, txt_dir)
        blind_path = os.path.join(base_dir, subdir, blind_dir)
        
        txt_files = [f for f in os.listdir(txt_path) if f.endswith('.txt')]
        
        for txt_file in txt_files:
            txt_full_path = os.path.join(txt_path, txt_file)
            base_name = os.path.splitext(txt_file)[0]
            blind_img_path = os.path.join(blind_path, f"{base_name}.png")
            
            if os.path.exists(blind_img_path):
                with open(txt_full_path, 'r') as file:
                    content = file.read().strip()
                    values = content.split()
                    if len(values) >= 7:
                        extracted = [float(values[i]) for i in range(4, 8)]
                        extracted = [extracted[0] * 640, extracted[1] * 480, extracted[2] * 640, extracted[3] * 480]
                        
                        original_right_x = extracted[0]
                        
                        if original_right_x < 275:
                            # 이미지 파일 이동
                            new_img_path = os.path.join(new_image_dir, f"{base_name}.png")
                            shutil.move(blind_img_path, new_img_path)
                            
                            # 레이블 파일 이동
                            new_txt_path = os.path.join(new_label_dir, txt_file)
                            shutil.move(txt_full_path, new_txt_path)
                            
                            print(f"이동됨: {base_name}")

# 메인 실행 부분
if __name__ == "__main__":
    if os.path.exists(base_dir):
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        process_files(base_dir, subdirs)
        print("필터링 및 이동 완료!")
    else:
        print(f"디렉토리를 찾을 수 없습니다: {base_dir}")

