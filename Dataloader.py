import os
import math
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
import torch
import random

# 역정규화를 위한 함수 정의
def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)

# 좌표 회전 함수
def rotate_point(x, y, angle, cx, cy):
    """주어진 각도(angle)만큼 좌표 (x, y)를 중심점 (cx, cy) 기준으로 회전"""
    radians = math.radians(angle)
    cos_val = math.cos(radians)
    sin_val = math.sin(radians)
    nx = cos_val * (x - cx) - sin_val * (y - cy) + cx
    ny = sin_val * (x - cx) + cos_val * (y - cy) + cy
    return nx, ny

class DynamicResize:
    def __init__(self, scale_factor_w, scale_factor_h, final_size):
        self.scale_factor_w = scale_factor_w  # 가로 스케일링 비율
        self.scale_factor_h = scale_factor_h  # 세로 스케일링 비율
        self.final_size = final_size          # 최종 크기

    def __call__(self, img):
        w, h = img.size  # 원래 이미지의 크기 (PIL.Image의 size는 (width, height) 반환)
        new_size = (int(w * self.scale_factor_w), int(h * self.scale_factor_h))  # 스케일링
        img = img.resize(new_size)  # 첫 번째 리사이즈
        
        # 두 번째 리사이즈 (고정된 최종 크기)
        img = img.resize((self.final_size, self.final_size))
        return img

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_labels = [file_name for file_name in os.listdir(img_dir) if file_name.endswith(('.txt'))]
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.angle = 0

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        txt_file_name = self.img_labels[idx]
        jpg_file_name = os.path.splitext(txt_file_name)[0] + '.jpg'
        img_path = os.path.join(self.img_dir, jpg_file_name)
        
        # PIL 이미지를 읽음
        image = Image.open(img_path).convert("RGB")
        
        # 텍스트 파일을 읽어 바운딩 박스와 랜드마크 좌표를 가져옴
        with open(os.path.join(self.img_dir, txt_file_name), 'r') as file:
            data = list(filter(None, file.read().split()))
        
        label_dict = {
            'class': data[0],
            'x': float(data[1]),
            'y': float(data[2]),
            'w': float(data[3]),
            'h': float(data[4]),
            'points': [(float(data[i]), float(data[i + 1])) for i in range(5, len(data), 3)]
        }
        
        # 이미지와 레이블에 변환 적용
        if self.transform: 
            
            # -15도에서 15도 사이의 랜덤 각도 생성
            angle = random.uniform(-30, 30)

            # # 이미지를 회전
            image = image.rotate(angle)

            # 바운딩 박스 좌표 변환 (회전)
            img_width, img_height = image.size
            center_x = label_dict['x'] * img_width
            center_y = label_dict['y'] * img_height
            center_x, center_y = rotate_point(center_x, center_y, -angle, img_width / 2, img_height / 2)

            # 랜드마크 좌표 변환 (회전)
            transformed_points = []
            for (x_coord, y_coord) in label_dict['points']:
                x_coord = x_coord * img_width
                y_coord = y_coord * img_height
                new_x, new_y = rotate_point(x_coord, y_coord, -angle, img_width / 2, img_height / 2)
                transformed_points.append((new_x / img_width, new_y / img_height))
            
            # 변환된 좌표 반영
            label_dict['x'] = center_x / img_width
            label_dict['y'] = center_y / img_height
            label_dict['points'] = transformed_points
            
            image = self.transform(image)

        return image, label_dict
    
    def __visualize__(self, idx):
        image, label_dict = self.__getitem__(idx)
        image = denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = image.permute(1, 2, 0).numpy()  # Tensor -> Numpy 변환
        image = (image * 255).astype('uint8')  # uint8로 변환
        img_height, img_width = image.shape[:2]
        center_x = label_dict['x'] * img_width
        center_y = label_dict['y'] * img_height
        width = label_dict['w'] * img_width
        height = label_dict['h'] * img_height
        x = center_x - (width / 2)
        y = center_y - (height / 2)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        bbox = patches.Rectangle(
            (x, y),
            width,
            height,
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(bbox)
        for (x_coord, y_coord) in label_dict['points']:
            x_coord *= img_width
            y_coord *= img_height
            ax.plot(x_coord, y_coord, 'bo')
        plt.show()

# Transformations 정의
transform = transforms.Compose([
    DynamicResize(0.1, 0.1, 640), 
    transforms.Grayscale(num_output_channels=3),  # 그레이스케일 변환 (3채널 유지)
    transforms.ToTensor(),  # Tensor로 변환 0-1
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화

])

# Example usage
img_dir = '/Users/hong-eun-yeong/Desktop/Seminar/0-4_4_True/'
dataset = CustomImageDataset(img_dir=img_dir, transform=transform)
dataset.__len__()

# 시각화할 인덱스 지정
indices_to_visualize = [1]
for idx in indices_to_visualize:
    dataset.__visualize__(idx)

