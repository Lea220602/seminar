import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import transforms
import math
import random
from PIL import Image


def rotate_point(x, y, angle, cx, cy):
    """주어진 각도(angle)만큼 좌표 (x, y)를 중심점 (cx, cy) 기준으로 회전"""
    radians = math.radians(angle)
    cos_val = math.cos(radians)
    sin_val = math.sin(radians)
    nx = cos_val * (x - cx) - sin_val * (y - cy) + cx
    ny = sin_val * (x - cx) + cos_val * (y - cy) + cy
    return nx, ny

def enhance_image(image,label):
    original_height, original_width = image.shape[:2]
    # 1. 이미지 크기 증가 (32x32에서 64x64로)
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_CUBIC)

        
    # 2. 선명도 향상
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    image = cv2.filter2D(image, -1, kernel)
    
    # 3. 대비 조정
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    image = clahe.apply(image)
    
    return image, label

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, is_train=True):
        self.img_labels = [file_name for file_name in os.listdir(img_dir) if file_name.endswith(('.txt'))]
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        txt_file_name = self.img_labels[idx]
        
        # jpg 또는 png 파일 찾기
        img_file_name = os.path.splitext(txt_file_name)[0]
        if os.path.exists(os.path.join(self.img_dir, img_file_name + '.jpg')):
            img_path = os.path.join(self.img_dir, img_file_name + '.jpg')
        elif os.path.exists(os.path.join(self.img_dir, img_file_name + '.png')):
            img_path = os.path.join(self.img_dir, img_file_name + '.png')
        else:
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {img_file_name}")
        
        # OpenCV를 사용하여 이미지를 그레이스케일로 읽음
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # 텍스트 파일을 읽어 랜드마크 좌표를 가져옴
        with open(os.path.join(self.img_dir, txt_file_name), 'r') as file:
            data = list(filter(None, file.read().split()))
        
        if len(data) < 2:
            print(f"경고: {txt_file_name}에 충분한 데이터가 없습니다. 기본값을 사용합니다.")
            label = torch.tensor([0.0, 0.0], dtype=torch.float32)
        else:
            label = torch.tensor([float(data[0]), float(data[1])], dtype=torch.float32)
        
 
        
        #image, label = enhance_image(image, label)
        
        if self.is_train:
            # 학습 시에만 회전 적용
            angle = random.uniform(-0, 5)
            
            # OpenCV를 사용하여 이미지 회전
            height, width = image.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
            image = cv2.warpAffine(image, rotation_matrix, (width, height))
            
            # 좌표 회전
            x, y = label[0].item() * width, label[1].item() * height
            new_x, new_y = rotate_point(x, y, -angle, width / 2, height / 2)
            
            label[0] = new_x / width
            label[1] = new_y / height
        
        # 이미지를 PIL Image로 변환
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def __visualize__(self, idx):
        original_is_train = self.is_train
        self.is_train = True
                
        image, label = self.__getitem__(idx)
        
        self.is_train = original_is_train  
              
        # 이미지가 텐서인 경우 numpy 배열로 변환
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()  # CHW -> HWC
            image = (image * 255).astype(np.uint8)  # 0-1 범위를 0-255 범위로 변환

            
        height, width = image.shape[:2]
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(image, cmap='gray')
        
        x_coord, y_coord = label[0].item(), label[1].item()
        #print(f"Original coordinates: x={x_coord:.4f}, y={y_coord:.4f}")
    # 이미지 크기에 맞게 좌표 조정
        x_pixel = x_coord * width
        y_pixel = y_coord * height
        ax.plot(x_pixel, y_pixel, 'ro', markersize=3)
        ax.set_title(f"Score: {label[2].item():.2f}")
        #print(f"Image {idx}: x={x_pixel:.2f}, y={y_pixel:.2f}")
        
        plt.show()

#Transformations 정의
transform = transforms.Compose([
    transforms.ToTensor(),  # Tensor로 변환 0-1
])

# Example usage
img_dir = '/Users/hong-eun-yeong/Codes/combined_dataset'
dataset = CustomImageDataset(img_dir=img_dir, transform=transform)
print(f"Dataset size: {dataset.__len__()}")

#시각화할 인덱스 지정
#indices_to_visualize = [1, 2, 3]  # 여러 이미지를 시각화하도록 변경
# for idx in indices_to_visualize:
#     dataset.__visualize__(idx)
