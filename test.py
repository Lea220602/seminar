import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# 이전에 정의한 CustomImageDataset과 모델을 import
from Dataloader.dataloader import CustomImageDataset
from Seminar.Model.nn_models import ImprovedPupilLandmarkNet_64

# 테스트 데이터 경로 설정
test_img_dir = '/Users/hong-eun-yeong/Codes/test_dataset'
output_dir = '/Users/hong-eun-yeong/Codes/test_results'

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

# 하이퍼파라미터 설정
batch_size = 1  # 시각화를 위해 배치 크기를 1로 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DataLoader 설정
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

test_dataset = CustomImageDataset(img_dir=test_img_dir, transform=transform, is_train=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 모델 로드
model = ImprovedPupilLandmarkNet_64().to(device)
checkpoint = torch.load('best_landmark_detection_model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 테스트 및 시각화 함수
def test_and_visualize(model, dataloader, device):
    model.eval()
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            # 이미지, 레이블, 예측값을 CPU로 이동하고 numpy 배열로 변환
            image = images[0].cpu().numpy().transpose(1, 2, 0)
            label = labels[0].cpu().numpy()
            pred = outputs[0].cpu().numpy()
            
            # 이미지 정규화 해제
            image = (image * 0.229 + 0.485) * 255
            image = image.astype(np.uint8)
            
            # 그레이스케일 이미지를 RGB로 변환
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # 이미지에 실제 랜드마크와 예측된 랜드마크 표시
            h, w = image.shape[:2]
            cv2.circle(image, (int(label[0]*w), int(label[1]*h)), 1, (0, 255, 0), -1)  # 실제 랜드마크 (녹색)
            cv2.circle(image, (int(pred[0]*w), int(pred[1]*h)), 1, (255, 0, 0), -1)   # 예측된 랜드마크 (빨간색)
            
            # 결과 저장
            plt.figure(figsize=(8, 8))
            plt.imshow(image)
            plt.title(f'Sample {i+1} - Green: Ground Truth, Red: Prediction')
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f'result_{i+1}.png'))
            plt.close()
            
            print(f'Processed image {i+1}/{len(dataloader)}')

# 모델 테스트 및 결과 시각화
test_and_visualize(model, test_dataloader, device)

print(f"All results saved in {output_dir}")
