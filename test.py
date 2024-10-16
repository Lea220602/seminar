import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
import numpy as np

from Dataloader.dataloader import CustomImageDataset, transform, enhance_image
from Model.nn_models import ImprovedPupilLandmarkNet_64

# 설정
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model_path = './landmark_detection_model.pth'

test_img_dir = '/path/to/test/images/'  # 테스트 이미지 디렉토리 경로를 지정하세요

# 모델 로드
model = ImprovedPupilLandmarkNet_64().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# 테스트 데이터셋 및 DataLoader 설정
test_dataset = CustomImageDataset(img_dir=test_img_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 테스트 함수
def test_model(model, test_loader, device):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels[:, :2])  # x, y 좌표에 대해서만 loss 계산
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f"Average test loss: {avg_loss:.4f}")

# 단일 이미지에 대한 예측 및 시각화 함수
def predict_and_visualize(model, image_path, device):
    # 이미지 로드 및 전처리
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = enhance_image(image)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 예측
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)

    # 예측 결과 시각화
    pred_x, pred_y = output[0].cpu().numpy()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.plot(pred_x, pred_y, 'ro', markersize=5)
    plt.title(f"Predicted landmark: ({pred_x:.2f}, {pred_y:.2f})")
    plt.axis('off')
    plt.show()

# 모델 테스트
test_model(model, test_loader, device)

# 단일 이미지에 대한 예측 및 시각화
sample_image_path = '/path/to/sample/image.jpg'  # 샘플 이미지 경로를 지정하세요
predict_and_visualize(model, sample_image_path, device)
