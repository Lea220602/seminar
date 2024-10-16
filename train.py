import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# 이전에 정의한 CustomImageDataset과 모델을 import
from Dataloader.dataloader import CustomImageDataset
from Model.nn_models import ImprovedPupilLandmarkNet_64  # 여기에 실제 모델 클래스 이름을 사용하세요

# 하이퍼파라미터 설정
batch_size = 32
learning_rate = 0.001
num_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DataLoader 설정
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

img_dir = '/Users/hong-eun-yeong/Codes/train/Detroit_cam0_apillar_gray_006_M_asian_ng_240425_0616_sunny/total/'
dataset = CustomImageDataset(img_dir=img_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 모델 초기화
model = ImprovedPupilLandmarkNet_64().to(device)

# 사용자 정의 Loss 함수
class CustomMSELoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, outputs, labels):
        return self.mse(outputs, labels[:, :2])  # x, y 좌표에 대해서만 MSE 계산


# Loss 함수 정의
criterion = CustomMSELoss()  # Mean Squared Error loss for regression

# Optimizer 설정
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습 루프
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        # 모델 출력
        outputs = model(images)
        
        # Loss 계산
        loss = criterion(outputs, labels)
        
        # 역전파 및 옵티마이저 스텝
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

print('Training finished!')

# 모델 저장
torch.save(model.state_dict(), 'landmark_detection_model.pth')