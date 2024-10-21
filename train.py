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

img_dir = '/Users/hong-eun-yeong/Codes/combined_dataset'
dataset = CustomImageDataset(img_dir=img_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 모델 초기화
model = ImprovedPupilLandmarkNet_64().to(device)

# Loss 함수 정의
criterion = nn.MSELoss()  # 일반 MSE loss 사용

# Optimizer 설정
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 최고 성능 추적을 위한 변수 초기화
best_loss = float('inf')

# 학습 루프
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
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
        
        epoch_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
    
    # 에포크 평균 손실 계산
    avg_loss = epoch_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    
    # 최고 성능 모델 저장
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
        }, 'best_landmark_detection_model.pt')
        print(f'New best model saved with loss: {best_loss:.4f}')

print('Training finished!')

# 최종 모델 저장
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_loss,
}, 'final_landmark_detection_model.pt')
