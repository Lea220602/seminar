# import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

class ImprovedPupilLandmarkNet_64(nn.Module):
    def __init__(self):
        super(ImprovedPupilLandmarkNet_64, self).__init__()
        # 컨볼루션 레이어
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)  # 더 큰 초기 필터
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        # 완전 연결 레이어
        self.fc1 = nn.Linear(4096, 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 64)
        self.bn7 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 2)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.3)  # 약간 높은 dropout 비율
        
        # Max Pooling
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # 컨볼루션 레이어 (ReLU 활성화 함수, Batch Normalization, Max Pooling 적용)
        x1 = self.pool(F.relu(self.bn1(self.conv1(x))))
        x2 = self.pool(F.relu(self.bn2(self.conv2(x1))))
        x3 = self.pool(F.relu(self.bn3(self.conv3(x2))))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        
        # 잔차 연결
        x4 = x4 + x3
        
        # Flatten
        x = x4.view(-1, 64 * 8 * 8)
        
        # 완전 연결 레이어
        x = self.dropout(F.relu(self.bn5(self.fc1(x))))
        x = self.dropout(F.relu(self.bn6(self.fc2(x))))
        x = self.dropout(F.relu(self.bn7(self.fc3(x))))
        x = self.fc4(x)
        return x

#모델 테스트
# improved_pupil_model = ImprovedPupilLandmarkNet_64()
# test_input = torch.randn(1, 1, 64, 64)
# test_output = improved_pupil_model(test_input)
# print(test_output.shape)  # torch.Size([1, 2])