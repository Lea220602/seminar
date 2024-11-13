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
        
        # 풀링 레이어
        self.pool = nn.MaxPool2d(2, 2)
        
        # 완전 연결 레이어
        self.fc1 = nn.Linear(4096, 64)  # 4096 = 64 channels * 8 * 8
        self.bn5 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 2)  # 64에서 2(x, y 좌표)로 출력
        
        # Dropout
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # 컨볼루션 레이어 (ReLU 활성화 함수, Batch Normalization, Max Pooling 적용)
        x1 = self.pool(F.relu(self.bn1(self.conv1(x))))
        x2 = self.pool(F.relu(self.bn2(self.conv2(x1))))
        x3 = self.pool(F.relu(self.bn3(self.conv3(x2))))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        
        # 잔차 연결
        x4 = x4 + x3
        
        # Flatten
        x = x4.view(x.size(0), -1)  # Flatten the output
        
        # 완전 연결 레이어
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#모델 테스트
# improved_pupil_model = ImprovedPupilLandmarkNet_64()
# test_input = torch.randn(1, 1, 64, 64)
# test_output = improved_pupil_model(test_input)
# print(test_output.shape)  # torch.Size([1, 2])




class ImprovedPupilLandmarkNet_64_driver(nn.Module):
    def __init__(self, pretrained_model):
        super(ImprovedPupilLandmarkNet_64_driver, self).__init__()
        self.base_model = pretrained_model
        
        # 기본 모델의 마지막 레이어를 제외한 모든 레이어를 동결
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # 새로운 fully connected 레이어 추가
        self.fc1 = nn.Linear(4096, 64)  # 4096 = 64 channels * 8 * 8
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # 기본 모델의 feature extraction 부분 사용
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = F.relu(x)
        x = self.base_model.pool(x)  # 32x32
        
        x = self.base_model.conv2(x)
        x = self.base_model.bn2(x)
        x = F.relu(x)
        x = self.base_model.pool(x)  # 16x16
        
        x = self.base_model.conv3(x)
        x = self.base_model.bn3(x)
        x = F.relu(x)
        x = self.base_model.pool(x)  # 8x8
        
        x = self.base_model.conv4(x)
        x = self.base_model.bn4(x)
        x = F.relu(x)  # [batch_size, 64, 8, 8]
        
        # Flatten
        x = x.view(x.size(0), -1)  # [batch_size, 64 * 8 * 8] = [batch_size, 4096]
        
        # 새로운 FC 레이어
        x = F.relu(self.fc1(x))  # [batch_size, 64]
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc2(x)  # [batch_size, 2]
        
        return x

    def freeze_base_model(self):
        for param in self.base_model.parameters():
            param.requires_grad = False



# 1. 기본 모델 학습
# base_model = ImprovedPupilLandmarkNet_64()
# # 일반 데이터로 학습
# train_base_model(base_model, general_dataset)

# # 2. 운전자 모델 생성 및 fine-tuning # 학습률을 낮춰서 미세조정
# driver_model = ImprovedPupilLandmarkNet_64_driver(pretrained_model=base_model)
# driver_model.freeze_base_model()  # 기본 모델 가중치 동결
# # 운전자 데이터로 fine-tuning
# train_driver_model(driver_model, driver_dataset)


import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedPupilLandmarkNet_64_v2(nn.Module):
    def __init__(self):
        super(ImprovedPupilLandmarkNet_64_v2, self).__init__()
        # 컨볼루션 레이어
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # 작은 필터와 더 많은 채널
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # 추가 Residual Block
        self.conv_res = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_res = nn.BatchNorm2d(128)

        # 풀링 레이어
        self.pool = nn.MaxPool2d(2, 2)
        
        # 완전 연결 레이어
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 2)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # 컨볼루션 레이어 (ReLU, Batch Norm, Max Pooling 적용)
        x1 = self.pool(F.relu(self.bn1(self.conv1(x))))
        x2 = self.pool(F.relu(self.bn2(self.conv2(x1))))
        x3 = self.pool(F.relu(self.bn3(self.conv3(x2))))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        
        # Residual Connection 추가
        res = F.relu(self.bn_res(self.conv_res(x4)))
        x4 = x4 + res  # 잔차 연결

        # Flatten
        x = x4.view(x.size(0), -1)
        
        # 완전 연결 레이어
        x = self.dropout(F.relu(self.bn5(self.fc1(x))))
        x = self.fc2(x)
        return x

class ImprovedPupilLandmarkNet_64_driver_v2(nn.Module):
    def __init__(self, pretrained_model):
        super(ImprovedPupilLandmarkNet_64_driver_v2, self).__init__()
        self.base_model = pretrained_model
        
        # 기본 모델의 마지막 레이어를 제외한 모든 레이어를 동결
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 새로운 fully connected 레이어 추가
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # 기본 모델의 feature extraction 부분 사용
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = F.relu(x)
        x = self.base_model.pool(x)
        
        x = self.base_model.conv2(x)
        x = self.base_model.bn2(x)
        x = F.relu(x)
        x = self.base_model.pool(x)
        
        x = self.base_model.conv3(x)
        x = self.base_model.bn3(x)
        x = F.relu(x)
        x = self.base_model.pool(x)
        
        x = self.base_model.conv4(x)
        x = self.base_model.bn4(x)
        x = F.relu(x)
        
        # Residual Connection 적용
        res = F.relu(self.base_model.bn_res(self.base_model.conv_res(x)))
        x = x + res  # 잔차 연결
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # 새로운 FC 레이어
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
