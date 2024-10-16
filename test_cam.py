import cv2
import torch
import numpy as np
from torchvision import transforms
from Model.nn_models import ImprovedPupilLandmarkNet_64

# 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImprovedPupilLandmarkNet_64().to(device)
model.load_state_dict(torch.load('landmark_detection_model.pth'))
model.eval()

# 전처리 변환
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# 웹캠 캡처 설정
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임을 그레이스케일로 변환
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 이미지 크기 조정 (모델 입력 크기에 맞게)
    resized_frame = cv2.resize(gray_frame, (64, 64))
    
    # 전처리
    input_tensor = transform(resized_frame).unsqueeze(0).to(device)
    
    # 모델 예측
    with torch.no_grad():
        output = model(input_tensor)
    
    # 예측 결과를 원본 프레임 크기에 맞게 조정
    pred_x, pred_y = output[0].cpu().numpy()
    frame_h, frame_w = frame.shape[:2]
    pred_x = int(pred_x * frame_w)
    pred_y = int(pred_y * frame_h)
    
    # 예측된 랜드마크를 프레임에 그리기
    cv2.circle(frame, (pred_x, pred_y), 5, (0, 255, 0), -1)
    
    # 결과 표시
    cv2.imshow('Pupil Landmark Detection', frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()