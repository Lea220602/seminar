import torch
import torch.onnx
from Model.nn_models import ImprovedPupilLandmarkNet_64, ImprovedPupilLandmarkNet_64_driver

# 기본 모델 로드
base_model = ImprovedPupilLandmarkNet_64()
# 운전자 모델 초기화
#model = ImprovedPupilLandmarkNet_64_driver(pretrained_model=base_model)

# 체크포인트 로드
#checkpoint = torch.load('./checkpoints/base/best_driver_model_loss_0.002923_20241106_215820.pt', map_location=torch.device('cpu'))
#model.load_state_dict(checkpoint['model_state_dict'])
# 체크포인트 로드
checkpoint = torch.load('./checkpoints/base/best_base_model_loss_0.000298_20241106_204906.pt', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # strict=False 추가
model.eval()

# 더미 입력 생성 (64x64로 수정)
dummy_input = torch.randn(1, 1, 64, 64)

# ONNX로 내보내기
torch.onnx.export(model,               # 실행될 모델
                 dummy_input,          # 모델 입력 (또는 입력 튜플)
                 "best_driver_model_loss_0.002923_20241106_215820.onnx",   # 모델 저장 경로
                 export_params=True,   # 모델 파일에 학습된 파라미터 가중치를 저장할지의 여부
                 opset_version=11,     # 모델을 변환할 때 사용할 ONNX 버전
                 do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                 input_names=['input'],    # 모델의 입력값을 가리키는 이름
                 output_names=['output'],  # 모델의 출력값을 가리키는 이름
                 dynamic_axes={'input': {0: 'batch_size'},    # 가변적인 길이를 가진 차원
                             'output': {0: 'batch_size'}})

print("Model has been converted to ONNX")