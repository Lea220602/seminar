import torch
import torch.onnx
from Model.nn_models import ImprovedPupilLandmarkNet_64

# PyTorch 모델 로드
model = ImprovedPupilLandmarkNet_64()
checkpoint = torch.load('1023_fine_epoch100_with1018.pt', map_location=torch.device('cpu'))
>>>>>>> c35b485... revise code
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 더미 입력 생성 (모델의 입력 크기에 맞게 조정)
dummy_input = torch.randn(1, 1, 32, 32)

# ONNX로 내보내기
torch.onnx.export(model,               # 실행될 모델
                  dummy_input,         # 모델 입력 (또는 입력 튜플)

                  "1023_fine_epoch100_with1018.onnx",   # 모델 저장 경로
>>>>>>> c35b485... revise code
                  export_params=True,  # 모델 파일에 학습된 파라미터 가중치를 저장할지의 여부
                  opset_version=11,    # 모델을 변환할 때 사용할 ONNX 버전
                  do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                  input_names = ['input'],   # 모델의 입력값을 가리키는 이름
                  output_names = ['output'], # 모델의 출력값을 가리키는 이름
                  dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
                                'output' : {0 : 'batch_size'}})

print("Model has been converted to ONNX")