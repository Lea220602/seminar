import torch
from Model.nn_models import ImprovedPupilLandmarkNet_64

# 모델 초기화
model = ImprovedPupilLandmarkNet_64()

# 사전 훈련된 모델의 상태를 로드
pt_file_path = "./checkpoints/base/best_base_model_loss_0.001867_20241107_220941.pt"  # 여기에 .pt 파일 이름을 넣습니다.
# 여기에 .pt 파일 이름을 넣습니다.
checkpoint = torch.load(pt_file_path)
model.load_state_dict(checkpoint['model_state_dict'])  # model_state_dict 키 사용
model.eval()  # 추론 모드로 설정

# 더미 입력 텐서 생성 (배치 크기 1, 채널 1, 높이 64, 너비 64)
dummy_input = torch.randn(1, 1, 64, 64)

# ONNX로 모델 변환
onnx_file_path = "best_base_model_1108.onnx"
torch.onnx.export(model, 
                  dummy_input, 
                  onnx_file_path, 
                  export_params=True, 
                  opset_version=11, 
                  do_constant_folding=True, 
                  input_names=['input'], 
                  output_names=['output'], 
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

print(f"모델이 {onnx_file_path}로 변환되었습니다.")