import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import os
from datetime import datetime  # 날짜/시간 처리를 위해 추가

from Dataloader.dataloader import CustomImageDataset
from Model.nn_models import ImprovedPupilLandmarkNet_64, ImprovedPupilLandmarkNet_64_driver

def train_base_model(args):
    # DataLoader 설정
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    # 훈련 및 검증 데이터셋 로드
    train_dataset = CustomImageDataset(img_dir=args.train_dir, transform=transform)
    val_dataset = CustomImageDataset(img_dir=args.val_dir, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    # 모델 초기화
    model = ImprovedPupilLandmarkNet_64().to(args.device)
    criterion = nn.MSELoss()
    
    # 더 작은 learning rate 사용
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler 추가
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.2,  # 더 작은 감소율
        patience=3,   # 더 짧은 patience
        verbose=True,
        min_lr=1e-6  # 최소 learning rate 설정
    )

    return train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, args, model_type='base')

def train_driver_model(args):
    # DataLoader 설정
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    # 훈련 및 검증 데이터셋 로드
    train_dataset = CustomImageDataset(img_dir=args.train_dir, transform=transform)
    val_dataset = CustomImageDataset(img_dir=args.val_dir, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    # 기본 모델 로드
    base_model = ImprovedPupilLandmarkNet_64()
    # 기본 모델에 체크포인트 로드
    checkpoint = torch.load(args.pretrained_path, map_location=args.device)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    
    # 운전자 모델 초기화 (pretrained base_model 사용)
    model = ImprovedPupilLandmarkNet_64_driver(pretrained_model=base_model).to(args.device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    
    # Learning rate scheduler 추가
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.2,
        patience=3,
        verbose=True,
        min_lr=1e-6
    )

    return train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, args, model_type='driver')

def train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, args, model_type='base'):
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 15  # Early stopping patience

    print(f"Training {model_type} model on {args.device}")
    print(f"Total training samples: {len(train_dataloader.dataset)}")
    print(f"Total validation samples: {len(val_dataloader.dataset)}")

    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            images = images.to(args.device)
            labels = labels.to(args.device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping 추가
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{batch_idx+1}/{len(train_dataloader)}], '
                      f'Train Loss: {loss.item():.4f}, '
                      f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        avg_train_loss = train_loss / len(train_dataloader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_dataloader:
                images = images.to(args.device)
                labels = labels.to(args.device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{args.epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model and check early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # 현재 시간과 loss 값을 포함한 파일명 생성
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'best_{model_type}_model_loss_{avg_val_loss:.6f}_{timestamp}.pt'
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': best_val_loss,
            }, os.path.join(args.save_dir, filename))
            print(f'New best model saved: {filename}')
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break
    
    # Save final model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_filename = f'final_{model_type}_model_loss_{avg_val_loss:.6f}_{timestamp}.pt'
    
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
    }, os.path.join(args.save_dir, final_filename))

    print(f'Final model saved: {final_filename}')
    
    # 인자를 텍스트 파일로 저장 (모델 저장 이름과 동일하게)
    args_file_path = os.path.join(args.save_dir, f"{final_filename.replace('.pt', '')}_arguments.txt")
    with open(args_file_path, 'w') as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

    print(f'Arguments saved to: {args_file_path}')
    print(f'{model_type.capitalize()} model training finished!')
    return model

def parse_args():
    parser = argparse.ArgumentParser(description='Train pupil landmark detection model')
    parser.add_argument('--mode', type=str, required=True, choices=['base', 'finetune'],
                        help='Training mode: base or finetune')
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Directory containing the training dataset')
    parser.add_argument('--val_dir', type=str, required=True,
                        help='Directory containing the validation dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save the model checkpoints')
    parser.add_argument('--pretrained_path', type=str,
                        help='Path to pretrained base model (required for finetune mode)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (e.g., cuda:0, cuda:1, cuda:2, cpu)')
    
    args = parser.parse_args()
    
    # device 설정
    if not torch.cuda.is_available() and 'cuda' in args.device:
        print("CUDA is not available. Using CPU instead.")
        args.device = 'cpu'
    args.device = torch.device(args.device)
    
    if args.mode == 'finetune' and args.pretrained_path is None:
        parser.error("finetune mode requires --pretrained_path")
    
    return args

if __name__ == "__main__":
    args = parse_args()
    
    if args.mode == 'base':
        train_base_model(args)
    else:  # finetune
        train_driver_model(args)

# # 기본 모델 학습
# python train.py --mode base \
#                 --train_dir /workspace/data/v4.2.2_pupil/train_data \
#                 --val_dir /workspace/data/v4.2.2_pupil/test_data \
#                 --save_dir ./checkpoints \
#                 --device cuda:2
# # Fine-tuning
# python train.py --mode finetune \
#                 --train_dir /workspace/data/project1_rev_train_lea/train \
#                 --val_dir /workspace/data/project1_rev_train_lea/val \
#                 --save_dir ./checkpoints \
#                 --pretrained_path ./checkpoints/best_base_model.pt \
#                 --learning_rate 0.0001 \
#                 --device cuda:2

