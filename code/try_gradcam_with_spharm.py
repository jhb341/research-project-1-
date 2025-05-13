import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from spharmnet import SPHARMNet
# TODO: 실제 SPHARM-Net 데이터셋 로더 경로로 수정하세요
from spharmnet.datasets import SphereDataset

from gradcam_module import GradCam, visualize_1d_heatmap


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    for features, labels in loader:
        features = features.to(device)     # (B, C, N_basis)
        labels   = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)          # (B, num_classes)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)
    return running_loss / len(loader.dataset)


def validate(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels   = labels.to(device)
            outputs = model(features)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item() * features.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = running_loss / len(loader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy


def run_training(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # 데이터셋 및 로더
    train_ds = SphereDataset(args.data_dir, split='train')
    val_ds   = SphereDataset(args.data_dir, split='val')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 모델, 옵티마이저, 스케줄러
    model = SPHARMNet(lmax=args.lmax,
                      n_out=args.num_classes,
                      use_batchnorm=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    loss_fn = nn.CrossEntropyLoss()

    # Grad-CAM 준비 (최종 합성곱 레이어 지정)
    # SPHARMNet의 마지막 conv 블록 이름을 실제로 확인하여 수정하세요
    target_layer = model.final_block.conv
    grad_cam = GradCam(model, target_layer)

    # 학습 루프
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        print(f"Epoch {epoch}/{args.epochs} - "
              f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

        # 매 epoch마다 Grad-CAM 1D heatmap 생성 및 저장
        sample_feat, sample_label = val_ds[0]
        input_tensor = sample_feat.unsqueeze(0).to(device)
        pred_class = model(input_tensor).argmax(dim=1).item()
        cam = grad_cam.generate_cam(input_tensor, target_class=pred_class)  # (N_basis,)

        fig_path = os.path.join(args.output_dir, f"gradcam_epoch{epoch:02d}.png")
        visualize_1d_heatmap(cam,
                             figsize=(10,1),
                             cmap='jet',
                             xlabel='Basis index',
                             title=f'Epoch {epoch} Grad-CAM')
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()

        # 모델 저장
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_epoch{epoch:02d}.pt"))

    print("Training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SPHARM-Net with Grad-CAM visualization')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to SPHARM dataset root')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Where to save models and heatmaps')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--step-size', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--lmax', type=int, default=15,
                        help='Maximum SH degree (SPHARM lmax)')
    parser.add_argument('--num-classes', type=int, default=2)
    args = parser.parse_args()

    run_training(args)
