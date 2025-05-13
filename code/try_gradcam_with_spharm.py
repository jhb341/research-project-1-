"""
Integrated training script for SPHARM-Net with 1D Grad-CAM visualization.
Based on your original train.py, this adds GradCam hooks and per-epoch heatmap saving.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from spharmnet.lib.utils import SphericalDataset, Logger, eval_accuracy, eval_dice
from spharmnet.lib.loss import DiceLoss
from spharmnet.lib.io import read_mesh
from spharmnet import SPHARMNet

from gradcam_spharm_rev import GradCam, visualize_1d_heatmap
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    # (keep all your previous parser.add_argument calls here...)
    # Dataset & dataloader
    parser.add_argument("--sphere", type=str, default="./sphere/ico6.vtk", help="Sphere mesh path")
    """
    
        중략
    
    """
    parser.add_argument("--classes", type=int, nargs='+', required=True)
    args = parser.parse_args()
    return args


def step(model, loader, device, criterion, epoch, logger, nclass, optimizer=None, pbar=False):
    if optimizer is not None:
        model.train()
    else:
        model.eval()
    iterator = tqdm(loader) if pbar else loader
    running_loss = 0.0
    total_dice = []
    correct = 0
    for i, (features, labels) in enumerate(iterator):
        features, labels = features.to(device), labels.to(device)
        if optimizer:
            optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        if optimizer:
            loss.backward()
            optimizer.step()

        # logging
        running_loss += loss
        acc = eval_accuracy(outputs, labels)
        dice = eval_dice(outputs, labels)
        correct += acc * features.size(0)
        total_dice.append(dice)
    avg_loss = running_loss.item() / len(loader)
    accuracy = correct / len(loader.dataset)
    # write to logger
    logger.write([epoch+1, avg_loss, accuracy, np.mean(total_dice)])
    return accuracy


def main(args):
    # CUDA 설정
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if args.cuda else "cpu")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    # Mesh 로드
    v, _ = read_mesh(args.sphere)

    # 데이터셋 준비
    partitions = ['train','val','test']
    dataset = {}
    loader = {}
    logger = {}
    for part in partitions:
        dataset[part] = SphericalDataset(
            data_dir=args.data_dir,
            partition=part,
            fold=args.fold,
            num_vert=v.shape[0],
            in_ch=args.in_ch,
            depth=args.depth,
            channel=args.channel,
            bandwidth=args.bandwidth,
            interval=args.interval,
            threads=args.threads,
            verbose=(part=='train'),
        )
        loader[part] = torch.utils.data.DataLoader(
            dataset[part], batch_size=args.batch_size,
            shuffle=(part=='train'), num_workers=4
        )
        logger[part] = Logger(os.path.join(args.log_dir, part + '.tsv'))

    # 모델 초기화
    model = SPHARMNet(
        depth=args.depth,
        channel=args.channel,
        bandwidth=args.bandwidth,
        in_ch=len(args.in_ch),
        num_classes=len(args.classes),
        interval=args.interval,
    ).to(device)
    print("Num of params", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Optimizer, 스케줄러
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', factor=0.1, patience=1, verbose=True
    )
    # Loss
    criterion = nn.CrossEntropyLoss() if args.loss=='ce' else DiceLoss()

    # Resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1

    # ---- Grad-CAM 설정 ----
    # TODO: 실제 최종 SHConvBlock 모듈로 변경하세요
    target_layer = model.final_block
    grad_cam = GradCam(model, target_layer)

    best_acc = 0.0
    for epoch in range(start_epoch, args.epochs):
        # train & val
        step(model, loader['train'], device, criterion, epoch, logger['train'], len(args.classes), optimizer, pbar=True)
        val_acc = step(model, loader['val'], device, criterion, epoch, logger['val'], len(args.classes))
        if not args.no_decay:
            scheduler.step(val_acc)

        # Grad-CAM 1D heatmap 생성 및 저장
        features, labels = next(iter(loader['val']))
        input_tensor = features.to(device)
        pred_class = model(input_tensor).argmax(dim=1)[0].item()
        cam = grad_cam.generate_cam(input_tensor[0:1], target_class=pred_class)  # (N_basis,)
        heatmap_path = os.path.join(args.log_dir, f"gradcam_epoch{epoch:03d}.png")
        # 시각화
        visualize_1d_heatmap(
            cam,
            figsize=(8,1),
            cmap='jet',
            xlabel='Basis index',
            title=f'Epoch {epoch} Grad-CAM'
        )
        fig = plt.gcf()
        fig.savefig(heatmap_path, bbox_inches='tight')
        plt.close(fig)

        # 체크포인트 저장
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'acc': best_acc,
            }, os.path.join(args.ckpt_dir, f"best_model_fold{args.fold}.pth"))

    # 테스트 단계
    ckpt = torch.load(os.path.join(args.ckpt_dir, f"best_model_fold{args.fold}.pth"))
    model.load_state_dict(ckpt['model_state_dict'])
    step(model, loader['test'], device, criterion, ckpt['epoch'], logger['test'], len(args.classes))


if __name__ == "__main__":
    args = get_args()
    main(args)
