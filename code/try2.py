# ===========================================
# 1. 필요한 패키지와 모듈 불러오기
# ===========================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import os
import numpy as np
import time
import copy
import matplotlib.pyplot as plt
import json
import cv2

# 모델 관련 모듈
import resmodel

# ===========================================
# 2. 데이터셋 로드 및 전처리
# ===========================================
path2data = '/home/jhb341/Desktop/resnet/'

if not os.path.exists(path2data):
    os.mkdir(path2data)

train_ds = datasets.STL10(path2data, split='train', download=True, transform=transforms.ToTensor())
val_ds = datasets.STL10(path2data, split='test', download=True, transform=transforms.ToTensor())

print(len(train_ds))
print(len(val_ds))

# 정규화를 위한 평균과 표준편차 계산
train_meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in train_ds]
train_stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in train_ds]

train_meanR = np.mean([m[0] for m in train_meanRGB])
train_meanG = np.mean([m[1] for m in train_meanRGB])
train_meanB = np.mean([m[2] for m in train_meanRGB])

train_stdR = np.mean([s[0] for s in train_stdRGB])
train_stdG = np.mean([s[1] for s in train_stdRGB])
train_stdB = np.mean([s[2] for s in train_stdRGB])

val_meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in val_ds]
val_stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in val_ds]

val_meanR = np.mean([m[0] for m in val_meanRGB])
val_meanG = np.mean([m[1] for m in val_meanRGB])
val_meanB = np.mean([m[2] for m in val_meanRGB])



train_transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.Normalize([train_meanR, train_meanG, train_meanB],
                         [train_stdR, train_stdG, train_stdB]),
    transforms.RandomHorizontalFlip(),
])

val_transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.Normalize([train_meanR, train_meanG, train_meanB],
                         [train_stdR, train_stdG, train_stdB]),
])
train_ds.transform = train_transformation
val_ds.transform = val_transformation

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)

# ===========================================
# 3. 샘플 이미지 표시
# ===========================================
def show(img, y=None, color=True):
    npimg = img.numpy()
    npimg_tr = np.transpose(npimg, (1,2,0))
    plt.imshow(npimg_tr)
    if y is not None:
        plt.title('labels :' + str(y))
        
np.random.seed(1)
torch.manual_seed(1)
grid_size = 4
rnd_inds = np.random.randint(0, len(train_ds), grid_size)
print('image indices:', rnd_inds)
x_grid = [train_ds[i][0] for i in rnd_inds]
y_grid = [train_ds[i][1] for i in rnd_inds]
x_grid = utils.make_grid(x_grid, nrow=grid_size, padding=2)
show(x_grid, y_grid)
plt.savefig("sample_images.png", bbox_inches='tight')
plt.close()

# ===========================================
# 4. 모델, 학습, 평가
# ===========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resmodel.resnet101().to(device)
x = torch.randn(3, 3, 224, 224).to(device)
output = model(x)
print(output.size())
summary(model, (3, 224, 224), device=device.type)

loss_func = nn.CrossEntropyLoss(reduction='sum')
opt = optim.Adam(model.parameters(), lr=0.001)
from torch.optim.lr_scheduler import ReduceLROnPlateau
lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def metric_batch(output, target):
    pred = output.argmax(1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects

def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    metric_b = metric_batch(output, target)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metric_b

def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)
    for xb, yb in dataset_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)
        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)
        running_loss += loss_b
        if metric_b is not None:
            running_metric += metric_b
        if sanity_check is True:
            break
    loss = running_loss / len_data
    metric = running_metric / len_data
    return loss, metric

def train_val(model, params):
    num_epochs = params['num_epochs']
    loss_func = params["loss_func"]
    opt = params["optimizer"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]
    sanity_check = params["sanity_check"]
    lr_scheduler = params["lr_scheduler"]
    path2weights = params["path2weights"]
    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}
    best_loss = float('inf')
    start_time = time.time()
    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs-1, current_lr))
        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, sanity_check, opt)
        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)
        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl, sanity_check)
        loss_history['val'].append(val_loss)
        metric_history['val'].append(val_metric)
        if val_loss < best_loss:
            best_loss = val_loss
            print('Get best val_loss')
        lr_scheduler.step(val_loss)
        print('train loss: %.6f, val loss: %.6f, accuracy: %.2f, time: %.4f min' %
              (train_loss, val_loss, 100*val_metric, (time.time()-start_time)/60))
        print('-'*10)
    return model, loss_history, metric_history

params_train = {
    'num_epochs': 20,           # epoch 수                    
    'optimizer': opt,
    'loss_func': loss_func,
    'train_dl': train_dl,
    'val_dl': val_dl,
    'sanity_check': False,
    'lr_scheduler': lr_scheduler,
    'path2weights': './models/weights.pt',
}

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error creating directory:', directory)
createFolder('./models')

model, loss_hist, metric_hist = train_val(model, params_train)

with open("train_log.json", "w") as f:
    json.dump({'loss_hist': loss_hist, 'metric_hist': metric_hist}, f, indent=2)

print(" >> Train log 저장 완료! << ")

# 평가 (테스트 데이터셋에 대한 정확도 출력)
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for xb, yb in val_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)
        predicted = output.argmax(dim=1)
        correct += (predicted == yb).sum().item()
        total += yb.size(0)
accuracy = 100 * correct / total
print(f"전체 테스트 이미지: {total}장")
print(f"올바른 추론 개수: {correct}장")
print(f"정확도: {accuracy:.2f}%")

# ===========================================
# 5. Grad-CAM 적용
# ===========================================

class GradCam(nn.Module):
    def __init__(self, model, target_layer):
        super().__init__()
        self.model = model
        self.target_layer = target_layer  # 예: model.layer4[2].conv3
        self.forward_feature = None
        self.backward_feature = None
        self._register_hooks()
    
    def _register_hooks(self):
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_backward_hook(self.backward_hook)
    
    def forward_hook(self, module, input, output):
        self.forward_feature = output  # [batch, channels, h, w]
    
    def backward_hook(self, module, grad_input, grad_output):
        self.backward_feature = grad_output[0]  # [batch, channels, h, w]
    
    def generate_cam(self, input_tensor, target_class=None):
        # 순전파 실행
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()  # 배치가 1일 경우
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # 수집된 gradient와 feature map
        gradients = self.backward_feature  # shape: [batch, channels, h, w]
        activations = self.forward_feature  # shape: [batch, channels, h, w]
        
        # 채널별로 spatial 평균 -> 가중치 역할
        pooled_gradients = torch.mean(gradients, dim=[2, 3], keepdim=True)  # [batch, channels, 1, 1]
        
        # feature map에 가중치 곱하기
        weighted_activations = activations * pooled_gradients
        cam = torch.sum(weighted_activations, dim=1).squeeze(0)  # [h, w]
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()
        return cam.cpu().detach().numpy()

# Grad-CAM 적용 예시
# 5.1. 샘플 이미지 준비 (이미 val_transformation 적용되어 있음)
sample_img, sample_label = val_ds[0]  # (C, H, W)
input_tensor = sample_img.unsqueeze(0).to(device)

# 5.2. GradCam 객체 생성 (ResNet101의 마지막 합성곱 레이어에 hook)
grad_cam = GradCam(model, model.conv5_x[-1].residual_function[6])

#grad_cam = GradCam(model, model.conv3_x[2].residual_function[3])


# 5.3. Grad-CAM 히트맵 생성
cam = grad_cam.generate_cam(input_tensor, target_class=None)  # 타겟 클래스를 None으로 주면 argmax 선택
# cam의 크기는 보통 (H, W); 보통 7x7 혹은 14x14일 것임

# 5.4. 히트맵을 224x224로 리사이즈 (OpenCV 사용)
cam_resized = cv2.resize(cam, (224, 224))

# 5.5. 원본 이미지를 시각화를 위해 numpy로 변환 및 채널 순서 변경 (C,H,W -> H,W,C)
img_np = sample_img.cpu().numpy().transpose(1, 2, 0)
# 정규화 해제: 원래 학습 시 적용한 mean, std 사용 (주의: 여기서는 train에서 계산한 값 사용)
mean = np.array([train_meanR, train_meanG, train_meanB])
std = np.array([train_stdR, train_stdG, train_stdB])
img_np = img_np * std + mean
img_np = np.clip(img_np, 0, 1)

# 5.6. 히트맵에 컬러맵 적용 및 원본 이미지와 오버레이
heatmap = np.uint8(255 * cam_resized)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
heatmap = np.float32(heatmap) / 255
overlay = heatmap + np.float32(img_np)
overlay = overlay / np.max(overlay)

# 5.7. 결과 시각화 및 저장
import random
import numpy as np


#=========================================================
#
#=========================================================

# 9개의 인덱스를 무작위로 선택합니다.
indices = random.sample(range(len(val_ds)), 9)

# 3x3 그리드 생성
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

# 각 이미지에 대해 Grad-CAM 히트맵과 정답/예측 정보를 표시합니다.
for idx, ax in zip(indices, axes.flatten()):
    # 이미지와 정답 레이블 가져오기 (이미지: Tensor, 정규화 되어 있음)
    img_tensor, true_label = val_ds[idx]
    
    # 배치 차원 추가 및 모델에 입력
    input_tensor = img_tensor.unsqueeze(0).to(device)
    output = model(input_tensor)
    pred_label = output.argmax(dim=1).item()
    
    # Grad-CAM 히트맵 생성 (타겟 클래스를 예측한 클래스(pred_label)로 지정)
    cam = grad_cam.generate_cam(input_tensor, target_class=pred_label)
    
    # 히트맵을 224x224 크기로 리사이즈 (입력 이미지 크기에 맞춤)
    cam_resized = cv2.resize(cam, (224, 224))
    
    # 원본 이미지의 정규화를 해제 (전처리 시 사용한 mean, std 적용)
    img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([train_meanR, train_meanG, train_meanB])
    std = np.array([train_stdR, train_stdG, train_stdB])
    img_denorm = img_np * std + mean
    img_denorm = np.clip(img_denorm, 0, 1)
    
    # 히트맵에 컬러맵 적용 (cv2는 기본적으로 BGR을 리턴하므로 RGB로 변환)
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    
    # 원본 이미지와 히트맵을 합성하여 오버레이 생성
    overlay = heatmap + np.float32(img_denorm)
    overlay = overlay / np.max(overlay)
    
    # 해당 subplot에 오버레이 이미지를 표시하고, 제목에 정답과 예측 정보를 기입
    ax.imshow(overlay)
    ax.set_title(f"True: {true_label}, Pred: {pred_label}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("gradcam_9_images.png", bbox_inches="tight")
plt.show()



# 이미지에 컬러맵을 입히고 오버레이하는 함수
def apply_colormap_on_image(org_im, activation, colormap=cv2.COLORMAP_JET):
    # org_im: (H, W, 3) numpy array, 값은 0~1
    # activation: (H, W) numpy array, 값은 0~1
    heatmap = np.uint8(255 * activation)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(org_im)
    cam = cam / np.max(cam)
    return cam

# ================================
# 여기서부터는 기존 모델, 데이터셋, 전처리 코드가 있다고 가정합니다.
# 예를 들어, model, val_ds, device, train_meanR, train_meanG, train_meanB,
# train_stdR, train_stdG, train_stdB 등이 이미 정의되어 있다고 가정합니다.
# ================================

# 두 위치의 Grad-CAM 객체 생성
grad_cam_conv3 = GradCam(model, model.conv3_x[2].residual_function[3])
grad_cam_conv5 = GradCam(model, model.conv5_x[-1].residual_function[6])

# 9장의 이미지 선택 (예: val_ds의 처음 9장)
num_images = 9
images = []
labels = []
for i in range(num_images):
    img, label = val_ds[i]  # img: tensor, label: int
    images.append(img)
    labels.append(label)

# 준비한 이미지를 각각 처리
# 결과를 저장할 리스트
results_conv3 = []
results_conv5 = []
originals = []

for img_tensor in images:
    # 배치 차원 추가 및 device 이동
    input_tensor = img_tensor.unsqueeze(0).to(device)
    # Grad-CAM 계산
    cam_conv3 = grad_cam_conv3.generate_cam(input_tensor)
    cam_conv5 = grad_cam_conv5.generate_cam(input_tensor)
    # 이미지 크기 (224,224)로 리사이즈 (cam은 보통 작게 나오므로)
    cam_conv3_resized = cv2.resize(cam_conv3, (224, 224))
    cam_conv5_resized = cv2.resize(cam_conv5, (224, 224))
    
    # 원본 이미지 준비 (tensor -> numpy, 차원변경, 정규화 해제)
    org_img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    # 정규화 해제: (x * std + mean) -- 여기서는 train에서 계산한 값 사용
    mean = np.array([train_meanR, train_meanG, train_meanB])
    std = np.array([train_stdR, train_stdG, train_stdB])
    org_img = org_img * std + mean
    org_img = np.clip(org_img, 0, 1)
    
    originals.append(org_img)
    results_conv3.append(cam_conv3_resized)
    results_conv5.append(cam_conv5_resized)

# 3행 6열의 그리드: 각 행마다 3개의 이미지, 각 이미지마다 conv3와 conv5 결과를 좌우로 표시
fig, axes = plt.subplots(3, 6, figsize=(20, 10))
for idx in range(num_images):
    # 해당 이미지의 두 Grad-CAM 결과 오버레이 생성
    overlay_conv3 = apply_colormap_on_image(originals[idx], results_conv3[idx])
    overlay_conv5 = apply_colormap_on_image(originals[idx], results_conv5[idx])
    
    # 행과 열 계산 (각 이미지가 두 열 차지)
    row = idx // 3
    col_base = (idx % 3) * 2
    # conv3 결과
    axes[row, col_base].imshow(overlay_conv3)
    axes[row, col_base].set_title(f"conv3_x GradCAM\n(label: {labels[idx]})")
    axes[row, col_base].axis("off")
    # conv5 결과
    axes[row, col_base+1].imshow(overlay_conv5)
    axes[row, col_base+1].set_title("conv5_x GradCAM")
    axes[row, col_base+1].axis("off")

plt.tight_layout()
plt.savefig("gradcam_comparison.png", bbox_inches="tight")
plt.show()