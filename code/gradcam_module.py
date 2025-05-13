import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


class GradCam(nn.Module):
    """
    Grad-CAM implementation for CNNs.
    사용법:
        grad_cam = GradCam(model, target_layer)
        cam = grad_cam.generate_cam(input_tensor, target_class)
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        super().__init__()
        self.model = model
        self.target_layer = target_layer
        self.forward_feature = None
        self.backward_feature = None
        self._register_hooks()

    def _register_hooks(self):
        # forward hook: feature map 저장
        self.target_layer.register_forward_hook(self._forward_hook)
        # backward hook: gradient 저장
        self.target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.forward_feature = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.backward_feature = grad_output[0]

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        input_tensor: (1, C, H, W) 형태의 배치
        target_class: 출력에서 사용할 클래스 인덱스, None일 경우 argmax 사용
        return: (H_map, W_map) 크기의 정규화된 CAM map (0~1)
        """
        # 순전파
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 기울기 초기화 및 one-hot
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        # 역전파
        output.backward(gradient=one_hot, retain_graph=True)

        # 저장된 gradient와 feature map 가져오기
        grads = self.backward_feature      # [1, C, h, w]
        acts = self.forward_feature        # [1, C, h, w]

        # 채널별로 공간 차원 평균 -> weight
        pooled_grads = torch.mean(grads, dim=[2, 3], keepdim=True)

        # 가중치 적용
        weighted_acts = acts * pooled_grads
        cam = torch.sum(weighted_acts, dim=1).squeeze(0)  # [h, w]
        cam = F.relu(cam)
        cam -= cam.min()
        if cam.max() != 0:
            cam /= cam.max()

        return cam.cpu().detach().numpy()


def apply_colormap_on_image(org_im: np.ndarray,
                             activation: np.ndarray,
                             colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    원본 이미지 위에 CAM 히트맵을 오버레이합니다.
    org_im: (H, W, 3), 0~1 float
    activation: (H, W), 0~1 float
    return: (H, W, 3), 0~1 float overlay image
    """
    heatmap = np.uint8(255 * activation)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32) / 255

    overlay = heatmap + org_im
    overlay = overlay / np.max(overlay)
    return overlay


# ================ 사용 예시 ================
# model: 학습된 모델 (nn.Module)
# target_layer: 예) model.layer4[2].conv3
# sample_img: 정규화된 Tensor (C, H, W)
#
# import torchvision.transforms as T
#
# input_tensor = sample_img.unsqueeze(0).to(device)
# grad_cam = GradCam(model, target_layer)
# cam_map = grad_cam.generate_cam(input_tensor)
# cam_resized = cv2.resize(cam_map, (H, W))
# org_im = sample_img.cpu().numpy().transpose(1,2,0)
# org_im = org_im * std + mean  # 정규화 해제
# overlay = apply_colormap_on_image(org_im, cam_resized)
# plt.imshow(overlay)
# plt.show()
