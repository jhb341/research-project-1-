"""
    
    VGG16에서 gradcam 적용
    
"""

from code.resnet.gradcam_module import GradCam, apply_colormap_on_image

# 1) VGG16 불러오기
model = torchvision.models.vgg16(pretrained=True).eval().to(device)

# 2) 타겟 레이어 선택 (features[28]은 마지막 conv. layer)
target_layer = model.features[28]

# 3) GradCam 객체 생성
grad_cam = GradCam(model, target_layer)

# 4) 입력 준비 및 CAM 생성
input_tensor = preprocess(img).unsqueeze(0).to(device)
pred = model(input_tensor).argmax(dim=1).item()
cam = grad_cam.generate_cam(input_tensor, target_class=pred)

# 5) 시각화
cam_resized = cv2.resize(cam, (img_w, img_h))
overlay = apply_colormap_on_image(denormalize(img), cam_resized)
plt.imshow(overlay); plt.axis('off'); plt.show()
