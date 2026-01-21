#!/user/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from mayavi import mlab
from pydicom import dcmread
from glob import glob
from scipy.ndimage import zoom


def read_medical_image(input_path: str) -> tuple[np.ndarray, np.ndarray]:
    # 读取NII文件
    if input_path.endswith(('.nii', '.nii.gz')):
        img = sitk.ReadImage(input_path)
        spacing = np.array(img.GetSpacing())[::-1]  # 转为（z,y,x）顺序
        data = sitk.GetArrayFromImage(img)  # 输出格式：(z,y,x)

    # 读取DICOM文件夹
    elif os.path.isdir(input_path):
        dicom_files = sorted(glob(os.path.join(input_path, '*.dcm')),
                             key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
        data_list = [dcmread(f).pixel_array for f in dicom_files]
        data = np.stack(data_list, axis=0)  # 格式：(z,y,x)
        spacing = np.array([dcmread(dicom_files[0]).SliceThickness,
                            dcmread(dicom_files[0]).PixelSpacing[1],
                            dcmread(dicom_files[0]).PixelSpacing[0]])

    else:
        raise ValueError("仅支持NII文件或DICOM文件夹输入")

    return data, spacing


def preprocess_tumor(data: np.ndarray, spacing: np.ndarray,
                     lower_threshold: int = 50, upper_threshold: int = 200) -> np.ndarray:

    # 1. 阈值分割（区分肿瘤与正常组织）
    tumor_mask = np.logical_and(data >= lower_threshold, data <= upper_threshold).astype(np.uint8)

    # 2. 各向同性采样（统一xyz轴间距，优化重建效果）
    target_spacing = np.array([1.0, 1.0, 1.0])  # 目标间距：1mm×1mm×1mm
    zoom_factor = spacing / target_spacing
    tumor_mask = zoom(tumor_mask, zoom_factor, order=0)  # 最近邻插值（保留二值特性）

    return tumor_mask


def visualize_3d_reconstruction(tumor_mask: np.ndarray) -> None:
    # 2D切片预览（辅助验证分割效果）
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    z_mid = tumor_mask.shape[0] // 2
    y_mid = tumor_mask.shape[1] // 2
    x_mid = tumor_mask.shape[2] // 2
    axes[0].imshow(tumor_mask[z_mid, :, :], cmap='gray')
    axes[0].set_title(f'Z轴中间切片（第{z_mid}层）')
    axes[1].imshow(tumor_mask[:, y_mid, :], cmap='gray')
    axes[1].set_title(f'Y轴中间切片（第{y_mid}层）')
    axes[2].imshow(tumor_mask[:, :, x_mid], cmap='gray')
    axes[2].set_title(f'X轴中间切片（第{x_mid}层）')
    plt.tight_layout()
    plt.show()

    # 3D重建可视化（表面绘制）
    mlab.figure(size=(1000, 800))
    # 提取肿瘤表面（vmin=0.5过滤背景）
    mlab.contour3d(tumor_mask, vmin=0.5, colormap='viridis', opacity=0.8)
    mlab.title('肿瘤三维重建结果', size=20)
    mlab.xlabel('X轴（mm）'), mlab.ylabel('Y轴（mm）'), mlab.zlabel('Z轴（mm）')
    mlab.show()


if __name__ == "__main__":
    # 根据实际路径修改！
    INPUT_PATH = "G:/mry1/TOM500/data preprocess/tumor.nii"  # NII文件或DICOM文件夹路径
    TUMOR_LOWER_THRESHOLD = 60  # 肿瘤下界阈值
    TUMOR_UPPER_THRESHOLD = 180  # 肿瘤上界阈值

    # 读取医学图像
    img_data, img_spacing = read_medical_image(INPUT_PATH)
    print(f"图像尺寸（z,y,x）：{img_data.shape}，像素间距：{np.round(img_spacing, 2)}mm")

    #预处理分割肿瘤
    tumor_mask = preprocess_tumor(img_data, img_spacing,
                                  TUMOR_LOWER_THRESHOLD, TUMOR_UPPER_THRESHOLD)
    print(f"预处理后肿瘤掩码尺寸：{tumor_mask.shape}")

    # 三维重建与可视化
    visualize_3d_reconstruction(tumor_mask)