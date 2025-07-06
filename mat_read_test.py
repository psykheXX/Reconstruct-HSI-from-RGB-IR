import numpy as np
from spectral import open_image
import hdf5storage
import h5py
import matplotlib.pyplot as plt

def save_matv73(mat_name, var_name, var):
    hdf5storage.savemat(mat_name, {var_name: var}, format='7.3', store_python_metadata=True)

def resample_spectrum(data, original_wavelengths, target_resolution=10):
    """
    下采样光谱图像到指定分辨率（nm），并舍去最后一个波段。

    参数：
    data: ndarray, 形状为 (height, width, bands)
    original_wavelengths: ndarray, 原始波长数组
    target_resolution: float, 目标分辨率（nm）

    返回：
    resampled_data: 下采样后的光谱数据，形状为 (height, width, n_new_bands-1)
    new_wavelengths: 下采样后的波长数组（舍去最后一个波段）
    """
    # 存储原始形状
    height, width, bands = data.shape

    # 将数据重塑为 2D 数组 (height*width, bands)
    data_2d = data.reshape(-1, bands)

    start_wl = original_wavelengths[0]
    end_wl = original_wavelengths[-1]

    new_wavelengths = np.arange(start_wl, end_wl + target_resolution, target_resolution)

    new_wavelengths = new_wavelengths[:-1]

    n_pixels = data_2d.shape[0]
    n_new_bands = len(new_wavelengths)
    resampled_data_2d = np.zeros((n_pixels, n_new_bands), dtype=np.float32)

    # 对每个像素的光谱进行插值
    for i in range(n_pixels):
        resampled_data_2d[i, :] = np.interp(
            new_wavelengths,
            original_wavelengths,
            data_2d[i, :],
            left=np.nan,
            right=np.nan
        )

    resampled_data = resampled_data_2d.reshape(height, width, n_new_bands)

    return resampled_data, new_wavelengths

def preprocess():
    # 数据加载和处理
    start_wl = 498.8
    end_wl = 902.12
    n_bands = 136
    # 生成原始波长（保持136个波段）
    original_wavelengths = np.linspace(start_wl, end_wl, n_bands)

    signal_dat_path = "E:/2024年乌兰察布马铃薯试验/6.25数据/0高光谱数据/1345/results/REFLECTANCE_1345.dat"
    signal_hdr_path = signal_dat_path.replace(".dat", ".hdr")

    # 加载高光谱图像
    hyperspectral_image = open_image(signal_hdr_path)
    image_data = hyperspectral_image.load()

    # 转换为 NumPy 数组并选择波段
    np_tensor = np.float32(image_data)
    data = np_tensor[:, :, 35:171]  # 选择第35到170波段（共136波段）

    # 调用下采样函数
    resampled_data, new_wavelengths = resample_spectrum(data, original_wavelengths, target_resolution=10)

    # 打印结果
    print(f"New wavelengths: {new_wavelengths}")
    print(f"Number of new bands: {len(new_wavelengths)}")
    print(f"Resampled data shape: {resampled_data.shape}")
    var_name = 'cube'
    mat_dir = 'REFLECTANCE_1345'
    save_matv73(mat_dir, var_name, resampled_data)

def read_mat(mat_dir='REFLECTANCE_1345.mat'):
    hyper_path = mat_dir
    with h5py.File(hyper_path, 'r') as mat:
        hyper = np.float32(np.array(mat['cube']))

    print(hyper.shape)
    return hyper

if __name__ == '__main__':
    # preprocess()
    hyper = read_mat(mat_dir="D:\potato_hyper_dataset\Train_Spec\REFLECTANCE_1372.mat")
    channel_5 = hyper[8, :, :]  # Shape: (512, 512)


    # Plot the 100th channel as a grayscale image
    plt.figure(figsize=(6, 6))
    plt.imshow(channel_5, cmap='gray')
    plt.title('Channel 5 (Grayscale)')
    plt.colorbar(label='Reflectance')
    plt.axis('off')  # Optional: hide axes for cleaner visualization
    plt.show()