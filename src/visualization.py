import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. 配置 ---
# 保存绘图结果的文件夹
OUTPUT_DIR = 'plots/'
# 这是一个索引列表，0 代表第1张图
IMAGES_TO_PLOT = list(range(10))  # 例如: [0, 1, 9] 只绘制第1, 2, 10张图
# 这是一个索引列表，0 代表第1个波段
BANDS_TO_PLOT = list(range(28)) # 例如: [0, 127, 255] 绘制第1, 128, 256个波段

# --- 2. 创建输出文件夹 ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"创建文件夹: {OUTPUT_DIR}")

for i in IMAGES_TO_PLOT:
    mat_file_path = f'recon_results/reconstructed_cube_{i}.mat'
    # --- 3. 加载数据 ---
    try:
        data = scio.loadmat(mat_file_path)
        truth = data['truth']  # 形状 (256, 256, 28)
        pred = data['pred']    # 形状 (256, 256, 28)
        print(f"成功加载数据，Truth 形状: {truth.shape}, Pred 形状: {pred.shape}")
    except FileNotFoundError:
        print(f"错误：未找到 .mat 文件: {mat_file_path}")
        exit()
    except KeyError:
        print(f"错误：.mat 文件中未找到 'truth' 或 'pred' 键。")
        exit()

    # 获取数据的实际维度
    (height, width, num_bands) = truth.shape
    output_dir_img = f'{OUTPUT_DIR}/image_{i+1}'
    if not os.path.exists(output_dir_img):
        os.makedirs(output_dir_img)
    print(f"创建文件夹: {output_dir_img}")

    for b in BANDS_TO_PLOT:
        # 检查波段索引是否越界
        if b >= num_bands:
            print(f"警告：波段索引 {b} 超出范围 (0-{num_bands-1})，已跳过。")
            continue
        
        print(f"  正在处理: 图像 {i+1}/{10}, 波段 {b+1}/{num_bands}...")

        # a. 提取当前图像和波段的数据
        truth_slice = truth[:, :, b]
        pred_slice = pred[:, :, b]
        
        # b. 计算 Difference (差值图)
        # 我们使用 (truth - pred)，这样蓝色表示预测偏高，红色表示预测偏低
        diff_slice = truth_slice - pred_slice

        # c. 创建 1x3 子图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 设置总标题
        fig.suptitle(f'Image {i+1} / Band {b+1}', fontsize=16)

        # --- 绘图 1: Truth (真实值) ---
        
        # 为了确保 truth 和 pred 的颜色条刻度一致
        # 我们找到两者共享的最小/最大值
        vmin = 0
        vmax = 1

        ax = axes[0]
        im1 = ax.imshow(truth_slice, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f'Truth \nMin: {truth_slice.min():.2f}, Max: {truth_slice.max():.2f}')
        ax.axis('off')
        fig.colorbar(im1, ax=ax, orientation='horizontal', fraction=0.05, pad=0.08)

        # --- 绘图 2: Prediction (预测值) ---
        ax = axes[1]
        im2 = ax.imshow(pred_slice, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f'Prediction \nMin: {pred_slice.min():.2f}, Max: {pred_slice.max():.2f}')
        ax.axis('off')
        fig.colorbar(im2, ax=ax, orientation='horizontal', fraction=0.05, pad=0.08)

        # --- 绘图 3: Difference (差值图) ---
        
        # 为了让差值图的 0 位于中间，我们使用 'coolwarm' 色彩映射
        # 并找到差值的最大绝对值
        diff_abs_max = np.max(np.abs(diff_slice))
        
        ax = axes[2]
        # vmin 和 vmax 对称设置，使 0 总是白色
        im3 = ax.imshow(diff_slice, cmap='coolwarm', vmin=-diff_abs_max, vmax=diff_abs_max)
        ax.set_title(f'Difference (Truth - Pred)\nMax Error: {diff_abs_max:.2f}')
        ax.axis('off')
        fig.colorbar(im3, ax=ax, orientation='horizontal', fraction=0.05, pad=0.08)

        # d. 保存图像
        # 使用 0 填充文件名，使其按顺序排列
        filename = f"image_{i+1:02d}_band_{b+1:03d}.png"
        save_path = os.path.join(output_dir_img, filename)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # 调整布局以适应总标题
        plt.savefig(save_path)
        
        # e. 关闭图像，释放内存 (非常重要！)
        plt.close(fig)

print("\n--- 全部完成 ---")
print(f"所有图像已保存到: {OUTPUT_DIR}")