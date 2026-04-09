import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
# 从同目录下的Models.py导入PhysicsInformedNN类
from Models import PhysicsInformedNN

def visualize_potential(X_Star, U_Star, u_pred, title='Potential Distribution', save_path='Results/potential.png'):
    """
    可视化真实电势和预测电势的对比图（仅支持极坐标系），分为两个子图显示
    
    参数说明：
        X_Star: 极坐标形式的空间坐标点，形状为[样本数, 2]，其中第一列为半径r，第二列为角度theta
        U_Star: 真实电势值，形状为[样本数, 1]
        u_pred: 预测电势值，形状为[样本数, 1]
        title: 图表标题，默认为'Potential Distribution'
        save_path: 保存图表的路径，默认为'Results/potential.png'
    """  
    # 计算电势的最大值和最小值，确保两个子图使用相同的颜色范围
    vmin = min(np.min(U_Star), np.min(u_pred))
    vmax = max(np.max(U_Star), np.max(u_pred))
    
    # X_Star是极坐标(r, sin(theta), cos(theta))
    r = X_Star[:, 0]
    sin_theta = X_Star[:, 1]
    cos_theta = X_Star[:, 2]
    theta = np.arctan2(sin_theta, cos_theta)
    
    # 调整极坐标的半径范围，确保图表居中
    r_max = np.max(r) * 1.1
    
    # 创建画布 - 1行2列的子图布局
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(projection='polar'))
    
    # 第一个子图：真实电势分布
    sc_real = ax1.scatter(theta, r, c=U_Star[:, 0], cmap='jet', s=10, vmin=vmin, vmax=vmax)
    ax1.set_title(r'$\varphi_{real}(r,\theta)/kV$', fontsize=20, pad=20)
    
    # 隐藏所有轴线和标注
    ax1.set_xticklabels([])  # 隐藏角度标注
    ax1.set_yticklabels([])  # 隐藏半径标注
    ax1.spines['polar'].set_visible(False)  # 隐藏极坐标轴线
    ax1.set_ylabel('')  # 隐藏半径方向的标签
    
    ax1.grid(False)
    
    # 设置极坐标图居中
    ax1.set_theta_offset(0)  # 让0度在右侧
    ax1.set_theta_direction(-1)  # 顺时针方向
    ax1.set_ylim(0, r_max)
    ax1.set_aspect('equal')
    
    # 使用set_position直接控制极坐标图在画布上的位置
    ax1.set_position([0.05, 0.25, 0.4, 0.5])  # 左, 下, 宽, 高
    
    # 为第一个子图添加颜色条
    plt.colorbar(sc_real, ax=ax1, fraction=0.046, pad=0.04)
    
    # 第二个子图：预测电势分布
    sc_pred = ax2.scatter(theta, r, c=u_pred[:, 0], cmap='jet', s=10, vmin=vmin, vmax=vmax)
    ax2.set_title(r'$\varphi_{pred}(r,\theta)/kV$', fontsize=20, pad=20)
    
    # 隐藏所有轴线和标注
    ax2.set_xticklabels([])  # 隐藏角度标注
    ax2.set_yticklabels([])  # 隐藏半径标注
    ax2.spines['polar'].set_visible(False)  # 隐藏极坐标轴线
    ax2.set_ylabel('')  # 隐藏半径方向的标签
    
    ax2.grid(False)
    
    # 设置极坐标图居中
    ax2.set_theta_offset(0)  # 让0度在右侧
    ax2.set_theta_direction(-1)  # 顺时针方向
    ax2.set_ylim(0, r_max)
    ax2.set_aspect('equal')
    
    # 使用set_position直接控制极坐标图在画布上的位置
    ax2.set_position([0.55, 0.25, 0.4, 0.5])  # 左, 下, 宽, 高
    
    # 为第二个子图添加颜色条
    plt.colorbar(sc_pred, ax=ax2, fraction=0.046, pad=0.04)
    
    # 在图表顶部添加主标题
    fig.suptitle(title, fontsize=24, y=0.98)
    
    plt.tight_layout()
    
    if save_path is not None:
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()
    
def visualize_error(X_Star, U_Star, u_pred, title='Error Distribution', save_path='Results/error.png'):
    """
    可视化真实电势和预测电势之间的误差分布（仅支持极坐标系）
    
    参数说明：
        X_Star: 极坐标形式的空间坐标点，形状为[样本数, 2]，其中第一列为半径r，第二列为角度theta
        U_Star: 真实电势值，形状为[样本数, 1]
        u_pred: 预测电势值，形状为[样本数, 1]
        title: 图表标题，默认为'Error Distribution'
        save_path: 保存图表的路径，默认为'Results/error.png'
    """
    # 创建画布 - 正方形比例适合极坐标
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # X_Star是极坐标(r, sin(theta), cos(theta))
    r = X_Star[:, 0]
    sin_theta = X_Star[:, 1]
    cos_theta = X_Star[:, 2]
    theta = np.arctan2(sin_theta, cos_theta)
    
    # 计算误差
    error = u_pred[:, 0] - U_Star[:, 0]
    
    # 计算误差的统计信息
    error_max = np.max(np.abs(error))
    
    # 绘制误差分布（极坐标），直接设置颜色范围
    sc = ax.scatter(theta, r, c=error, cmap='jet', vmin=-error_max, vmax=error_max)
    
    ax.set_title(r'$\Delta\varphi(kV)$', fontsize=20, pad=20)
    
    # 隐藏所有轴线和标注
    ax.set_xticklabels([])  # 隐藏角度标注
    ax.set_yticklabels([])  # 隐藏半径标注
    ax.spines['polar'].set_visible(False)  # 隐藏极坐标轴线
    ax.set_ylabel('')  # 隐藏半径方向的标签
    
    ax.grid(False)
    
    # 设置极坐标图居中
    ax.set_theta_offset(0)  # 让0度在右侧
    ax.set_theta_direction(-1)  # 顺时针方向
    
    # 调整极坐标的半径范围，确保图表居中
    r_max = np.max(r) * 1.1
    ax.set_ylim(0, r_max)
    
    # 设置极坐标图的宽高比为1:1，确保圆形居中
    ax.set_aspect('equal')
    
    # 使用set_position直接控制极坐标图在画布上的位置
    ax.set_position([0.5 - 0.4, 0.5 - 0.4, 0.8, 0.8])  # 左, 下, 宽, 高
    
    # 添加颜色条
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    
    # 保存图表
    if save_path is not None:
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.tight_layout()
    plt.show()

    
def visualize_pde(lambda_1_value, lambda_2_value, noise=0, title='Identified PDE', save_path='Results/pde.png'):
    # 创建一个简单的正方形画布
    fig = plt.figure(figsize=(10, 8))
    
    # 创建一个占满整个画布的坐标系
    ax = plt.subplot(111)
    ax.axis('off')  # 关闭坐标轴
    
    # 创建PDE方程表格文本
    s1 = r'$\begin{tabular}{ |c|c| }  \hline Correct PDE & $\frac{\partial^2\varphi}{\partial r^2} + \frac{1}{r}\frac{\partial\varphi}{\partial r} + \frac{1}{r^2}\frac{\partial^2\varphi}{\partial \theta^2} = 0$ \\  \hline Identified PDE (%d\%% noise) & '%(int(noise))
    s2 = r'$\frac{\partial^2\varphi}{\partial r^2} + %.5f \frac{1}{r}\frac{\partial\varphi}{\partial r} + %.5f \frac{1}{r^2}\frac{\partial^2\varphi}{\partial \theta^2} = 0$ \\  \hline ' % (lambda_1_value, lambda_2_value)
    s5 = r'\end{tabular}$'
    s = s1+s2+s5
    
    # 在坐标系中心位置显示文本
    ax.text(0.5, 0.5, s, size=28, ha='center', va='center', transform=ax.transAxes)
    
    # 在图表顶部添加标题
    plt.suptitle(title, fontsize=22, y=0.95)
    
    # 调整布局，确保内容居中
    plt.tight_layout()
    
    # 保存图表
    if save_path is not None:
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()


def load_model_and_predict(X_Star, U_Star, model_filename='Identification_model.pth'):
    """
    加载保存的模型并进行预测
    
    参数说明：
        X_Star: 坐标点数组，形状为[样本数, 2]
        model_filename: 模型文件名，默认为'Identification_model.pth'
    
    返回值：
        u_pred: 预测电势值，形状为[样本数, 1]
        f_pred: 预测残差值，形状为[样本数, 1]
        lambda_1_value: 模型参数lambda_1，形状为[1]
        lambda_2_value: 模型参数lambda_2，形状为[1]
    """
    # 加载模型参数
    # 获取项目根目录路径
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(root_dir, 'Model', model_filename)
    checkpoint = torch.load(model_path, weights_only=False)
    
    # 重建模型
    layers = checkpoint['layers']
    lb = checkpoint['lb']
    ub = checkpoint['ub']
    
    # 创建模型实例（使用假数据初始化）
    X_u_train = np.zeros((1, 3))
    u_train = np.zeros((1, 1))
    
    model = PhysicsInformedNN(X_u_train, u_train, layers, lb, ub)
    
    # 加载模型权重
    model.load_model(model_filename)
    
    # 进行预测
    u_pred, f_pred = model.predict(X_Star)
    # 反归一化预测结果
    u_pred = u_pred * (U_Star.max() - U_Star.min()) + U_Star.min()
    # 计算误差
    error_u = np.linalg.norm(U_Star - u_pred, 2) / np.linalg.norm(U_Star, 2)
    # 从模型中获取参数值
    lambda_1_value = model.lambda_1.detach().cpu().numpy()[0]  # 提取标量值
    lambda_2_value = model.lambda_2.detach().cpu().numpy()[0]  # 提取标量值

    error_lambda_1 = np.abs(1.0-lambda_1_value) * 100
    error_lambda_2 = np.abs(1.0-lambda_2_value) * 100

    print('Error u: %e' % (error_u))    
    print('Error l1: %.5f%%' % (error_lambda_1))                             
    print('Error l2: %.5f%%' % (error_lambda_2))  
    
    # 设置LaTeX字体
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    return u_pred, f_pred, lambda_1_value, lambda_2_value


if __name__ == '__main__':
    # 获取项目根目录路径
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = np.loadtxt(os.path.join(root_dir, 'Data\Data.txt'), delimiter=';')
    
    # 数据导入
    U_Star = data[:, 2:3]  # 真实电势值
    # 将笛卡尔坐标转换为极坐标
    x = data[:, 0]
    y = data[:, 1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    # 使用sin(theta)和cos(theta)代替直接使用theta，保持周期性
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    X_star = np.column_stack((r, sin_theta, cos_theta))  # 极坐标 (r, sin(theta), cos(theta))
    
    # 加载模型并进行预测（使用笛卡尔坐标）
    u_pred, f_pred, lambda_1_value, lambda_2_value = load_model_and_predict(X_star, U_Star)
    
    # 使用极坐标可视化电势分布对比
    visualize_potential(X_star, U_Star, u_pred, title='Potential Distribution')
    
    # 可视化误差分布
    visualize_error(X_star, U_Star, u_pred, title='Error Distribution')

    # 可视化PDE
    visualize_pde(lambda_1_value, lambda_2_value, noise=0, title='Identified PDE')
