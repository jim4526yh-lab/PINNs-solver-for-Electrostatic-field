import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# 从同目录下的Models.py导入PhysicsInformedNN类
from Models import PhysicsInformedNN

def visualize_potential(X_Star, U_Star, u_pred, title='Potential Distribution Comparison', save_path='Results/potential.png'):
    """
    可视化真实电势和预测电势的对比图
    
    参数说明：
        X_Star: 坐标点数组，形状为[样本数, 2]
        U_Star: 真实电势值，形状为[样本数, 1]
        u_pred: 预测电势值，形状为[样本数, 1]
        title: 图表标题，默认为'电势分布对比'
        save_path: 保存图表的路径，默认为None（不保存）
    """
    # 设置LaTeX字体
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    # 创建画布
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    
    # 绘制真实电势分布
    sc1 = ax[0].scatter(X_Star[:, 0], X_Star[:, 1], c=U_Star[:, 0], cmap='jet')
    ax[0].set_title(r'$\varphi_{real}(x,y)/kV$', fontsize=20)
    ax[0].set_xlabel(r'$x$', fontsize=18)
    ax[0].set_ylabel(r'$y$', fontsize=18, rotation=0)
    ax[0].tick_params(labelsize=12)
    plt.colorbar(sc1, ax=ax[0], fraction=0.046, pad=0.04)
    
    # 绘制预测电势分布
    sc2 = ax[1].scatter(X_Star[:, 0], X_Star[:, 1], c=u_pred[:, 0], cmap='jet')
    ax[1].set_title(r'$\varphi_{pred}(x,y)/kV$', fontsize=20)
    ax[1].set_xlabel(r'$x$', fontsize=18)
    ax[1].set_ylabel(r'$y$', fontsize=18, rotation=0)
    ax[1].tick_params(labelsize=12)
    plt.colorbar(sc2, ax=ax[1], fraction=0.046, pad=0.04)
    
    plt.suptitle(title, fontsize=22)
    plt.tight_layout()
    
    # 保存图表
    if save_path is not None:
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    # plt.show()

def visualize_residual(X_Star, f_pred, title='Physics Equation Residual', save_path='Results/residual.png'):
    """
    可视化物理方程残差
    
    参数说明：
        X_Star: 坐标点数组，形状为[样本数, 2]
        f_pred: 预测的物理方程残差，形状为[样本数, 1]
        title: 图表标题，默认为'物理方程残差'
        save_path: 保存图表的路径，默认为None（不保存）
    """
    # 设置LaTeX字体
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    # 创建画布
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # 绘制残差分布
    sc = ax.scatter(X_Star[:, 0], X_Star[:, 1], c=f_pred[:, 0], cmap='jet')
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(r'$x$', fontsize=18)
    ax.set_ylabel(r'$y$', fontsize=18, rotation=0)
    ax.tick_params(labelsize=12)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    plt.title(r'$f_{pred}(x,y)$', fontsize=20)
    
    plt.tight_layout()
    
    # 保存图表
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    # plt.show()

def visualize_error(X_Star, U_Star, u_pred, title='Prediction Error Distribution', save_path='Results/error.png'):
    """
    可视化预测误差分布
    
    参数说明：
        X_Star: 坐标点数组，形状为[样本数, 2]
        U_Star: 真实电势值，形状为[样本数, 1]
        u_pred: 预测电势值，形状为[样本数, 1]
        title: 图表标题，默认为'预测误差分布'
        save_path: 保存图表的路径，默认为None（不保存）
    """
    # 计算误差
    error = np.abs(U_Star - u_pred)
    
    # 设置LaTeX字体
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    # 创建画布
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # 绘制误差分布
    sc = ax.scatter(X_Star[:, 0], X_Star[:, 1], c=error[:, 0], cmap='jet')
    ax.set_title(r'$|\varphi_{real}(x,y)-\varphi_{pred}(x,y)|/kV$', fontsize=20)
    ax.set_xlabel(r'$x$', fontsize=18)
    ax.set_ylabel(r'$y$', fontsize=18, rotation=0)
    ax.tick_params(labelsize=12)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # 保存图表
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    # plt.show()

def load_model_and_predict(X_Star, model_filename='Inference_model.pth'):
    """
    加载保存的模型并进行预测
    
    参数说明：
        X_Star: 坐标点数组，形状为[样本数, 2]
        model_filename: 模型文件名，默认为'Inference_model.pth'
    
    返回值：
        u_pred: 预测电势值，形状为[样本数, 1]
        f_pred: 预测残差值，形状为[样本数, 1]
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
    X_u_train = np.zeros((1, 2))
    u_train = np.zeros((1, 1))
    X_f_train = np.zeros((1, 2))
    
    model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub)
    
    # 加载模型权重
    model.load_model(model_filename)
    
    # 进行预测
    u_pred, f_pred = model.predict(X_Star)
    
    return u_pred, f_pred


if __name__ == '__main__':
    # 获取项目根目录路径
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = np.loadtxt(os.path.join(root_dir, 'Data\Data.txt'), delimiter=';')
    X_Star = data[:, 0:2]  # 坐标值
    U_Star = data[:, 2:3]  # 真实电势值
    
    # 加载模型并进行预测
    u_pred, f_pred = load_model_and_predict(X_Star)
    
    # 反归一化预测结果（如果需要）
    u_pred = u_pred * (U_Star.max() - U_Star.min()) + U_Star.min()
    
    # 计算误差
    error_u = np.linalg.norm(U_Star - u_pred, 2) / np.linalg.norm(U_Star, 2)
    print(f'预测误差: {error_u:.5e}')
    
    # 可视化结果
    visualize_potential(X_Star, U_Star, u_pred, title='Potential Distribution Comparison')
    visualize_residual(X_Star, f_pred, title='Physics Equation Residual')
    visualize_error(X_Star, U_Star, u_pred, title='Prediction Error Distribution')
