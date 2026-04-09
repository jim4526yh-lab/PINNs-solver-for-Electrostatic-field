import sys
import os
import torch
import numpy as np
from pyDOE import lhs

# 添加Utilities目录到Python路径
utilities_path = os.path.join(os.path.dirname(__file__), 'Utilities')
if utilities_path not in sys.path:
    sys.path.append(utilities_path)

# 导入可视化模块
from Utilities.Visualization import load_model_and_predict, visualize_potential, visualize_pde, visualize_error
# 从Utilities.Models导入模型类
from Utilities.Models import PhysicsInformedNN

# 设置随机种子
np.random.seed(1234)


if __name__ == '__main__':

    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 1]

    data = np.loadtxt('Data\Data.txt',delimiter=';')#导入训练数据
    # 数据导入
    U_Star = data[:, 2:3]  # 真实电势值
    # 将笛卡尔坐标转换为极坐标
    x = data[:, 0]
    y = data[:, 1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # 使用sin(theta)和cos(theta)代替直接使用theta，保持周期性
    sin_theta = y / r
    cos_theta = x / r
    X_Star = np.column_stack((r, sin_theta, cos_theta))  # 极坐标 (r, sin(theta), cos(theta))
    
    #获取坐标张量边界值
    lb = X_Star.min(0)#坐标数据下界
    ub = X_Star.max(0)#坐标数据上界

    #打乱训练数据
    idx = np.random.permutation(X_Star.shape[0])
    X_u_train = X_Star[idx,:]
    u_train = U_Star[idx,:]
    # 添加噪声
    noise = 0.00
    u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
    # 训练数据归一化
    u_train = (u_train - U_Star.min())/(U_Star.max()-U_Star.min())
    # 初始化模型
    model = PhysicsInformedNN(X_u_train, u_train, layers, lb, ub)
    # 训练模型
    model.train(300)#使用Adam优化器预训练
    # 保存训练好的模型
    model.save_model()
    # 加载模型并进行预测
    u_pred, f_pred, lambda_1_value, lambda_2_value = load_model_and_predict(X_Star, U_Star)
    # 使用可视化模块绘制结果
    visualize_potential(X_Star, U_Star, u_pred, title='Potential Distribution')
    # 可视化误差分布
    visualize_error(X_Star, U_Star, u_pred, title='Error Distribution')
    # 可视化PDE
    visualize_pde(lambda_1_value, lambda_2_value, noise=noise, title='Identified PDE')
