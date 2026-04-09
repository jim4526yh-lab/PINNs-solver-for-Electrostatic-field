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
from Utilities.Visualization import visualize_potential, visualize_residual, visualize_error
# 从Utilities.Models导入模型类
from Utilities.Models import PhysicsInformedNN

# 设置随机种子
np.random.seed(1234)


if __name__ == '__main__':

    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

    data = np.loadtxt('Data\Data.txt',delimiter=';')#导入训练数据
    X_Star = data[:,0:2] #验证数据点的坐标值
    U_Star = data[:,2:3] #验证数据点的Comsol仿真值，用于效果对比
    
    #获取坐标张量边界值
    lb = X_Star.min(0)#数据坐标下界
    ub = X_Star.max(0)#数据坐标上界

    #寻找边界点索引，需要根据实际问题自定义
    surface_idx=np.where((U_Star<1e-10) | (800-U_Star<1e-10))[0]#边界点索引
    np.random.shuffle(surface_idx)#打乱边界点索引
    
    N_u = surface_idx.shape[0]
    N_f = 10000

    X_u_train=X_Star[surface_idx,:]#边界点坐标
    u_train=U_Star[surface_idx]#边界点电势
    #对边界点电势进行归一化
    u_train=(u_train-U_Star.min())/(U_Star.max()-U_Star.min())

    # 生成随机内部坐标点用于训练
    # 几何区域:内环半径0.2，外环半径1.0
    r = 0.2 + 0.8 * np.random.rand(N_f, 1)
    # 角度范围0到2π
    theta = 2 * np.pi * np.random.rand(N_f, 1)
    # 转换为笛卡尔坐标
    X = r * np.cos(theta)
    Y = r * np.sin(theta)
    # 将x和y组合成坐标矩阵
    X_f_train = np.hstack((X, Y))
    X_f_train = np.vstack((X_f_train, X_u_train))
    
    # 初始化模型
    model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub)
    # 训练模型
    model.train()
    # 保存训练好的模型
    model.save_model()
    # 预测验证数据点的电势值和物理方程残差
    u_pred, f_pred = model.predict(X_Star)
    #对预测电势进行反归一化
    u_pred=u_pred*(U_Star.max()-U_Star.min())+U_Star.min()

    error_u = np.linalg.norm(U_Star-u_pred,2)/np.linalg.norm(U_Star,2)
    print('Error u: %e' % (error_u))                     

    # 使用可视化模块绘制结果
    visualize_potential(X_Star, U_Star, u_pred, title='Potential Distribution Comparison')
    visualize_residual(X_Star, f_pred, title='Physics Equation Residual')
    visualize_error(X_Star, U_Star, u_pred, title='Prediction Error Distribution')