import os
import warnings
import torch
import numpy as np
from collections import OrderedDict

# 过滤掉CUDA cuBLAS上下文警告
warnings.filterwarnings("ignore", message="Attempting to run cuBLAS, but there was no current CUDA context!")

# CUDA support 
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# 深度神经网络类，用于构建物理引导神经网络(PINN)的基础网络结构
class DNN(torch.nn.Module):
    def __init__(self, layers):
        """
        初始化深度神经网络
        
        参数:
        ----------
        layers : list
            定义神经网络各层神经元数量的列表，例如[3, 20, 20, 20, 20, 20, 20, 20, 20, 1]表示
            输入层3个神经元（r, sin(theta), cos(theta)），五个隐藏层各20个神经元，输出层1个神经元
        """
        # 调用PyTorch的Module父类构造函数，确保神经网络模块正确初始化
        super(DNN, self).__init__()
        
        # 参数设置：计算网络深度（层数-1）
        # 例如，layers=[2, 20, 20, 1]时，depth=3（输入层不计入深度）
        self.depth = len(layers) - 1
        
        # 设置激活函数：选择双曲正切函数(Tanh)作为非线性激活函数
        # Tanh函数将输出值映射到[-1, 1]区间，适合物理场预测问题
        self.activation = torch.nn.Tanh
        
        # 创建层列表：用于存储网络中的所有层及其名称
        layer_list = list()
        
        # 循环构建隐藏层结构
        # 对于除最后一层外的所有层，每层后都添加一个激活函数
        for i in range(self.depth - 1): 
            # 添加全连接线性层，连接第i层和第i+1层
            # 使用元组格式(名称, 层对象)存储，方便后续构建OrderedDict
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            # 在每个线性层后添加激活函数层，引入非线性变换能力
            layer_list.append(('activation_%d' % i, self.activation()))
        
        # 添加输出层：最后一个线性层，没有激活函数
        # 连接倒数第二层和输出层
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        
        # 将层列表转换为有序字典(OrderedDict)
        # 使用OrderedDict确保层的执行顺序与添加顺序一致，并允许通过名称引用层
        layerDict = OrderedDict(layer_list)
        
        # 部署神经网络：使用PyTorch的Sequential容器组装网络
        # Sequential会按顺序执行其中的所有层，实现前向传播功能
        self.layers = torch.nn.Sequential(layerDict)
        
    def forward(self, x):
        out = self.layers(x)
        return out

# the physics-guided neural network
class PhysicsInformedNN():
    def __init__(self, X, u, layers, lb, ub): 
        # 初始化物理信息神经网络模型
        # 参数说明：
        #   X_u: 边界条件和初始条件的空间-角度坐标点，形状[样本数, 3]
        #   u: 边界条件和初始条件的真实值，形状[样本数, 1]
        #   layers: 神经网络层结构配置，例如 [3, 20, 20, ..., 20, 1]
        #   lb: 输入特征的下界，用于数据归一化
        #   ub: 输入特征的上界，用于数据归一化
        
        # 边界条件设置 - 用于数据归一化
        self.lb = torch.tensor(lb).float().to(device)  # 输入特征下界张量
        self.ub = torch.tensor(ub).float().to(device)  # 输入特征上界张量
        self.layers = layers
        # 数据预处理 - 转换为PyTorch张量并移至指定设备
        # 内部数据点的空间坐标
        self.r = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        # 内部数据点的角度正弦和余弦值
        self.sin_theta = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        self.cos_theta = torch.tensor(X[:, 2:3], requires_grad=True).float().to(device)
        # 边界条件和初始条件的真实值
        self.u = torch.tensor(u).float().to(device)
        
        # PDE系数初值设置
        self.lambda_1 = torch.tensor([0.0], requires_grad=True).to(device)
        self.lambda_2 = torch.tensor([0.0], requires_grad=True).to(device)
        
        self.lambda_1 = torch.nn.Parameter(self.lambda_1)
        self.lambda_2 = torch.nn.Parameter(self.lambda_2)

        # 实例化深度神经网络 - 用于近似Laplace方程的解
        self.dnn = DNN(layers).to(device)
        self.dnn.register_parameter('lambda_1', self.lambda_1)
        self.dnn.register_parameter('lambda_2', self.lambda_2)

        # 配置优化器 - 使用L-BFGS算法，适合小批量问题和精确梯度计算
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),      # 需要优化的网络参数
            lr=1.0,                     # 学习率
            max_iter=50000,             # 最大迭代次数
            max_eval=50000,             # 最大函数评估次数
            history_size=50,            # 历史梯度信息保存数量
            tolerance_grad=1e-5,        # 梯度容差
            tolerance_change=1.0 * np.finfo(float).eps,  # 参数变化容差
            line_search_fn="strong_wolfe"  # 强Wolfe线搜索算法
        )
        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters())# Adam优化器
        # 迭代计数器初始化
        self.iter = 0
        
    def net_u(self, r, sin_theta, cos_theta):  
        u = self.dnn(torch.cat([r, sin_theta, cos_theta], dim=1))
        # 返回预测的物理场值
        return u

    def net_f(self, r, sin_theta, cos_theta):
        # 计算Laplace方程的物理残差项

        lambda_1 = self.lambda_1  
        lambda_2 = self.lambda_2
        u = self.net_u(r, sin_theta, cos_theta)
        
        # 使用链式法则计算关于theta的一阶导数：u_theta = u_sin_theta * cos_theta - u_cos_theta * sin_theta
        u_sin_theta = torch.autograd.grad(
            u, sin_theta,                    # 对u关于sin(theta)求导
            grad_outputs=torch.ones_like(u),  # 梯度权重设置为1
            retain_graph=True,                # 保留计算图，允许后续梯度计算
            create_graph=True                 # 创建计算图，支持高阶导数
        )[0]
        
        u_cos_theta = torch.autograd.grad(
            u, cos_theta,                    # 对u关于cos(theta)求导
            grad_outputs=torch.ones_like(u),  # 梯度权重设置为1
            retain_graph=True,                # 保留计算图，允许后续梯度计算
            create_graph=True                 # 创建计算图，支持高阶导数
        )[0]
        
        u_theta = u_sin_theta * cos_theta - u_cos_theta * sin_theta
        
        # 使用链式法则计算关于theta的二阶导数
        u_sin_theta_sin_theta = torch.autograd.grad(
            u_sin_theta, sin_theta,          # 对u_sin_theta关于sin(theta)求导
            grad_outputs=torch.ones_like(u_sin_theta),
            retain_graph=True,
            create_graph=True
        )[0]
        
        u_sin_theta_cos_theta = torch.autograd.grad(
            u_sin_theta, cos_theta,          # 对u_sin_theta关于cos(theta)求导
            grad_outputs=torch.ones_like(u_sin_theta),
            retain_graph=True,
            create_graph=True
        )[0]
        
        u_cos_theta_sin_theta = torch.autograd.grad(
            u_cos_theta, sin_theta,          # 对u_cos_theta关于sin(theta)求导
            grad_outputs=torch.ones_like(u_cos_theta),
            retain_graph=True,
            create_graph=True
        )[0]
        
        u_cos_theta_cos_theta = torch.autograd.grad(
            u_cos_theta, cos_theta,          # 对u_cos_theta关于cos(theta)求导
            grad_outputs=torch.ones_like(u_cos_theta),
            retain_graph=True,
            create_graph=True
        )[0]
        
        # 计算二阶导数 u_thetatheta
        # 链式法则：d²u/dθ² = d/θ (u_sin_theta * cos_theta - u_cos_theta * sin_theta)
        u_thetatheta = (u_sin_theta_sin_theta * cos_theta * cos_theta - 
                        u_sin_theta_cos_theta * sin_theta - 
                        u_sin_theta * sin_theta - 
                        u_cos_theta_sin_theta * sin_theta * cos_theta + 
                        u_cos_theta_cos_theta * sin_theta * sin_theta - 
                        u_cos_theta * cos_theta)
        
        u_r = torch.autograd.grad(
            u, r,                     # 对u关于r求导
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True         # 添加create_graph=True参数，确保可以计算u_xx的梯度
        )[0]
        
        # 计算二阶空间导数 ∂²u/∂r²（电势的空间曲率）
        u_rr = torch.autograd.grad(
            u_r, r,                   # 对u_r关于r求导，得到二阶导数
            grad_outputs=torch.ones_like(u_r),
            retain_graph=True,
            create_graph=True
        )[0]
        
        # 计算Laplace方程的残差
        f = u_rr + lambda_1 * u_r/r +lambda_2 * u_thetatheta/r**2
        
        # 返回计算得到的物理残差
        # 在PINN训练过程中，最小化此残差是确保解满足物理方程的关键
        return f
    
    def loss_func(self):
        u_pred = self.net_u(self.r, self.sin_theta, self.cos_theta)  # 预测电势场
        f_pred = self.net_f(self.r, self.sin_theta, self.cos_theta)  # 计算残差
        loss = torch.mean((self.u - u_pred) ** 2) + torch.mean(f_pred ** 2)  # 总损失：数据损失 + 物理损失
        self.optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播
        
        self.iter += 1  # 迭代计数加1
        if self.iter % 100 == 0:  # 每100次迭代打印一次信息
            print(
                'It: %d, Loss: %e, l1: %.5f, l2: %.5f' % 
                (
                    self.iter,
                    loss.item(), 
                    self.lambda_1.item(), 
                    self.lambda_2.item()
                )
            )
        return loss

    def train(self, nIter):  # 训练函数
        self.dnn.train()  # 设置网络为训练模式
        for epoch in range(nIter):  # 迭代训练
            u_pred = self.net_u(self.r, self.sin_theta, self.cos_theta)  # 预测电势场
            f_pred = self.net_f(self.r, self.sin_theta, self.cos_theta)  # 计算残差
            loss =  torch.mean((self.u - u_pred) ** 2) + torch.mean(f_pred ** 2)  # 总损失
            self.iter += 1  # 迭代计数加1
            # Backward and optimize
            self.optimizer_Adam.zero_grad()  # 清零梯度
            loss.backward()  # 反向传播
            self.optimizer_Adam.step()  # 更新参数
            
            if epoch % 100 == 0:  # 每100次迭代打印一次信息
                print(
                    'It: %d, Loss: %.3e, Lambda_1: %.3f, Lambda_2: %.6f' % 
                    (
                        epoch, 
                        loss.item(), 
                        self.lambda_1.item(), 
                        self.lambda_2.item()
                    )
                )
        # Backward and optimize
        self.optimizer.step(self.loss_func)  # 使用L-BFGS优化器进一步优化

    def predict(self, X):  # 预测函数
        r = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)  # 空间r坐标输入
        sin_theta = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)  # 角度正弦值
        cos_theta = torch.tensor(X[:, 2:3], requires_grad=True).float().to(device)  # 角度余弦值

        self.dnn.eval()  # 设置网络为评估模式
        u = self.net_u(r, sin_theta, cos_theta)  # 预测电势场
        f = self.net_f(r, sin_theta, cos_theta)  # 计算残差
        u = u.detach().cpu().numpy()  # 将结果从GPU转移到CPU，并转换为NumPy数组
        f = f.detach().cpu().numpy()  # 将结果从GPU转移到CPU，并转换为NumPy数组
        return u, f
    
    def save_model(self, filename='Identification_model.pth'):
        # 保存模型到Model文件夹
        # 参数说明：
        #   filename: 保存的模型文件名，默认为'Identification_model.pth'
        
        # 获取项目根目录路径
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # 模型保存目录
        model_dir = os.path.join(root_dir, 'Model')
        
        # 如果目录不存在则创建
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # 构建完整的模型文件路径
        model_path = os.path.join(model_dir, filename)
        
        # 保存模型状态字典
        torch.save({
            'dnn_state_dict': self.dnn.state_dict(),
            'layers': self.layers,
            'lb': self.lb.cpu().numpy(),
            'ub': self.ub.cpu().numpy()
        }, model_path)
        
        print(f"模型已成功保存到: {model_path}")
    
    def load_model(self, filename='Identification_model.pth'):
        # 从Model文件夹加载模型
        # 参数说明：
        # filename: 加载的模型文件名，默认为'Identification_model.pth'
        
        # 获取项目根目录路径
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # 构建完整的模型文件路径
        model_path = os.path.join(root_dir, 'Model', filename)
        
        # 加载模型状态字典
        checkpoint = torch.load(model_path, weights_only=False)
        
        # 恢复模型参数
        self.dnn.load_state_dict(checkpoint['dnn_state_dict'])
        self.layers = checkpoint['layers']
        self.lb = torch.tensor(checkpoint['lb']).float().to(device)
        self.ub = torch.tensor(checkpoint['ub']).float().to(device)
        
        print(f"模型已成功从: {model_path} 加载")