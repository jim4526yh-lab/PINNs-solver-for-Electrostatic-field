import os
import torch
import numpy as np
from collections import OrderedDict

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
            定义神经网络各层神经元数量的列表，例如[2, 20, 20, 1]表示
            输入层2个神经元，两个隐藏层各20个神经元，输出层1个神经元
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
    def __init__(self, X_u, u, X_f, layers, lb, ub):
        # 初始化物理信息神经网络模型
        # 参数说明：
        #   X_u: 边界条件和初始条件的空间-时间坐标点，形状[样本数, 2]
        #   u: 边界条件和初始条件的真实值，形状[样本数, 1]
        #   X_f: 用于满足物理方程的数据点，形状[样本数, 2]
        #   layers: 神经网络层结构配置，例如 [2, 20, 20, ..., 20, 1]
        #   lb: 输入特征的下界，用于数据归一化
        #   ub: 输入特征的上界，用于数据归一化
        
        # 边界条件设置 - 用于数据归一化
        self.lb = torch.tensor(lb).float().to(device)  # 输入特征下界张量
        self.ub = torch.tensor(ub).float().to(device)  # 输入特征上界张量
        
        # 数据预处理 - 转换为PyTorch张量并移至指定设备
        # 边界条件和初始条件的空间坐标
        self.x_u = torch.tensor(X_u[:, 0:1], requires_grad=True).float().to(device)
        # 边界条件和初始条件的时间坐标
        self.y_u = torch.tensor(X_u[:, 1:2], requires_grad=True).float().to(device)
        # 物理方程数据点的空间坐标
        self.x_f = torch.tensor(X_f[:, 0:1], requires_grad=True).float().to(device)
        # 物理方程数据点的时间坐标
        self.y_f = torch.tensor(X_f[:, 1:2], requires_grad=True).float().to(device)
        # 边界条件和初始条件的真实值
        self.u = torch.tensor(u).float().to(device)
        
        # 保存模型配置参数
        self.layers = layers  # 神经网络层结构
        
        # 实例化深度神经网络 - 用于近似Laplace方程的解
        self.dnn = DNN(layers).to(device)
        
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
        # 迭代计数器初始化
        self.iter = 0
        
    def net_u(self, x, y):  
        u = self.dnn(torch.cat([x, y], dim=1))
        # 返回预测的物理场值
        return u

    def net_f(self, x, y):
        # 计算Laplace方程的物理残差项

        u = self.net_u(x, y)
        
        u_y = torch.autograd.grad(
            u, y,                     # 对u关于y求导
            grad_outputs=torch.ones_like(u),  # 梯度权重设置为1
            retain_graph=True,        # 保留计算图，允许后续梯度计算
            create_graph=True         # 创建计算图，支持高阶导数
        )[0]  # autograd.grad返回梯度列表，[0]获取第一个元素

        # 4. 计算二阶空间导数 ∂²u/∂y²（电势的空间曲率）
        u_yy = torch.autograd.grad(
            u_y, y,                   # 对u_y关于y求导，得到二阶导数
            grad_outputs=torch.ones_like(u_y),
            retain_graph=True,
            create_graph=True
        )[0]
        
        u_x = torch.autograd.grad(
            u, x,                     # 对u关于x求导
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        
        # 4. 计算二阶空间导数 ∂²u/∂x²（电势的空间曲率）
        u_xx = torch.autograd.grad(
            u_x, x,                   # 对u_x关于x求导，得到二阶导数
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]
        
        # 5. 计算Laplace方程的残差
        f = u_xx + u_yy
        
        # 返回计算得到的物理残差
        # 在PINN训练过程中，最小化此残差是确保解满足物理方程的关键
        return f
    
    def loss_func(self):
        # PINN模型的损失函数计算方法
        # 计算并返回数据损失和物理约束损失的总和
        # 这是L-BFGS优化器调用的闭包函数
        
        # 1. 清除之前积累的梯度信息
        self.optimizer.zero_grad()
        
        # 2. 计算边界条件和初始条件处的预测值
        u_pred = self.net_u(self.x_u, self.y_u)
        
        # 3. 计算物理方程内部点的残差值
        f_pred = self.net_f(self.x_f, self.y_f)
        
        # 4. 计算数据损失项 - 预测值与真实值的均方误差
        #    确保网络在边界/初始条件点上拟合已知数据
        loss_u = torch.mean((self.u - u_pred) ** 2)
        
        # 5. 计算物理约束损失项 - 物理方程残差的均方误差
        #    确保网络预测的解满足Laplace方程的约束
        loss_f = torch.mean(f_pred ** 2)
        
        # 6. 计算总损失 - 数据损失和物理约束损失的加权和
        #    这里使用等权重，也可以根据需要调整权重比例
        loss = loss_u + loss_f
        
        # 7. 执行反向传播，计算梯度
        loss.backward()
        
        # 8. 更新迭代计数器
        self.iter += 1
        
        # 9. 每100次迭代打印一次损失信息，便于监控训练进度
        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e' % \
                (self.iter, loss.item(), loss_u.item(), loss_f.item())
            )
        
        # 10. 返回总损失值，供L-BFGS优化器使用
        return loss
    def train(self):
        # 训练PINN模型的方法
        # 执行一次L-BFGS优化器的完整优化过程
        
        # 1. 设置神经网络为训练模式
        #    虽然在这个简单实现中可能没有批归一化等需要区分训练/测试模式的层，
        #    但这是一个良好实践，确保所有层都处于正确的计算模式
        self.dnn.train()
                
        # 2. 执行L-BFGS优化器的一步优化
        #    L-BFGS是一种拟牛顿法优化器，适合小批量问题和精确梯度计算
        #    self.loss_func作为闭包函数传递，它内部会：
        #    - 计算当前网络的预测值
        #    - 计算损失函数（数据损失+物理约束损失）
        #    - 执行反向传播计算梯度
        #    - 返回损失值供优化器使用
        self.optimizer.step(self.loss_func)
        
        # 注意：与SGD等优化器不同，L-BFGS的step()方法会在内部多次调用loss_func
        # 并进行线搜索，以找到最优的参数更新方向和步长
        # 一次调用可能会执行多次迭代，直到达到收敛条件或最大迭代次数

    def predict(self, X):
        # 使用训练好的PINN模型进行预测
        # 参数说明：
        #   X: 输入的时空坐标点数组，形状为[样本数, 2]，其中第一列为空间坐标x，第二列为空间坐标y
        # 返回值：
        #   u: 预测的电势值，形状为[样本数, 1]
        #   f: 预测的物理方程残差，形状为[样本数, 1]，理想情况下应接近0
        
        # 1. 数据预处理 - 从输入数组中提取空间坐标x和空间坐标y，并转换为PyTorch张量
        # 设置requires_grad=True以支持梯度计算（用于计算残差）
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        y = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        # 2. 设置模型为评估模式
        # 禁用训练时的特定行为（如dropout），确保推理一致性
        self.dnn.eval()
        
        # 3. 使用模型预测电势值u(x,y)
        # 通过net_u方法调用深度神经网络进行前向传播
        u = self.net_u(x, y)
        
        # 4. 计算预测点处的物理方程残差
        # 通过net_f方法计算Laplace方程的残差，用于评估预测解的物理一致性
        f = self.net_f(x, y)
        
        # 5. 数据后处理 - 将PyTorch张量转换为NumPy数组
        # detach(): 从计算图中分离张量
        # cpu(): 将张量从GPU移至CPU（如果使用GPU）
        # numpy(): 转换为NumPy数组格式，便于后续分析和可视化
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        
        # 6. 返回预测结果
        # u: 预测的物理场值
        # f: 物理方程残差（残差越小表示解越满足物理方程约束）
        return u, f
    
    def save_model(self, filename='Inference_model.pth'):
        # 保存模型到Model文件夹
        # 参数说明：
        #   filename: 保存的模型文件名，默认为'Inference_model.pth'
        
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
    
    def load_model(self, filename='Inference_model.pth'):
        # 从Model文件夹加载模型
        # 参数说明：
        #   filename: 加载的模型文件名，默认为'laplace_model.pth'
        
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