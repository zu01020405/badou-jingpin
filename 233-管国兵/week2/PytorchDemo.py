# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import torch.optim as optim

"""

基于pytorch框架编写模型训练
实现一个自行构造的找分类(机器学习)任务
分类：x是一个6维向量，
    如果第1个数>第2个数，则为第1类
    如果第3个数>第4个数，则为第2类
    如果第4个数>第5个数，则为第2类

"""

#定义2层线性层的模型，每层后都接激活层
class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(TorchModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1) # 6 , 9
        self.layer2 = nn.Linear(hidden_size1, hidden_size2) # 9 , 3
        self.act = torch.softmax # sigmod归一化
        self.loss = nn.CrossEntropyLoss # 交叉熵

    def forward(self, x, y=None):
        hidden = self.layer1(x) # 20*6 -> 20*9
        y_pred = self.act(hidden) # 激活层不改变形状
        y_pred = self.layer2(y_pred) # 20 * 3
        y_pred = self.act(y_pred)
        if y is not None:
            loss = self.loss(y_pred, y)
            return loss
        else:
            return y_pred

# 分类规则，创建训练样本
# x是长度6的向量，y是one-hot类型
def build_sample():
    x = np.random.random(6)
    if x[0] > x[1]:
        return x, [1, 0, 0] # 第一类
    elif x[2] > x[3]:
        return x, [0, 1, 0] # 第二类
    elif x[4] > x[5]:
        return x, [0, 0, 1]# 第三类
    else:
        return build_sample();

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    while len(X) < total_sample_num:
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(Y))

X, Y = build_dataset(10)
# X = tensor([[0.6349, 0.3288, 0.9351, 0.3435, 0.5702, 0.4199],
#         [0.9741, 0.9030, 0.7117, 0.7334, 0.9790, 0.5259],
#         [0.0830, 0.0889, 0.0802, 0.7943, 0.9375, 0.3662],
#         [0.0185, 0.2207, 0.7742, 0.7664, 0.9057, 0.1738],
#         [0.3694, 0.8043, 0.1318, 0.4677, 0.3696, 0.0175],
#         [0.4653, 0.7395, 0.4892, 0.1021, 0.6315, 0.9655],
#         [0.8266, 0.9315, 0.9767, 0.8642, 0.0232, 0.0034],
#         [0.6208, 0.5251, 0.8273, 0.3856, 0.7574, 0.8058],
#         [0.7226, 0.3082, 0.8806, 0.9964, 0.6943, 0.2087],
#         [0.2635, 0.7596, 0.7196, 0.5752, 0.9399, 0.8313]])
# Y = tensor([[1., 0., 0.],
#         [1., 0., 0.],
#         [0., 0., 1.],
#         [0., 1., 0.],
#         [0., 0., 1.],
#         [0., 1., 0.],
#         [0., 1., 0.],
#         [1., 0., 0.],
#         [1., 0., 0.],
#         [0., 1., 0.]])

# 损失函数用 交叉熵，计算 y_pre 与 target 之间的损失
# CrossEntropyLoss(y_pre, target) ~= 0.3 损失，默认会为输入参数y_pre做 softMax

# def evaluate(model):
#     model.eval()
#     test_sample_num = 100
#     x, y = build_dataset(test_sample_num)
#
#     correct, wrong = 0, 0
#
#     with torch.no_grad():
#         y_pred = model(x)  # 模型预测
#
#
#     print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
#     return correct / (correct + wrong)

def main():
    epoch_num = 10 # 训练轮数
    batch_size = 20 # 每次训练样本个数
    train_sample = 5000 # 每轮训练总样本个数
    input_size = 6 # 输入向量的维度
    hidden_size1 = 9 # 隐含层维度
    hidden_size2 = 3 # 最后输出层维度
    learning_rate = 0.001 # 学习率
    # 建立模型
    model = TorchModel(input_size, hidden_size1, hidden_size2)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample) # 5000 对训练样本

    for epoch in range(epoch_num): # 一轮一轮来训练，这里是训练10轮
        model.train() # 模型调为训练模式
        watch_loss = [] # 记录loss
        for batch_index in range(train_sample // batch_size): # 循环 5000/20 次，依次取20对样本进行训练
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward() # 计算梯度
            optim.step() # 更新权重
            optim.zero_grad() # 梯度归零
            watch_loss.append(loss.item()) # loss汇总
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        # acc = evaluate(model)  # 测试本轮模型结果
        # log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 画图
    # print(log)
    # plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    # plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    # plt.legend()
    # plt.show()
    return

if __name__ == "__main__":
    main()

    # test_vec = [[0.47889086,0.15229675,0.31082123,0.03504317,0.18920843, 0.03504317],
    #             [0.44963533,0.5524256,0.95758807,0.94520434,0.84890681,0.03504317],
    #             [0.58797868,0.67482528,0.13625847,0.34675372,0.29871392, 0.03504317],
    #             [0.89349776,0.59416669,0.92579291,0.41567412,0.7358894, 0.03504317]]
    #
    # model = TorchModel()
    # # 加载模型权重
    # model.load_state_dict(torch.load("model.pth"))
    # model.eval()  # 设置模型为评估模式
    #
    #
    # # 定义预测函数
    # def predict(x):
    #     model.eval()  # 设置模型为评估模式
    #     with torch.no_grad():  # 在预测时不计算梯度，节省内存和计算资源
    #         x = torch.tensor(x, dtype=torch.float32)
    #         output = model(x.unsqueeze(0))
    #         _, predicted_class = torch.max(output, 1)  # 获取最大值对应的类别索引
    #         return predicted_class.item()
    #
    #
    # # 例子：使用预测函数进行预测
    # sample_to_predict = [0.2, 0.3, 0.1, 0.4, 0.8, 0.6]
    # predicted_class = predict(sample_to_predict)
    # print(f"Predicted Class: {predicted_class}")