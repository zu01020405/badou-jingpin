import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：
# 0：第一个大于第六个数
# 1：第二个大于第五个数
# 2：第三个大于第四个数
# 3：其他
# 优先级从上往下，如第一个数大于第六个，第二个大于第五个，则判断为第0类
"""
# 定义TorchModel类，这是一个简单的神经网络模型。
class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TorchModel, self).__init__()
        # 线性层，将输入映射到输出大小
        self.linear = nn.Linear(input_size, hidden_size)
        # self.linear2 = nn.Linear(hidden_size, output_size)
        # 在第1维上使用Softmax激活函数，将原始分数转换为概率
        self.activation = nn.Softmax(dim=1)
        # 使用CrossEntropyLoss用于多类别分类问题
        self.loss = nn.CrossEntropyLoss()

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)
        y_pred = self.activation(x)
        # y_pred = self.linear2(y_pred)
        # y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 函数用于生成一个带有对应标签的单个样本。
# 随机生成一个6维向量，对应不同的类别,
# 优先级从上往下，如第一个数大于第六个，第二个大于第五个，则判断为第0类
# 0：第一个大于第六个数
# 1：第二个大于第五个数
# 2：第三个大于第四个数
# 3：其他
def build_sample():
    x = np.random.random(6)
    # 通过比较输入的不同元素来确定标签
    if x[0] > x[5]:
        return x, torch.tensor(0)  # 标签0
    elif x[1] > x[4]:
        return x, torch.tensor(1)  # 标签1
    elif x[2] > x[3]:
        return x, torch.tensor(2)  # 标签2
    else:
        return x, torch.tensor(3)  # 标签3


# 函数用于构建包含给定样本数的数据集。
def build_dataset(total_sample_num):
    X = []
    Y = []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


#为四的时候最精确,抽到高维反而loss提高，应该是信息丢失，这种模型用不到高维
hidden_size=4
def main():
    # 配置参数
    input_size = 6
    output_size = 4
    sample_num = 10000
    batch_size = 5
    epoch_num = 10
    learning_rate = 0.08

    # 创建模型
    model = TorchModel(input_size, hidden_size)
    # model = TorchModel(input_size, hidden_size,output_size)

    # 构建训练和验证数据集
    train_x, train_y = build_dataset(sample_num)
    val_x, val_y = build_dataset(5000)
    train_dataset = TensorDataset(torch.FloatTensor(train_x), torch.LongTensor(train_y))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(torch.FloatTensor(val_x), torch.LongTensor(val_y))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 定义优化器和学习率调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    # 训练循环
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, y in train_loader:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = model.loss(y_pred, y.squeeze())
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())

        train_loss = np.mean(watch_loss)
        val_loss, val_acc = evaluate(model, val_loader)
        print("Epoch %d, Train Loss: %.5f, Val Loss: %.5f, Val Acc: %.5f" % (epoch + 1, train_loss, val_loss, val_acc))
        scheduler.step(val_loss)
    torch.save(model.state_dict(), "model.pth")


# 评估函数，用于计算数据集上的损失和准确率。
def evaluate(model, data_loader):
    model.eval()
    watch_loss = []
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            y_pred = model(x)
            loss = model.loss(y_pred, y.squeeze())
            watch_loss.append(loss.item())
            _, predicted = torch.max(y_pred, 1)
            total += y.size(0)
            correct += (predicted == y.squeeze()).sum().item()

    val_loss = np.mean(watch_loss)
    val_acc = correct / total
    return val_loss, val_acc


# 函数用于使用训练好的模型预测输入向量的类别标签。
def predict(model_path, input_vec):
    input_size = 6
    output_size = 4
    model = TorchModel(input_size, hidden_size)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())

    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_vec)
        result = model(input_tensor)
        _, predicted = torch.max(result, dim=1)
        for vec, res in zip(input_vec, predicted):
            print("输入：%s, 预测类别：%d" % (vec, res))


if __name__ == '__main__':
    # 训练模型并保存
    main()

    # 使用训练好的模型预测输入向量的类别标签
    test_vec = [
        [0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843, 0.18920843],
        [0.04963533, 0.9524256, 0.95758807, 0.9520434, 0.84890681, 0.18920843],
        [0.0797868, 0.007482528, 0.53625847, 0.34675372, 0.09871392, 0.28920843],
        [0.89349776, 0.59416669, 0.92579291, 0.41567412, 0.9358894, 0.98920843]
    ]
    predict("model.pth", test_vec)
