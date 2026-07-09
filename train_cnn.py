"""
简单 1D-CNN 训练脚本 —— 适用于小样本多变量时序分类
结果自动保存到 train_cnn_record/model_<时间戳>/ 文件夹中
包括训练曲线可视化
"""
import os, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, f1_score
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，不弹窗，直接保存文件
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "data.npz")
LABEL_MAP_PATH = os.path.join(SCRIPT_DIR, "data", "label_map.json")

BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 创建结果保存文件夹 ====================
RECORD_BASE = os.path.join(SCRIPT_DIR, "train_cnn_record")
os.makedirs(RECORD_BASE, exist_ok=True)

# 用当前时间生成子文件夹名，例如 model_2025-07-01_14-30-22
time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
SAVE_DIR = os.path.join(RECORD_BASE, f"model_{time_str}")
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"本次训练结果将保存至: {SAVE_DIR}")

# ==================== 加载数据 ====================
print("加载数据...")
data = np.load(DATA_PATH)
# X表示已知条件数据（滑动窗口截取的一段信号变化），Y表示对应的故障类别编号
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']

# 把数据集数据转换为模型训练所需张量格式
def to_tensor(x, y):
    return torch.FloatTensor(x), torch.LongTensor(y)

# 构建数据与标签一一对应的数据集
train_ds = TensorDataset(*to_tensor(X_train, y_train))
val_ds = TensorDataset(*to_tensor(X_val, y_val))
test_ds = TensorDataset(*to_tensor(X_test, y_test))

# DataLoader负责从数据集中拿取数据给模型训练，batch_size代表每次拿取的大小，shuffle【洗牌】代表是否随机打乱
# 训练肯定要随机打乱，不然模型可以通过记住排序来推断答案
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# 把异常类型用数字代替
with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
    label2idx = json.load(f)
idx2label = {v: k for k, v in label2idx.items()}
NUM_CLASSES = len(label2idx)

# 类别权重，参数y表示样本标签，让模型更好的关注数量少的样本
def compute_class_weights(y):
    classes = np.arange(NUM_CLASSES)
    counts = np.array([np.sum(y == c) for c in classes])
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    # 归一化，让所有权重值平均值为1，保证模型学习节奏平稳
    weights = weights / weights.sum() * NUM_CLASSES
    # weights = weights / weights.sum()
    return torch.FloatTensor(weights).to(DEVICE)

class_weights = compute_class_weights(y_train)

# ==================== 简单 1D-CNN 模型 ====================
class SimpleCNN1D(nn.Module):
    # 工具
    def __init__(self, input_channels, num_classes):
        super().__init__()
        # 卷积层（nn.Conv1d）：特征提取器
        # 四个参数，intput（输入的数据量），output（超参数，经过卷积运算输出的数据通道数，也就是数据特征），kernel_size代表窗口长度为7秒，padding表示为了满足原有的240秒的滑动窗口大小不变）
        # 也就是每7秒的数据通过卷积运算变成新的1秒的数据，最后构成新的240秒的数据并得到64个数据特征
        # 7秒是开发者经验所得，需要人来调整得超参数，padding是让输出与输入个数相同，一一对应，左右各padding个0来补充
        # padding=(kernel_size - 1)/2.
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, padding=3)
        # 批归一化层（nn.BatchNorm1d）：稳定剂，把每一步得数据都拉到均值为0，方差为1得标准范围内，让数据再同一数据量级，避免模型区别对待数据特征
        self.bn1 = nn.BatchNorm1d(64)
        # 池化层（nn.MaxPoll1d）：降维与聚焦，每2秒信息压缩成1秒，保留最强的信号。减少模型计算量
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        # 全局平均池化（nn.AdaptiveAvgPool1d）：压缩成精华，直接把前面的数据特征压缩为256个数字（240秒压缩为1秒的数据）
        self.gap = nn.AdaptiveAvgPool1d(1)
        # Dropout与全连接层（nn.Dropout,nn.Linear）：决策者
        # 随机“关闭”一半的神经元，强迫剩下的神经元学会通用的规律
        # 防止过拟合（死记硬背），也就是模型不能准确判断训练集以外的数据了
        self.dropout = nn.Dropout(0.5)
        # 用256个浓缩特征，通过线性变换，输出属于每个类别的“得分”
        self.fc = nn.Linear(256, num_classes)

    # 实际操作的流水线
    # 输入->转置->【卷积->归一化->激活->池化】x2->【卷积->归一化->激活】->全局池化->展平->随机失活->全连接输出
    # torch.relu是一个激活函数，负责把负数变成0，只保留正数。这引入了非线性，让模型能够学习复杂的关系。
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
# 计算预测分数
model = SimpleCNN1D(input_channels=X_train.shape[2], num_classes=NUM_CLASSES).to(DEVICE)
# 损失函数
criterion = nn.CrossEntropyLoss(weight=class_weights)
# 优化器（根据梯度更新参数，自动调整每个参数的学习率，变化快的降低，变化慢的提高）
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# 学习率调度器：当模型在patience=20个训练轮数内F1分数不再上升，则表示要接近最佳模型，则需要降低训练节奏，也就是学习率来避免错过最优点，每次降低一半
# 当训练集损失在持续下降，但验证集F1不再上升甚至开始下降时，过拟合就发生了。
# F1分数就是解决“准确率幻觉”的手段，让模型既不漏报又不误报，也就是一个综合测评的手段。
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20)

# ==================== 评估函数 ====================
# 对模型进行考试
def evaluate(loader):
    # 切换为考试模式，Dropout就不会关闭神经元了，确保同一份试卷是一个人作答
    model.eval()
    # 装考卷和标准答案
    preds, trues = [], []
    # 停止更新参数，不让偷偷学习
    with torch.no_grad():
        # 发试卷
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            # 模型测试
            out = model(xb)
            # 只取最有把握的答案，确定模型预测的类别是什么
            preds.append(out.argmax(dim=1).cpu())
            # 收试卷
            trues.append(yb.cpu())
    # 返回答卷和标准答案
    return torch.cat(preds).numpy(), torch.cat(trues).numpy()

# ==================== 训练循环 ====================
best_f1 = 0.0
best_model_path = os.path.join(SAVE_DIR, "best_model_cnn.pth")
best_epoch = 0

# 【新增】用于记录训练历史的列表
train_losses = []
val_f1_scores = []

for epoch in range(EPOCHS):
    # 切换模型为训练模式
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        # 清空之前批次数据的梯度信息
        optimizer.zero_grad()
        # 计算损失值
        loss = criterion(model(xb), yb)
        # 反向传播，计算每个可学习的参数对错误的贡献程度（梯度）
        loss.backward()
        # 调整梯度和学习率
        optimizer.step()
        # loss.item()是pytorch返回的这个批次样本数据返回的平均损失值
        total_loss += loss.item() * xb.size(0)

    avg_train_loss = total_loss / len(train_ds)
    train_losses.append(avg_train_loss)  # 【新增】记录训练损失

    # 随堂测试
    val_preds, val_trues = evaluate(val_loader)
    f1_macro = f1_score(val_trues, val_preds, average='macro', zero_division=0)
    val_f1_scores.append(f1_macro)  # 【新增】记录验证F1

    # 调整学习节奏
    scheduler.step(f1_macro)

    # 保存最佳模型
    if f1_macro > best_f1:
        best_f1 = f1_macro
        best_epoch = epoch  # 【新增】记录最佳 epoch，便于在图上标注
        torch.save(model.state_dict(), best_model_path)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_train_loss:.4f} | Val Macro-F1: {f1_macro:.4f}")

# ==================== 绘制训练曲线 ====================
# 创建画布和坐标轴对象
fig, ax1 = plt.subplots(figsize=(10, 6))
epochs_range = np.arange(1, EPOCHS + 1)

# ---------- 左轴：训练损失 ----------
ax1.plot(epochs_range, train_losses, label='Training Loss', color='tab:blue', linewidth=1.5)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# ---------- 右轴：验证F1 ----------
ax2 = ax1.twinx()
ax2.plot(epochs_range, val_f1_scores, label='Validation Macro-F1', color='tab:red', linewidth=1.5)
ax2.set_ylabel('Macro-F1', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

# ---------- 标注最佳F1点 ----------
if best_f1 > 0 and best_epoch is not None:
    ax2.scatter(best_epoch + 1, best_f1, color='darkred', s=50, zorder=5)
    ax2.text(best_epoch + 1, best_f1 - 0.02, f'Best F1: {best_f1:.3f}',
             color='darkred', fontsize=10, ha='center')

# ---------- 合并左右轴的图例 ----------
# 分别从两个轴获取线条和标签
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
# 使用 fig.legend() 在图形级别添加图例，避免轴干扰
fig.legend(lines1 + lines2,
    labels1 + labels2,
    loc='center right',       # 图例的某角与锚点对齐
    bbox_to_anchor=(1, 0.5),  # 锚点位置
    bbox_transform=ax1.transAxes)

plt.title('Training Curve')
fig.tight_layout()  # 注意：使用 fig.tight_layout()，而不是 plt.tight_layout()
grid = ax1.grid(True, alpha=0.3)  # 在左轴上添加网格即可

# 保存图像
curve_path = os.path.join(SAVE_DIR, 'training_curve.png')
fig.savefig(curve_path, dpi=150)
plt.close(fig)
print(f"训练曲线已保存至: {curve_path}")

# ==================== 测试与保存结果 ====================
# 用最好的模型参加测试集的毕业考试
model.load_state_dict(torch.load(best_model_path))
test_preds, test_trues = evaluate(test_loader)

# 生成与保存成绩单
# 找出实际出现过的类别编号
present_labels = np.unique(np.concatenate([test_trues, test_preds]))
# 只为出现过的标签找中文名称
target_names = [idx2label[i] for i in present_labels]
report = classification_report(test_trues, test_preds, labels=present_labels, target_names=target_names, zero_division=0)

print("\n测试集分类报告:")
print(report)

# 将测试报告和关键参数写入结果文件
result_path = os.path.join(SAVE_DIR, "test_report.txt")
with open(result_path, 'w', encoding='utf-8') as f:
    f.write(f"训练时间: {time_str}\n")
    f.write(f"最佳验证 Macro-F1: {best_f1:.4f}\n")
    f.write(f"模型参数: {sum(p.numel() for p in model.parameters()):,}\n")
    f.write(f"数据形状: {X_train.shape}, {X_val.shape}, {X_test.shape}\n")
    f.write(f"类别映射: {label2idx}\n")
    f.write("\n--- 测试集分类报告 ---\n")
    f.write(report)

print(f"结果已保存至: {SAVE_DIR}")