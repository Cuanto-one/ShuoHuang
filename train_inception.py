"""
InceptionTime 模型训练脚本
用于 DTE-L 机车故障分类（多变量时间序列分类）
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, f1_score
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
DATA_PATH = "data/data.npz"
LABEL_MAP_PATH = "data/label_map.json"

BATCH_SIZE = 32
EPOCHS = 300
LEARNING_RATE = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 加载数据 ====================
print("加载数据...")
data = np.load(DATA_PATH)
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']

print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")


def to_tensor(x, y):
    return torch.FloatTensor(x), torch.LongTensor(y)


train_ds = TensorDataset(*to_tensor(X_train, y_train))
val_ds = TensorDataset(*to_tensor(X_val, y_val))
test_ds = TensorDataset(*to_tensor(X_test, y_test))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# 加载标签映射
with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
    label2idx = json.load(f)
idx2label = {v: k for k, v in label2idx.items()}
NUM_CLASSES = len(label2idx)
print(f"类别数: {NUM_CLASSES}")
print(f"标签映射: {label2idx}")


# ==================== 类别权重（处理不平衡） ====================
def compute_class_weights(y):
    classes = np.arange(NUM_CLASSES)
    counts = np.array([np.sum(y == c) for c in classes])
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * NUM_CLASSES
    return torch.FloatTensor(weights).to(DEVICE)


class_weights = compute_class_weights(y_train)
print("类别权重:", class_weights.cpu().numpy())


# ==================== InceptionTime 模型定义 ====================
class InceptionModule(nn.Module):
    """单个 Inception 模块：多尺度并行卷积"""

    def __init__(self, in_channels, out_channels, kernel_sizes=[9, 19, 39]):
        super().__init__()
        # 1x1 卷积（瓶颈层）
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        # 三个不同尺度的卷积
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_sizes[0],
                               padding=kernel_sizes[0] // 2)
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_sizes[1],
                               padding=kernel_sizes[1] // 2)
        self.conv4 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_sizes[2],
                               padding=kernel_sizes[2] // 2)
        # 合并后的瓶颈层
        self.bottleneck = nn.Conv1d(out_channels * 4, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        p1 = self.conv1(x)
        p2 = self.conv2(x)
        p3 = self.conv3(x)
        p4 = self.conv4(x)
        concat = torch.cat([p1, p2, p3, p4], dim=1)  # 在通道维度拼接
        out = self.bottleneck(concat)
        out = self.bn(out)
        out = self.relu(out)
        return out

class InceptionTime(nn.Module):
    """InceptionTime 主体：堆叠多个 Inception 模块"""

    def __init__(self, input_channels, num_classes, depth=6, hidden_channels=32):
        super().__init__()
        self.inception_modules = nn.ModuleList()
        in_ch = input_channels
        for _ in range(depth):
            self.inception_modules.append(InceptionModule(in_ch, hidden_channels))
            in_ch = hidden_channels  # 每个模块输出 hidden_channels
        self.gap = nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        # x: (batch, time_steps, features) → (batch, features, time_steps)
        x = x.transpose(1, 2)
        for module in self.inception_modules:
            x = module(x)
        x = self.gap(x).squeeze(-1)  # (batch, hidden_channels)
        x = self.dropout(x)
        x = self.fc(x)
        return x


model = InceptionTime(input_channels=X_train.shape[2], num_classes=NUM_CLASSES).to(DEVICE)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20)


# ==================== 评估函数 ====================
def evaluate(loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            preds.append(out.argmax(dim=1).cpu())
            trues.append(yb.cpu())
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    return preds, trues


# ==================== 训练循环 ====================
print("\n开始训练...")
best_f1 = 0.0
best_epoch = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)

    avg_loss = total_loss / len(train_ds)

    # 验证
    val_preds, val_trues = evaluate(val_loader)
    f1_macro = f1_score(val_trues, val_preds, average='macro', zero_division=0)
    scheduler.step(f1_macro)

    # 保存最佳模型
    if f1_macro > best_f1:
        best_f1 = f1_macro
        best_epoch = epoch + 1
        torch.save(model.state_dict(), "best_model_inception.pth")

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Macro-F1: {f1_macro:.4f}")

print(f"\n训练完成！最佳验证 Macro-F1: {best_f1:.4f} (Epoch {best_epoch})")

# ==================== 测试集评估 ====================
print("\n加载最佳模型进行测试...")
# 测试集评估
model.load_state_dict(torch.load("best_model_inception.pth"))
test_preds, test_trues = evaluate(test_loader)

# 动态获取测试集中实际存在的标签ID
present_labels = np.unique(np.concatenate([test_trues, test_preds]))
target_names = [idx2label[i] for i in present_labels]

print("\n测试集分类报告:")
print(classification_report(test_trues, test_preds, labels=present_labels, target_names=target_names, zero_division=0))

# 各类别F1（只计算测试集中出现的类别）
print("\n各类别 F1-score:")
test_f1 = f1_score(test_trues, test_preds, labels=present_labels, average=None, zero_division=0)
for i, f1 in zip(present_labels, test_f1):
    print(f"  {idx2label[i]}: {f1:.4f}")