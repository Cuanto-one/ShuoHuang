"""
数据验证脚本：检查 processed_data.npz 是否正确。
使用前请确保已安装依赖：pandas numpy scikit-learn matplotlib openpyxl
"""
import os
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

class Tee:
    def __init__(self, filename):
        self.console = sys.stdout
        self.file = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.console.write(message)   # 写到屏幕
        self.file.write(message)      # 写到文件

    def flush(self):
        self.console.flush()
        self.file.flush()

# ==================== 动态路径设置 ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_NPZ = os.path.join(SCRIPT_DIR, "data.npz")
LABEL_JSON = os.path.join(SCRIPT_DIR, "label_map.json")
BASE_DIR = os.path.join(PROJECT_ROOT, "第二批次导出")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "check_data_output.txt")
sys.stdout = Tee(OUTPUT_FILE)

# ==================== 设置中文字体 ====================
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

def find_one_labeled_xlsx(base):
    for root, dirs, files in os.walk(base):
        for f in files:
            if f.endswith('_labeled.xlsx'):
                return os.path.join(root, f)
    return None

# ==================== 第一步：加载数据 ====================
print("="*60)
print("加载数据...")
data = np.load(DATA_NPZ)
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']

with open(LABEL_JSON, 'r', encoding='utf-8') as f:
    label2idx = json.load(f)
idx2label = {v: k for k, v in label2idx.items()}

print("训练集形状:", X_train.shape, y_train.shape)
print("验证集形状:", X_val.shape, y_val.shape)
print("测试集形状:", X_test.shape, y_test.shape)
print("特征矩阵 dtype:", X_train.dtype)
print("标签 dtype:", y_train.dtype)
print("包含 NaN:", np.isnan(X_train).any())
print("包含 Inf:", np.isinf(X_train).any())

# ==================== 第二步：标签分布 ====================
def show_dist(y, name):
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n{name} 标签分布:")
    for u, c in zip(unique, counts):
        label_name = idx2label.get(u, f"未知标签{u}")
        print(f"  {label_name} (id:{u}): {c} 个")

show_dist(y_train, "训练集")
show_dist(y_val, "验证集")
show_dist(y_test, "测试集")

# ==================== 第三步：获取特征列名索引 ====================
feature_cols = None
sample_xlsx = find_one_labeled_xlsx(BASE_DIR)
if sample_xlsx:
    print(f"\n使用示例文件获取特征列名: {sample_xlsx}")
    try:
        df_sample = pd.read_excel(sample_xlsx)
        drop_cols = ['记录时间', '状态', '机车号', '本机IP', 'IMSI信息',
                     '小区信息', '列尾ID', '软件版本']
        feature_cols = [c for c in df_sample.columns if c not in drop_cols]
        print("特征列顺序（前15个）:", feature_cols[:15])
    except Exception as e:
        print("读取示例文件失败:", e)
else:
    print("未找到 _labeled.xlsx 文件，无法自动获取特征列名。")

# 尝试根据列名定位关键特征索引
key_feature_names = ['附着状态', '连接状态', '信号强度', '空口数据接收状态']
key_indices = {}
if feature_cols:
    for name in key_feature_names:
        if name in feature_cols:
            key_indices[name] = feature_cols.index(name)
else:
    # 回退到常见索引（基于之前的输出）
    key_indices = {'附着状态': 9, '连接状态': 10, '信号强度': 11, '空口数据接收状态': 12}

# ==================== 第四步：可视化抽查 ====================
target_label_name = '网络状态异常'
if target_label_name in label2idx:
    target_id = label2idx[target_label_name]
    train_indices = np.where(y_train == target_id)[0]
    if len(train_indices) > 0:
        sample_idx = train_indices[0]
        window = X_train[sample_idx]
        print(f"\n可视化抽查: 类别 '{target_label_name}' 的一个窗口 (样本索引 {sample_idx})")

        plt.figure(figsize=(12, 8))
        time_axis = np.arange(window.shape[0])
        anomaly_time = 60

        plot_items = [
            (key_indices.get('附着状态'), '附着状态', 'b-'),
            (key_indices.get('连接状态'), '连接状态', 'r-'),
            (key_indices.get('信号强度'), '信号强度', 'g-'),
            (key_indices.get('空口数据接收状态'), '空口数据接收', 'm-')
        ]

        for idx, name, style in plot_items:
            if idx is not None and idx < window.shape[1]:
                plt.plot(time_axis, window[:, idx], style, label=name)
            else:
                print(f"警告: 特征 '{name}' 索引不存在或越界")

        plt.axvline(x=anomaly_time, color='k', linestyle='--', label='异常检测点 (t=60s)')
        plt.xlabel('时间 (秒)')
        plt.ylabel('特征值')
        plt.title(f"样本类别: {target_label_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(SCRIPT_DIR, "sample_check.png")
        plt.savefig(out_path)
        print(f"可视化图片已保存为 {out_path}")
        plt.close()
    else:
        print(f"类别 '{target_label_name}' 在训练集中无样本。")
else:
    print(f"未找到类别 '{target_label_name}'。")

# ==================== 第五步：快速可学性检查 ====================
print("\n快速可学性检查：使用随机森林")
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)

clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train_flat, y_train)
y_pred = clf.predict(X_val_flat)

# 获取验证集中实际出现的标签
unique_val_labels = np.unique(np.concatenate([y_val, y_pred]))
target_names = [idx2label.get(i, f"标签{i}") for i in sorted(unique_val_labels)]
print("\n验证集分类报告:")
print(classification_report(y_val, y_pred, labels=sorted(unique_val_labels),
                            target_names=target_names, zero_division=0))

print("\n验证完成！请检查图形和分类报告，确认数据正确。")

sys.stdout.file.close()
sys.stdout = sys.stdout.console