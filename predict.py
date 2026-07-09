"""
故障诊断推理脚本：对原始Excel日志进行预测，输出故障片段报告。
用法：python predict.py --input <输入Excel路径>
输出：在输入文件同目录下的 report 文件夹中生成 <输入文件名>_report.xlsx
"""
import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ==================== 配置 ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "train_cnn_record/model_2026-07-02_22-43-30/best_model_cnn.pth")  # 训练好的模型
LABEL_MAP_PATH = os.path.join(SCRIPT_DIR, "data", "label_map.json")

WINDOW_SIZE = 240
PRE_SEC = 60
POST_SEC = 180
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 加载标签映射 ====================
with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
    label2idx = json.load(f)
idx2label = {v: k for k, v in label2idx.items()}
NUM_CLASSES = len(label2idx)

# ==================== 数据清洗函数（与训练时一致） ====================
def clean_df(df):
    """清洗DataFrame，返回特征矩阵和时间戳"""
    df = df.copy()

    if '记录时间' in df.columns:
        time_col = pd.to_datetime(df['记录时间'], errors='coerce')
        if time_col.isna().all():
            time_col = pd.date_range('2025-01-01', periods=len(df), freq='s')
        df['记录时间'] = time_col
        df = df.sort_values('记录时间').reset_index(drop=True)
    else:
        df['记录时间'] = pd.date_range('2025-01-01', periods=len(df), freq='s')

    code_map = {'正常': 0, '无': 0, 'CRC故障': 1, 'IP异常': 2}
    for col in ['OCE网口故障代码', 'OCE串口故障代码', 'CIR串口故障代码',
                'AB端网口故障代码', '空口故障代码']:
        if col in df.columns:
            df[col] = df[col].map(code_map).fillna(-1).astype(int)

    if '空口命令代码' in df.columns:
        cmd_map = {'业务帧': 0, '链路检测': 1}
        df['空口命令代码'] = df['空口命令代码'].map(cmd_map).fillna(-1).astype(int)

    if '本车序号' in df.columns:
        seq_map = {'主控': 0, '从1': 1, '从2': 2, '从3': 3}
        df['本车序号'] = df['本车序号'].apply(lambda x: seq_map.get(x, x))
        df['本车序号'] = pd.to_numeric(df['本车序号'], errors='coerce').fillna(85).astype(int)

    if '上下行' in df.columns:
        dir_map = {'上行': 0, '下行': 1}
        df['上下行'] = df['上下行'].map(dir_map).fillna(-1).astype(int)

    if '操作节' in df.columns:
        df['操作节'] = df['操作节'].map({'是': 1, '否': 0}).fillna(0).astype(int)

    numeric_cols = [
        'OCE网口', 'OCE串口', 'CIR串口', 'AB端网口', 'AT串口',
        '附着状态', '连接状态', '信号强度', '空口数据接收状态',
        '寻址状态', '编组状态', 'OCE状态号', '编组数量',
        '主控机车号', '从1机车号', '参考信号接收功率', '信噪比',
        '小区号', '频率', '参考信号接收质量', '接收信号强度指示',
        '列尾信息获取状态', '列尾连接状态', '列尾寻址状态',
        '控制板累计使用时间', '拨号次数', '读取SIM卡失败次数',
        '模块日志读取状态', '软关机次数', '自愈重启次数',
        '本机空口无数据计数', '开机时间', '可用内存', '承诺内存',
        'TAX公里标', 'TAX速度', 'GPS速度', '线程'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1)

    if '全体空口无数据计数' in df.columns:
        def parse_global_count(val):
            try:
                parts = str(val).split('-')
                return sum(int(x) for x in parts[:4] if x.isdigit())
            except:
                return -1
        df['全体空口无数据计数'] = df['全体空口无数据计数'].apply(parse_global_count)

    drop_cols = ['记录时间', '状态', '机车号', '本机IP', 'IMSI信息',
                 '小区信息', '列尾ID', '软件版本']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    df_features = df[feature_cols].astype(np.float32)

    return df_features, df['记录时间'].values

# ==================== 模型定义 ====================
class SimpleCNN1D(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ==================== 推理函数 ====================
def predict_file(filepath, model):
    # 优先使用 _labeled.xlsx（如果存在且输入是 .xls）
    if filepath.endswith('.xls'):
        base = filepath.rsplit('.', 1)[0]
        labeled_path = base + '_labeled.xlsx'
        if os.path.exists(labeled_path):
            filepath = labeled_path

    # 读取文件，根据后缀选择引擎
    if filepath.endswith('.xls'):
        # 真正的旧版 .xls
        df = pd.read_excel(filepath, engine='xlrd')
    else:
        # .xlsx 或伪装成 .xls 的 .xlsx 文件
        df = pd.read_excel(filepath, engine='openpyxl')
    df_feat, timestamps = clean_df(df)
    feat_array = df_feat.values
    total_seconds = feat_array.shape[0]

    if total_seconds < WINDOW_SIZE:
        raise ValueError("文件时间长度不足240秒，无法生成窗口。")

    predictions = np.zeros(total_seconds, dtype=int)
    valid_mask = np.zeros(total_seconds, dtype=bool)

    model.eval()
    with torch.no_grad():
        for center in range(PRE_SEC, total_seconds - POST_SEC):
            start = center - PRE_SEC
            end = center + POST_SEC
            window = feat_array[start:end]
            x = torch.FloatTensor(window).unsqueeze(0).to(DEVICE)
            out = model(x)
            pred = out.argmax(dim=1).cpu().numpy()[0]
            predictions[center] = pred
            valid_mask[center] = True

    valid_indices = np.where(valid_mask)[0]
    valid_preds = predictions[valid_indices]
    valid_times = timestamps[valid_indices]

    # 中值滤波平滑
    smoothed_preds = pd.Series(valid_preds).rolling(window=5, center=True, min_periods=1).median().values

    # 片段提取
    segments = []
    if len(smoothed_preds) == 0:
        return segments

    current_label = smoothed_preds[0]
    start_time = valid_times[0]
    for i in range(1, len(smoothed_preds)):
        if smoothed_preds[i] != current_label:
            end_time = valid_times[i-1]
            segments.append({
                'start_time': pd.Timestamp(start_time),
                'end_time': pd.Timestamp(end_time),
                'fault_type': idx2label[current_label]
            })
            start_time = valid_times[i]
            current_label = smoothed_preds[i]
    segments.append({
        'start_time': pd.Timestamp(start_time),
        'end_time': pd.Timestamp(valid_times[-1]),
        'fault_type': idx2label[current_label]
    })

    return segments

# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser(description="DTE-L 故障诊断推理")
    parser.add_argument("--input", type=str, required=True, help="输入Excel文件路径")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    # 自动生成输出路径：在输入文件同目录下创建 reports 文件夹
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)

    # 输出文件名：输入文件名_report.xlsx
    input_basename = os.path.splitext(os.path.basename(input_path))[0]
    output_file = os.path.join(output_dir, f"{input_basename}_report.xlsx")

    # 加载模型
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}，请先训练模型。")
    model = SimpleCNN1D(input_channels=49, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 推理
    segments = predict_file(input_path, model)
    if not segments:
        print("未检测到任何有效预测窗口。")
        return

    # 保存结果
    df_segments = pd.DataFrame(segments)
    df_segments.to_excel(output_file, index=False)
    print(f"故障诊断报告已保存至: {output_file}")
    print("故障片段摘要：")
    print(df_segments)

if __name__ == "__main__":
    main()