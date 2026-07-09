"""
批量故障诊断推理脚本：遍历所有案例文件夹，对所有 _labeled.xlsx 进行预测，
生成一个汇总 Excel 表，包含文件名、起始时间、结束时间、故障类型。
用法：python batch_predict.py
输出：在项目根目录下生成 batch_summary_report.xlsx
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ==================== 配置 ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "train_cnn_record/model_2026-07-02_22-43-30/best_model_cnn.pth")
LABEL_MAP_PATH = os.path.join(SCRIPT_DIR, "data", "label_map.json")
BASE_DIR = os.path.join(SCRIPT_DIR, "第二批次导出")

WINDOW_SIZE = 240
PRE_SEC = 60
POST_SEC = 180
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_FILE = os.path.join(SCRIPT_DIR, "reports/batch_summary_report.xlsx")

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
def predict_dataframe(df_feat, model):
    """对清洗后的特征DataFrame进行推理，返回有效预测索引和预测标签"""
    feat_array = df_feat.values
    total_seconds = feat_array.shape[0]
    if total_seconds < WINDOW_SIZE:
        return np.array([]), np.array([])

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
    if len(valid_indices) == 0:
        return np.array([]), np.array([])

    return valid_indices, predictions[valid_indices]


def extract_segments(timestamps, valid_indices, valid_preds):
    """提取连续相同预测的片段"""
    if len(valid_preds) == 0:
        return []

    smoothed = pd.Series(valid_preds).rolling(window=5, center=True, min_periods=1).median().values
    valid_times = timestamps[valid_indices]

    segments = []
    current_label = smoothed[0]
    start_time = valid_times[0]
    for i in range(1, len(smoothed)):
        if smoothed[i] != current_label:
            end_time = valid_times[i - 1]
            segments.append({
                'start_time': pd.Timestamp(start_time),
                'end_time': pd.Timestamp(end_time),
                'fault_type': idx2label.get(current_label, f'未知({current_label})')
            })
            start_time = valid_times[i]
            current_label = smoothed[i]

    segments.append({
        'start_time': pd.Timestamp(start_time),
        'end_time': pd.Timestamp(valid_times[-1]),
        'fault_type': idx2label.get(current_label, f'未知({current_label})')
    })
    return segments


def merge_short_segments(segments, min_duration_sec=5):
    """合并持续时间极短的片段到相邻片段"""
    if len(segments) <= 1:
        return segments

    merged = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        duration = (seg['end_time'] - seg['start_time']).total_seconds()

        if duration < min_duration_sec:
            # 短片段：尝试与前后合并
            if len(merged) > 0:
                # 合并到前一个片段
                prev = merged[-1]
                prev['end_time'] = seg['end_time']
                i += 1
                continue
            elif i + 1 < len(segments):
                # 合并到后一个片段
                next_seg = segments[i + 1]
                next_seg['start_time'] = seg['start_time']
                i += 1
                continue

        merged.append(seg)
        i += 1

    return merged

# ==================== 主程序 ====================
def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}，请先训练模型。")
    model = SimpleCNN1D(input_channels=49, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 收集所有 _labeled.xlsx 文件
    labeled_files = []
    for root, dirs, files in os.walk(BASE_DIR):
        for f in files:
            if f.endswith('_labeled.xlsx'):
                labeled_files.append(os.path.join(root, f))

    print(f"共找到 {len(labeled_files)} 个标签化日志文件。")

    all_rows = []  # 存放每条故障片段

    for filepath in labeled_files:
        print(f"正在处理: {filepath}")
        try:
            df = pd.read_excel(filepath, engine='openpyxl')
        except Exception as e:
            print(f"  读取失败: {e}")
            continue

        try:
            df_feat, timestamps = clean_df(df)
        except Exception as e:
            print(f"  清洗失败: {e}")
            continue

        valid_indices, valid_preds = predict_dataframe(df_feat, model)
        if len(valid_preds) == 0:
            print(f"  无有效预测窗口，跳过。")
            continue

        segments = extract_segments(timestamps, valid_indices, valid_preds)
        segments = merge_short_segments(segments, min_duration_sec=15)  # 合并小于5秒的片段

        # 生成层级文件名：案例文件夹/机车号/文件名
        # 文件路径形如: BASE/月份/案例文件夹/机车号/xxx_labeled.xlsx
        path_parts = os.path.normpath(filepath).split(os.sep)
        # 倒数第1是文件名，倒数第2是机车号文件夹，倒数第3是案例文件夹
        if len(path_parts) >= 3:
            case_folder = path_parts[-3]  # 案例文件夹名
            loco_folder = path_parts[-2]  # 机车号
        else:
            case_folder, loco_folder = "未知案例", "未知机车"

        file_basename = os.path.splitext(os.path.basename(filepath))[0]
        # 去掉结尾的“_labeled”（如果有），让文件名更简洁
        if file_basename.endswith('_labeled'):
            file_basename = file_basename[:-len('_labeled')]

        display_name = f"{case_folder}/{loco_folder}/{file_basename}"

        for seg in segments:
            all_rows.append({
                '文件名': display_name,
                '起始时间': seg['start_time'],
                '结束时间': seg['end_time'],
                '故障类型': seg['fault_type']
            })

    if all_rows:
        df_summary = pd.DataFrame(all_rows)
        # 按文件名和起始时间排序
        df_summary = df_summary.sort_values(['文件名', '起始时间']).reset_index(drop=True)
        df_summary.to_excel(OUTPUT_FILE, index=False)
        print(f"\n汇总报告已保存至: {OUTPUT_FILE}")
        print(f"共识别出 {len(df_summary)} 个故障片段。")
        print("各故障类型片段数：")
        print(df_summary['故障类型'].value_counts())
    else:
        print("未生成任何有效报告。")


if __name__ == "__main__":
    main()