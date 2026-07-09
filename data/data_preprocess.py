"""
数据预处理脚本 v2：增加样本量，包含正常窗口和持续异常的重叠采样。
"""
import os
import re
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ==================== 配置参数 ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BASE_DIR = os.path.join(PROJECT_ROOT, "第二批次导出")

WINDOW_SIZE = 240
PRE_SEC = 60
POST_SEC = 180
MIN_GAP = 60                     # 异常事件最小间隔（秒），放宽以保留更多事件

OUTPUT_NPZ = "processed_data_v2.npz"
LABEL_MAP_FILE = "label_map_v2.json"
RANDOM_SEED = 42

pd.set_option('future.no_silent_downcasting', True)

# ==================== 工具函数 ====================

def parse_case_folder_name(folder_name):
    """从案例文件夹名中提取 {机车号(带AB): 故障描述} 字典。"""
    match = re.search(r'问题[：:](.*)', folder_name)
    if not match:
        return {}
    problem_str = match.group(1)
    items = re.findall(r'(\d+[AB])([^,，、]+?)(?=[,，、]|$)', problem_str)
    mapping = {}
    if items:
        for loco, desc in items:
            mapping[loco.strip()] = desc.strip()
        return mapping

    # 如果问题描述中无机车号，从“编组：”部分获取
    group_match = re.search(r'编组[：:](.*?)(?:\s|$)', folder_name)
    if group_match:
        group_str = group_match.group(1)
        locos = re.findall(r'\d+[AB]', group_str)
        if len(locos) == 1:
            mapping[locos[0]] = problem_str.strip()
    return mapping

def map_fault_type(desc):
    if 'SIM卡' in desc or '读卡' in desc or '登网失败' in desc or '入网失败' in desc or '模块故障' in desc:
        return '网络接入故障'
    if '异常开机' in desc or '自愈重启' in desc or ('串口' in desc and '中断' in desc):
        return '设备自愈重启'
    if '网络状态异常' in desc:
        return '网络状态异常'
    if '空口无数据' in desc or '未收到主车' in desc or '未收到从车' in desc:
        return '空口无数据'
    if '正常' in desc:
        return '正常'
    if '网络单通' in desc:
        return '网络单通故障'
    return f'未分类({desc[:20]})'


def clean_df(df):
    """清洗单个DataFrame，返回 (原始df, 纯数值特征DataFrame)"""
    df = df.copy()

    if '记录时间' in df.columns:
        df['记录时间'] = pd.to_datetime(df['记录时间'], errors='coerce')
        if df['记录时间'].isna().all():
            df['记录时间'] = pd.date_range('2025-01-01', periods=len(df), freq='s')
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
    return df, df_features


def get_anomaly_flags(df_features):
    """返回每个时间点的异常标志（布尔Series）"""
    flags = (
        (df_features['附着状态'] == 0) |
        (df_features['连接状态'] == 0) |
        (df_features['信号强度'].isin([0, 99])) |
        (df_features.get('OCE串口', pd.Series([1]*len(df_features))) == 0) |
        (df_features.get('CIR串口', pd.Series([1]*len(df_features))) == 0) |
        (df_features.get('AB端网口', pd.Series([1]*len(df_features))) == 0) |
        (df_features.get('AT串口', pd.Series([1]*len(df_features))) == 0) |
        (df_features.get('空口数据接收状态', pd.Series([1]*len(df_features))) == 0)
    )
    return flags


def find_anomaly_events(df_features):
    """返回异常事件上升沿索引列表（去重后）"""
    anomaly_flags = get_anomaly_flags(df_features)
    prev = anomaly_flags.shift(1, fill_value=False)
    rise = anomaly_flags & (~prev)
    event_indices = df_features.index[rise].tolist()

    filtered = []
    last_idx = -MIN_GAP
    for idx in event_indices:
        if idx - last_idx >= MIN_GAP:
            filtered.append(idx)
            last_idx = idx
    return filtered


def slice_window(df_features, center_idx):
    """以 center_idx 为中心截取窗口"""
    start = center_idx - PRE_SEC
    end = center_idx + POST_SEC
    if start < 0 or end > len(df_features):
        return None
    return df_features.iloc[start:end].values.astype(np.float32)


# ==================== 主流程 ====================

def main():
    file_label_pairs = []

    # 第一步：收集文件及标签（同上一版）
    for root, dirs, files in os.walk(BASE_DIR):
        for f in files:
            if not f.endswith('_labeled.xlsx'):
                continue
            file_path = os.path.join(root, f)

            parts = root.split(os.sep)
            case_folder = None
            for i in range(len(parts)-1, -1, -1):
                if '问题：' in parts[i] or '问题:' in parts[i]:
                    case_folder = parts[i]
                    break
            if case_folder is None:
                continue

            loco_subdir = os.path.basename(root)
            if loco_subdir == case_folder:
                loco_fault_map = parse_case_folder_name(case_folder)
                if not loco_fault_map or len(loco_fault_map) != 1:
                    continue
                loco_id, matched_fault_desc = next(iter(loco_fault_map.items()))
                fault_type = map_fault_type(matched_fault_desc)
                file_label_pairs.append((file_path, fault_type))
                continue

            loco_num_match = re.match(r'^(\d+)', loco_subdir)
            if not loco_num_match:
                continue
            loco_num = loco_num_match.group(1)
            loco_fault_map = parse_case_folder_name(case_folder)
            if not loco_fault_map:
                continue
            matched_fault_desc = None
            for loco_id, fault_desc in loco_fault_map.items():
                if loco_id.startswith(loco_num):
                    matched_fault_desc = fault_desc
                    break
            if matched_fault_desc is None:
                continue
            fault_type = map_fault_type(matched_fault_desc)
            file_label_pairs.append((file_path, fault_type))

    print(f"共找到 {len(file_label_pairs)} 个标签化的日志文件。")
    if len(file_label_pairs) == 0:
        print("未找到任何数据，请检查 BASE_DIR 路径。")
        return

    # 建立标签映射（确保包含“正常”类）
    unique_labels = sorted(list(set(label for _, label in file_label_pairs)))
    if '正常' not in unique_labels:
        unique_labels.append('正常')
    label2idx = {lab: i for i, lab in enumerate(unique_labels)}
    print("故障类别映射:", label2idx)

    all_windows = []
    all_labels = []

    for file_path, label in file_label_pairs:
        try:
            print(f"正在处理: {file_path}")
            df = pd.read_excel(file_path)
        except Exception as e:
            print(f"读取文件失败 {file_path}: {e}")
            continue

        try:
            df, df_feat = clean_df(df)
        except Exception as e:
            print(f"清洗文件失败 {file_path}: {e}")
            continue

        anomaly_flags = get_anomaly_flags(df_feat)
        events = find_anomaly_events(df_feat)

        # ----- 生成故障窗口（事件触发 + 重叠采样） -----
        if events:
            for center in events:
                # 原始窗口
                window = slice_window(df_feat, center)
                if window is not None:
                    all_windows.append(window)
                    all_labels.append(label2idx[label])
                # 重叠采样：在异常持续期内，每隔 120 秒再取一个窗口
                for offset in range(120, POST_SEC, 120):
                    window = slice_window(df_feat, center + offset)
                    if window is not None:
                        all_windows.append(window)
                        all_labels.append(label2idx[label])

            # 额外：在该文件的安全正常区段（例如前 60 秒之前），取 2 个正常窗口
            normal_centers = [
                PRE_SEC + 10,                        # 文件开头附近
                len(df_feat) - POST_SEC - 10        # 文件结尾附近
            ]
            for nc in normal_centers:
                if nc < PRE_SEC or nc >= len(df_feat) - POST_SEC:
                    continue
                # 确认窗口内无异常
                start = nc - PRE_SEC
                end = nc + POST_SEC
                if not anomaly_flags.iloc[start:end].any():
                    window = slice_window(df_feat, nc)
                    if window is not None:
                        all_windows.append(window)
                        all_labels.append(label2idx['正常'])

        else:
            # 整个文件无异常，全部当作正常，取 5 个窗口
            mid = len(df_feat) // 2
            for offset in [-2*WINDOW_SIZE, -WINDOW_SIZE, 0, WINDOW_SIZE, 2*WINDOW_SIZE]:
                nc = mid + offset
                window = slice_window(df_feat, nc)
                if window is not None:
                    all_windows.append(window)
                    all_labels.append(label2idx['正常'])

    print(f"成功生成 {len(all_windows)} 个窗口样本。")
    if len(all_windows) == 0:
        print("未生成任何窗口样本。")
        return

    X = np.array(all_windows)
    y = np.array(all_labels)
    print(f"特征矩阵形状: {X.shape}, 标签形状: {y.shape}")

    # 划分数据集
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=RANDOM_SEED
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_SEED
        )
    except ValueError as e:
        print(f"分层划分失败 ({e})，退化为普通随机划分。")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=RANDOM_SEED
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED
        )

    # 保存
    out_npz = os.path.join(SCRIPT_DIR, "data.npz")
    out_json = os.path.join(SCRIPT_DIR, "label_map.json")

    np.savez_compressed(out_npz,
                        X_train=X_train, y_train=y_train,
                        X_val=X_val, y_val=y_val,
                        X_test=X_test, y_test=y_test)
    print(f"数据已保存至 {out_npz}")

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(label2idx, f, ensure_ascii=False, indent=2)
    print(f"标签映射已保存至 {out_json}")


if __name__ == "__main__":
    main()