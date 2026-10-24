# -*- coding: utf-8 -*-
"""
朔黄铁路数据路径统一管理

当前 PyCharm 项目结构（与你现有项目保持一致）：

ShuoHuang/
├── .git/
├── main.py
├── train_cnn.py
├── Demo/
│   └── Data_process.py
├── 第二批次导出/
│   ├── 2025年11月日志汇总 - 规范格式/
│   ├── 2025年12月日志汇总 - 规范格式/
│   └── ...
└── 本数据处理代码/
    ├── 00_paths.py
    ├── 01_clean_suhuang.py
    ├── 02_standardize_suhuang.py
    └── 03_visualize_suhuang.py

核心原则：
1. 不使用 PyCharm Working Directory 作为数据根目录；
2. 默认以“项目根目录/第二批次导出”为原始数据根目录；
3. 递归寻找 *_labeled.xlsx；
4. 可以通过 --input / --data-root 手动覆盖；
5. 输出路径统一放在项目根目录下，避免污染原始数据。
"""

from __future__ import annotations

# 用于寻找环境变量
import os
# 用于处理文件路径
from pathlib import Path

"""# 创建一个路径对象
path = Path("D:/python/ShuoHuang")
# 判断path是否存在
print(path.exists())
# 判断这个路径的是文件还是文件夹
print(path.is_file())
print(path.is_dir())
# 输出父级目录
print(path.parent)
# 输出当前目录名称
print(path.name)
# 输出当前目录绝对路径
print(path.resolve())"""


# -----------------------------------------------------------------------------
# 1. 找到项目根目录
# -----------------------------------------------------------------------------
# 这个文件通常放在 ShuoHuang/ 根目录。如果以后把代码放到子目录，
# 仍会沿父目录向上寻找 .git 或 main.py 等项目标志。
_THIS_DIR = Path(__file__).resolve().parent # D:\python\ShuoHuang\Data_process

# 找到项目根目录
# 箭头表示函数预计返回一个Path
def _find_project_root(start: Path) -> Path:
    # 用于确定项目根目录的特征文件
    markers = {".git", "main.py", "train_cnn.py", "predict.py"}

    # 从某个目录开始一层一层往上寻找
    # *是一个解包操作，可以把列表展开
    for candidate in [start, *start.parents]:
        try:
            # iterdir()查看目录里有哪些文件和文件夹
            names = {p.name for p in candidate.iterdir()}
            # print(names)
        except OSError:
            continue
        # .git是一个隐藏文件夹
        """
        如果当前目录里有 .git，或者当前目录至少同时包含两个项目标志文件，那么就认为它是项目根目录。
        """
        if ".git" in names or len(markers.intersection(names)) >= 2:
            return candidate

    # 找不到时，退化为当前脚本目录
    return start


PROJECT_ROOT = _find_project_root(_THIS_DIR)
# print(PROJECT_ROOT)

# -----------------------------------------------------------------------------
# 2. 默认数据目录 / 输出目录
# -----------------------------------------------------------------------------
DEFAULT_DATA_ROOT = PROJECT_ROOT / "第二批次导出"
DEFAULT_CLEAN_ROOT = PROJECT_ROOT / "01_清洗结果"
DEFAULT_STANDARD_ROOT = PROJECT_ROOT / "02_标准化结果"
DEFAULT_VIS_ROOT = PROJECT_ROOT / "03_可视化结果"

# 可以用环境变量临时切换数据位置，例如：
# Windows CMD:
#   set SUHUANG_DATA_ROOT=D:\数据\第二批次导出
# PowerShell:
#   $env:SUHUANG_DATA_ROOT = "D:\数据\第二批次导出"
# 没有就为空""
_env_data_root = os.getenv("SUHUANG_DATA_ROOT", "").strip()
DATA_ROOT = Path(_env_data_root).expanduser() if _env_data_root else DEFAULT_DATA_ROOT

# -----------------------------------------------------------------------------
# 3. 路径解析
# -----------------------------------------------------------------------------
def candidate_paths(path: str | Path) -> list[Path]:
    """为用户传入的相对路径构造多个候选位置。"""
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return [raw]

    candidates = [
        Path.cwd() / raw,
        PROJECT_ROOT / raw,
        DATA_ROOT / raw,
    ]

    # 去重，同时保持顺序
    result: list[Path] = []
    seen = set()
    for p in candidates:
        key = str(p.resolve(strict=False)).lower()
        if key not in seen:
            result.append(p)
            seen.add(key)
    return result


def resolve_existing_path(path: str | Path) -> Path:
    """解析一个已经存在的文件/目录。"""
    candidates = candidate_paths(path)
    for p in candidates:
        if p.exists():
            return p.resolve()

    raise FileNotFoundError(
        "找不到指定路径：\n"
        f"  输入：{path}\n"
        "已尝试：\n"
        + "\n".join(f"  - {p}" for p in candidates)
    )


def resolve_input_root(data_root: str | Path | None = None) -> Path:
    """解析原始数据根目录；未指定时默认使用项目/第二批次导出。"""
    if data_root is None or str(data_root).strip() == "":
        root = DATA_ROOT
    else:
        root = resolve_existing_path(data_root)

    if not root.exists():
        raise FileNotFoundError(
            f"原始数据目录不存在：{root}\n"
            "请检查项目中是否存在“第二批次导出”，或使用 --data-root 指定目录。"
        )
    return root.resolve()


# -----------------------------------------------------------------------------
# 4. 自动寻找原始数据文件
# -----------------------------------------------------------------------------
def find_labeled_files(
    root: str | Path | None = None,
    pattern: str = "*_labeled.xlsx",
    recursive: bool = True,
) -> list[Path]:
    """
    自动递归寻找标注后的朔黄铁路 Excel。

    默认示例：
    第二批次导出/**/**/**/*_labeled.xlsx

    也兼容 *_labeled(1).xlsx 这种测试副本，因为它同样满足“文件名包含
    _labeled”的语义筛选。
    """
    base = resolve_input_root(root)

    if base.is_file():
        return [base] if "_labeled" in base.stem.lower() else []

    iterator = base.rglob(pattern) if recursive else base.glob(pattern)
    files = [
        p.resolve()
        for p in iterator
        if p.is_file()
        and not p.name.startswith("~$")
        and "_labeled" in p.stem.lower()
    ]

    # 兼容带有 (1) / (2) 的副本，即使主 pattern 没命中也补充查找
    if recursive:
        for p in base.rglob("*.xlsx"):
            if p.is_file() and "_labeled" in p.stem.lower() and not p.name.startswith("~$"):
                files.append(p.resolve())

    return sorted(set(files), key=lambda x: str(x).lower())


def find_by_keyword(
    root: str | Path | None = None,
    keyword: str | None = None,
) -> list[Path]:
    """在自动发现的标注文件中按文件名关键词筛选。"""
    files = find_labeled_files(root)
    if not keyword:
        return files

    key = str(keyword).lower()
    return [p for p in files if key in p.name.lower()]


def find_single_file(
    file_name: str,
    root: str | Path | None = None,
) -> Path:
    """
    根据文件名寻找文件。

    例如：
    find_single_file("17. 25-11-12_08-42-47 to 25-11-12_11-42-45_labeled.xlsx")
    """
    raw = Path(file_name).expanduser()
    if raw.exists():
        return raw.resolve()

    base = resolve_input_root(root)
    matches = [
        p.resolve()
        for p in base.rglob("*.xlsx")
        if p.is_file() and p.name.lower() == raw.name.lower()
    ]

    if not matches:
        raise FileNotFoundError(
            f"在数据根目录中找不到文件：{file_name}\n数据根目录：{base}"
        )
    if len(matches) > 1:
        print("注意：发现多个同名文件，默认使用第一个：")
        for p in matches:
            print("  ", p)
    return sorted(matches, key=lambda x: str(x).lower())[0]


# -----------------------------------------------------------------------------
# 5. 输出目录
# -----------------------------------------------------------------------------
def get_output_root(kind: str) -> Path:
    mapping = {
        "clean": DEFAULT_CLEAN_ROOT,
        "standard": DEFAULT_STANDARD_ROOT,
        "visual": DEFAULT_VIS_ROOT,
    }
    if kind not in mapping:
        raise ValueError(f"未知输出类型：{kind}，可选：{list(mapping)}")
    path = mapping[kind]
    path.mkdir(parents=True, exist_ok=True)
    return path


def relative_output_path(source_file: Path, source_root: Path, output_root: Path) -> Path:
    """
    按原始数据目录结构生成输出路径。

    例如：
    原始：第二批次导出/2025年11月.../7058/xxx_labeled.xlsx
    输出：01_清洗结果/2025年11月.../7058/xxx_cleaned.xlsx

    这样即使不同案例里存在同名文件，也不会互相覆盖。
    """
    try:
        rel = source_file.resolve().relative_to(source_root.resolve())
    except ValueError:
        rel = Path(source_file.name)
    return output_root / rel


def print_data_info() -> None:
    """打印当前项目和数据目录信息。"""
    print("=" * 72)
    print("朔黄铁路数据路径信息")
    print("项目根目录：", PROJECT_ROOT)
    print("原始数据根目录：", DATA_ROOT)
    print("清洗结果根目录：", DEFAULT_CLEAN_ROOT)
    print("标准化结果根目录：", DEFAULT_STANDARD_ROOT)
    print("可视化结果根目录：", DEFAULT_VIS_ROOT)
    print("=" * 72)

    if DATA_ROOT.exists():
        files = find_labeled_files()
        print("自动发现 labeled Excel：", len(files), "个")
        if files:
            print("第一个文件：", files[0])
    else:
        print("警告：默认数据目录尚不存在，请检查“第二批次导出”位置。")


if __name__ == "__main__":
    print_data_info()
