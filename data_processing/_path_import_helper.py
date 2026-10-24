# -*- coding: utf-8 -*-
"""兼容 PyCharm 直接运行脚本时导入同目录 00_paths.py。"""
from __future__ import annotations

import importlib.util
from pathlib import Path


def import_path_module():
    here = Path(__file__).resolve().parent
    target = here / "00_paths.py"
    spec = importlib.util.spec_from_file_location("suhuang_paths", target)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载路径配置：{target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
