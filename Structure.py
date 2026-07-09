import os


def list_files(startpath, skip_dirs=None, skip_files=None):
    """
    列出项目结构并跳过指定文件夹和文件

    Args:
        startpath (str): 要扫描的根目录路径
        skip_dirs (list): 要跳过的文件夹名称列表
        skip_files (list): 要跳过的文件名称列表
    """
    if skip_dirs is None:
        skip_dirs = []
    if skip_files is None:
        skip_files = []

    # 确保跳过列表中的名称都是小写（用于不区分大小写的匹配）
    skip_dirs = [dir_name.lower() for dir_name in skip_dirs]
    skip_files = [file_name.lower() for file_name in skip_files]

    with open('project_structure.txt', 'w', encoding='utf-8') as f:
        for root, dirs, files in os.walk(startpath):
            # 从当前层级中移除要跳过的目录
            dirs[:] = [d for d in dirs if d.lower() not in skip_dirs]

            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * level

            # 写入当前目录
            f.write(f'{indent}{os.path.basename(root)}/\n')

            subindent = ' ' * 4 * (level + 1)
            for file in files:
                # 跳过指定的文件
                if file.lower() not in skip_files:
                    f.write(f'{subindent}{file}\n')


if __name__ == "__main__":
    path_to_project = "D:\python\ShuoHuang"  # 替换为你的项目路径

    skip_folders = ['.venv', 'config', '__pycache__', '.idea', 'static/imgs_old']
    skip_files = ['Structure.py', 'note.txt', 'project_structure.txt', 'ffmpeg_batch.py', 'merged_output.txt'
                  , 'CopyUtil.py', 'install_dependencies.py', 'practice.py', 'export_neo4j_data.py']  # 添加要跳过的文件

    list_files(path_to_project, skip_folders, skip_files)
    print("项目结构已导出到 project_structure.txt")