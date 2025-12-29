# simple_clean.py
import os
import shutil

# 要保留的文件名
keep_files = ['best_model_acc_85.98.pth', 'final_model.pth', 'README.md']

# 创建备份文件夹
os.makedirs('backup_models', exist_ok=True)

# 遍历并移动
for file in os.listdir('checkpoints'):
    if file not in keep_files:
        src = os.path.join('checkpoints', file)
        dst = os.path.join('backup_models', file)
        shutil.move(src, dst)
        print(f'移动: {file}')

print('清理完成！')
print(f'剩余文件: {os.listdir("checkpoints")}')