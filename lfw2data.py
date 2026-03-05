import shutil
import os


src,dst = './lfw','./data'
os.makedirs(dst,exist_ok=True)

for name in os.listdir(src):
    folder = os.path.join(src,name)
    if os.path.join(folder):
        img = os.listdir(folder)[0]
        shutil.copy(os.path.join(folder,img),os.path.join(dst,img))
print("转换完成!所有图片已放到data文件夹")





