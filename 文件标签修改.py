import os
files = os.listdir(r"D:\huaqing\2207biji\class\Facenet\dataset\labels\valid")

#获取每个文件的绝对路径
files = [os.path.join(r"D:\huaqing\2207biji\class\Facenet\dataset\labels\valid",file) for file in files]

for file in files:
    with open(file,'r') as f:
        content = f.read()
        content = content.replace('18','0')
    with open(file,'w') as f:
        f.write(content)

print("完成")