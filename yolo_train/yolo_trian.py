from ultralytics import YOLO
import torch
model = YOLO('yolo11n.pt')

if __name__ == '__main__':
    model.train(data=r"D:\huaqing\2207biji\class\Facenet\dataset\mydata.yaml",epochs=50)
'''
from ultralytics import YOLO

model = YOLO('yolo11n.pt')

if __name__ == '__main__':
    model.train(
        data=r"D:\huaqing\2207biji\class\Facenet\dataset\mydata.yaml",
        epochs=50,
        device='cuda'      # 指定使用GPU，0表示第一块GPU，'cuda'自动选择可用GPU
    )'''

