






#返回人脸位置
def getFacePosition(img:str):
    """
    得到人脸在图片中的坐标位置(行,列)
    四个点
    x0,x1表示第x0行到第x1行,
    y0,y1表示第y0列到第y1列
    (未检测到则返回None)
    """
    x0,y0,x1,y1 = 0,0,0,0
    return x0,y0,x1,y1




def facenet(img:str,pos:tuple):
    """
    人脸区域通过facenet毛线哦转换为512维向量,返回出来
    img: 图片路径
    pos: 人脸区域位置
    512维向量,np数组

    """
