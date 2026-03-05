
import sys
from PyQt5.QtCore import Qt,QBuffer,QByteArray
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog,QMessageBox
from PyQt5.QtGui import QPixmap
from face_ai import Ui_Form
from ThreadClass import cameraThread,MySqlThread
from datetime import datetime


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)   # 将UI设置到当前窗口
        self.Facedatabase_thread = MySqlThread()
        self.Facedatabase_thread.start()
        self.cam_thread = cameraThread()#创建摄像头线程
        #连接摄像头信号到更新标签的槽函数
        self.cam_thread.send_result_img.connect(self.update_camera)
        # 在这里连接按钮信号到槽函数
        #界面1:人脸录入
        self.ui.pushButton_4.clicked.connect(self.on_face_entry)
        #界面2:人脸识别
        self.ui.pushButton_3.clicked.connect(self.face_Facialrecognition)
        #界面3:数据库管理
        self.ui.pushButton_2.clicked.connect(self.sqlite)
        #界面4:退出
        self.ui.pushButton.clicked.connect(self.exit)

        #界面1-功能1:上传文件
        self.ui.pushButton_5.clicked.connect(self.upload_image)

        #界面1:功能:开始摄像头
        self.ui.pushButton_6.clicked.connect(self.start_camera)
        #界面1:功能:关闭摄像头
        self.ui.pushButton_7.clicked.connect(self.stop_camera)
        #界面1:功能:拍照摄像头
        self.ui.pushButton_8.clicked.connect(self.capture_photo)
        #界面1:功能:保存信息
        self.ui.pushButton_9.clicked.connect(self.save_info)

        self.captured_pixmap = None  # 存储拍照的 QPixmap
        self.captured_features = []  # 存储对应的人脸特征

        print("label_3 是否存在:", hasattr(self.ui, 'label_3'))
        print("label_3 尺寸:", self.ui.label_3.size())

        # ... 其他按钮连接

    def on_face_entry(self):
        print("人脸录入按钮被点击")
        # 切换到对应页面（假设 stackedWidget 的页面索引0是人脸录入）
        self.ui.stackedWidget.setCurrentIndex(0)

    def face_Facialrecognition(self):
        print("人脸识别按钮被点击")
        self.ui.stackedWidget.setCurrentIndex(1)
    def sqlite(self):
        print("数据库管理按钮被点击")
        self.ui.stackedWidget.setCurrentIndex(2)


    def exit(self):
        print("退出按钮被点击")
        self.close()

    #界面1-功能1:上传图片
    def upload_image(self):
        file_path,_ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if file_path:#选择了文件
            pixmap = QPixmap(file_path)
            if pixmap.isNull():
                QMessageBox.warning(self,"提示","无法加载图片,文件可能已损坏或者格式不支持")
                return

            self.ui.label_3.setPixmap(pixmap)
            self.ui.lineEdit.setText(file_path)

        else:#用户取消操作
            pass


    #摄像头画面实时传输到label_3
    def update_camera(self,qimage):
        print("收到图像")
        pixmap = QPixmap.fromImage(qimage)
        scaled = pixmap.scaled(self.ui.label_3.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation)
        self.ui.label_3.setPixmap(scaled)

    #开启摄像头
    def start_camera(self):
        self.cam_thread.start()
    #关闭摄像头
    def stop_camera(self):
        self.cam_thread.stop()

    def save_info(self):
        if self.captured_pixmap is None:
            QMessageBox.warning(self,"提示","请先拍照")
            return

        # 从ui读取输入信息
        name = self.ui.lineEdit_3.text().strip()
        age = self.ui.lineEdit_2.text().strip()
        student_id = self.ui.lineEdit_4.text().strip()

        if self.ui.radioButton_2.isChecked():
            sex = "男"
        elif self.ui.radioButton.isChecked():
            sex = "女"
        else:
            sex = "未知"

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 将QImage转换为二进制数据
        byte_array = QByteArray()
        buffer = QBuffer(byte_array)
        buffer.open(QBuffer.WriteOnly)
        self.captured_pixmap.save(buffer, "JPG")
        image_data = byte_array.data()

        # 取第一个人脸
        if self.captured_features and len(self.captured_features) > 0:
            face_feature_bytes = self.captured_features[0].tobytes()
        else:
            face_feature_bytes = b''

        content = {
            "name": name,
            "sex": sex,
            "age": int(age) if age.isdigit() else 0,
            "studentID": student_id,
            "time": current_time,
            "img": image_data,
            "facefeature": face_feature_bytes

        }

        # 发送到数据库
        self.Facedatabase_thread.q_deal_sql_cmd.put({"cmd": MySqlThread.INSERT_NEW, "content": content})
        QMessageBox.information(self, "提示", "信息已提交到数据库")


    def capture_photo(self):
        # 从摄像头线程获取最新帧和特征
        qimage, features = self.cam_thread.get_latest_data()
        if qimage is None:
            QMessageBox.warning(self, "提示", "没有画面，请先开启摄像头")
            return


        #点击拍照按钮后把得到的图片传入到右侧label中显示
        # 从 QImage 创建 QPixmap 并存储
        pixmap = QPixmap.fromImage(qimage)
        self.captured_pixmap = pixmap
        self.captured_features = features

        # 保存图片文件
        pixmap.save("capture_photo.jpg")
        # 显示到右侧 label_8
        scaled = pixmap.scaled(self.ui.label_8.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.ui.label_8.setPixmap(scaled)
        print("拍照成功，图片已保存为 capture_photo.jpg")
        QMessageBox.information(self, "提示", "拍照完成")







if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())