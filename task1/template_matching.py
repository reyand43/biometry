import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QGridLayout, QHBoxLayout, QWidget, QPushButton, QVBoxLayout
from PyQt6.QtGui import QPixmap



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("My App")
        
        self.globalLayout = QGridLayout()
        self.globalLayout.setSpacing(10)

        photo = QLabel('Enter photo path')
        photoInput = QLineEdit()
        photoButton = QPushButton("Add photo")
        photoButton.clicked.connect(lambda: self.upload_photo(photoInput.text()))

        self.globalLayout.addWidget(photo, 1, 0)
        self.globalLayout.addWidget(photoInput, 1, 1)
        self.globalLayout.addWidget(photoButton, 1, 4)
        
        self.setFixedSize(QSize(1000, 700))
        container = QWidget()
        container.setLayout(self.globalLayout)

        self.setCentralWidget(container)

    def upload_photo(self, path):
        lbl = QLabel(self)
        pixmap = QPixmap(path)
        lbl.setPixmap(pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio))
        vLayout = QVBoxLayout()
        findButton = QPushButton("Find face")
        findButton.clicked.connect(lambda: self.findFace(path))
        findButton2 = QPushButton("Viola-Jones")
        findButton2.clicked.connect(lambda: self.viola(path))
        vLayout.addWidget(findButton)
        vLayout.addWidget(findButton2)
        widget = QWidget()
        widget.setLayout(vLayout)
        self.globalLayout.addWidget(lbl, 2, 0)
        self.globalLayout.addWidget(widget, 2, 1)


    def findFace(self, path) -> None:
        template = cv2.imread('template1.jpg', 0)
        img = cv2.imread(path)
        self.rgb_template_matching(template, img)

    def rgb_template_matching(self, template, image) -> None:
        methods = [
          'cv2.TM_SQDIFF',
          'cv2.TM_CCORR',
          'cv2.TM_CCOEFF',
        ]

        i = 131

        for meth in methods:
            img = image.copy()
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            w, h = template.shape[::-1]
            method = eval(meth)
            res = cv2.matchTemplate(img_gray,template,method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(img, top_left, bottom_right, 255, 2)
            plt.subplot(i),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.subplot(i),plt.title(meth)
            plt.xticks([]), plt.yticks([])
            i += 1
        plt.show()

    def viola(self, path):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
        cv2.imwrite("detected_" + path, img)
        plt.subplot(121),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.xticks([]), plt.yticks([])
        plt.show()
          

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()

if __name__ == '__main__':
    main()