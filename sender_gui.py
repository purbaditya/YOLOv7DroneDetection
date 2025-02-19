import time
import cv2
import sys
import pyqtgraph as pg

pg.setConfigOptions(imageAxisOrder='row-major')
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QPushButton,QGridLayout,QCheckBox,QWidget,QApplication,
                             QMainWindow,QComboBox,QLabel,QLineEdit,QFrame,QSlider,
                             QGroupBox,QFileDialog,QSpinBox,QRadioButton,QMessageBox)
from PyQt5 import QtGui
from threading import Thread
import multiprocessing as mp
import configparser
# print(cv2.getBuildInformation())

###### READ - SEND CAPTURE FROM USB DEVICE ##########
# gst-launch-1.0.exe -v mfvideosrc device-index=0 ! queue ! videoconvert ! queue ! x264enc tune=zerolatency ! rtph264pay ! udpsink host = 10.246.58.64 port=5001 sync=false
# gst-launch-1.0.exe -v mfvideosrc device-index=0 ! queue ! videoconvert n-threads = 8 ! video/x-raw,format=I420 ! queue ! x264enc tune=zerolatency ! rtph264pay ! udpsink host = 10.246.58.64 port=5001 sync=false
# mfvideosrc device-index=0 ! videoconvert n-threads = 8 ! video/x-raw,format=I420,width=1280,height=720 ! x264enc tune=zerolatency ! rtph264pay ! udpsink host = 10.246.58.64 port=5001 sync=false
global stop_threads

class GUI_view_controller(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        grid = QGridLayout()
        grid.addWidget(self.groupStreaming(),*(0,0),*(1,1))

        centralWidget = QFrame()
        centralWidget.setFrameShape(QFrame.StyledPanel)
        centralWidget.setFrameShadow(QFrame.Plain)
        centralWidget.setLayout(grid)
        self.setCentralWidget(centralWidget)

        self.setGeometry(400, 400, 350, 200)
        self.setWindowTitle('Sender')
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.show()

        #self.proc = None
        self.cap = None
        self.writer = None

    def groupStreaming(self):
        groupBox = QGroupBox('Sending of Stream')
        self.btn_stream = QPushButton('Start')
        self.btn_stop = QPushButton('Stop')
        self.btn_stop.setEnabled(False)
        self.edit_port = QLineEdit(self)
        self.edit_port.setText("5001")                         # default port to send to, forwarded in yolo docker
        self.label_port = QLabel('Port Number:', self)
        self.edit_ip = QLineEdit(self)
        self.edit_ip.setText('172.19.42.18')                   # ip to send stream to, where yolo docker is running
        self.label_ip = QLabel('IP:', self)
        self.edit_res = QComboBox(self)
        self.edit_res.addItems(['SD','HD','FHD','QHD'])
        self.edit_res.setCurrentIndex(0)
        self.label_res = QLabel('Resolution:', self)
        self.edit_src = QComboBox(self)
        self.edit_src.addItems(['USB:0','USB:1','USB:2'])
        self.edit_src.setCurrentIndex(0)
        self.label_src = QLabel('Source:', self)

        #self.label_rtsp = QLabel('RTSP:', self)
        #self.rtsp = QCheckBox(self)

        self.btn_stream.clicked.connect(self.btn_stream_clicked)
        self.btn_stop.clicked.connect(self.btn_stop_clicked)

        grid = QGridLayout()
        grid.setSpacing(10)
        grid.setColumnStretch(0,1)
        grid.setColumnStretch(1,1)
        grid.addWidget(self.btn_stream,*(0,0),*(1,6))
        grid.addWidget(self.btn_stop,*(1,0),*(1,6))
        grid.addWidget(self.label_port,*(2,0),*(1,2))
        grid.addWidget(self.edit_port,*(2,2),*(1,4))
        grid.addWidget(self.label_ip,*(3,0),*(1,2))
        grid.addWidget(self.edit_ip,*(3,2),*(1,4))
        grid.addWidget(self.label_res,*(4,0),*(1,3))
        grid.addWidget(self.edit_res,*(4,3),*(1,3))
        grid.addWidget(self.label_src,*(5,0),*(1,3))
        grid.addWidget(self.edit_src,*(5,3),*(1,3))
        groupBox.setLayout(grid)

        #grid.addWidget(self.label_rtsp,*(6,0),*(1,3))
        #grid.addWidget(self.rtsp,*(6,3),*(1,3))
        
        return groupBox

    def btn_stream_clicked(self):
        global stop_threads
        feature = 0
        usb_index = 1
        if self.btn_stream.isEnabled() == True:
            self.btn_stream.setEnabled(False)
            #self.btn_stop.setEnabled(True)

        stop_threads = False
        self.thread = Thread(target=self.start_capture, args=(self.edit_port.text(),self.edit_ip.text(),self.edit_res.currentIndex(),self.edit_src.currentIndex(),feature))
        self.thread.daemon = True
        self.thread.start()

    def btn_stop_clicked(self):
        global stop_threads
        feature = 0
        if self.btn_stop.isEnabled() == True:
            self.btn_stop.setEnabled(False)
            self.btn_stream.setEnabled(True)
            stop_threads = True
            self.thread.join()
            self.cap.release()

    def closeEvent(self, event):
        global stop_threads
        close = QMessageBox(text='Quitting...')
        if self.thread != None:
            stop_threads = True
            self.thread.join()
        if self.cap != None:
            self.cap.release()
        close = close.exec()

    def start_capture(self,port,ip,resolution,usb_index,feature):
        self.cap = cv2.VideoCapture(usb_index)
        if resolution == 0:
            w,h  = 640, 480
        elif resolution == 1:
            w, h = 1280, 720
        elif resolution == 2:
            w, h = 1920, 1080
        else:
            w, h = 2560, 1440

        while not self.cap.read()[0]:
            self.cap = cv2.VideoCapture(usb_index-1)
        if self.cap.read()[0]:
            fps_s,w_s,h_s  = round(self.cap.get(cv2.CAP_PROP_FPS)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        '''
        if self.rtsp.isChecked():
            pipeline = 'appsrc ! videoconvert ! videoscale ! video/x-raw ! x264enc speed-preset=veryfast tune=zerolatency bitrate=500 ! rtspclientsink location=rtsp://10.209.223.4:8554/test1'
        else:
            pipeline = 'appsrc ! videoconvert n-threads = 8 ! video/x-raw,format=I420 ! x264enc speed-preset=veryfast tune=zerolatency bitrate=1000 ! rtph264pay ! udpsink host=' + ip + ' port=' + port + ' sync=false'
        '''

        pipeline = 'appsrc ! videoconvert n-threads = 8 ! video/x-raw,format=I420 ! x264enc speed-preset=veryfast tune=zerolatency bitrate=1000 ! rtph264pay ! udpsink host=' + ip + ' port=' + port + ' sync=false'
        print('Waiting for packets...')
        #fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.writer = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, fps_s, (w,h), True)

        if not self.cap.isOpened():
            print('Error: Unable to open pipeline')
            exit()

        self.btn_stop.setEnabled(True)

        while self.cap.isOpened():
            global stop_threads
            if stop_threads:
                break
            # Read a frame from the pipeline
            print('Sending packets ... ',fps_s,w,h)
            ret, frame = self.cap.read()
            if not ret:
                print('Error: Unable to read frame')
                break
            if ret:
                if w > 640:
                    frame = cv2.resize(frame,(w,h),interpolation = cv2.INTER_AREA)
                self.writer.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        #self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()

    gui = GUI_view_controller()
    
    sys.exit(app.exec_())
