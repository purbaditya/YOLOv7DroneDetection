import time
import cv2
import sys
import pyqtgraph as pg
import numpy as np

pg.setConfigOptions(imageAxisOrder='row-major')
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QPushButton,QGridLayout,QCheckBox,QWidget,QApplication,
                             QMainWindow,QComboBox,QLabel,QLineEdit,QFrame,QSlider,
                             QGroupBox,QFileDialog,QSpinBox,QRadioButton,QMessageBox)
from PyQt5 import QtGui
from threading import Thread
import multiprocessing as mp
import configparser
#print(cv2.getBuildInformation())

# gst-launch-1.0.exe -v mfvideosrc device-index=0 ! queue ! videoconvert ! queue ! x264enc tune=zerolatency ! rtph264pay ! udpsink host = 10.246.58.64 port=5001 sync=false
# gst-launch-1.0.exe -v mfvideosrc device-index=0 ! queue ! videoconvert n-threads = 8 ! video/x-raw,format=I420 ! queue ! x264enc tune=zerolatency ! rtph264pay ! udpsink host = 10.246.58.64 port=5001 sync=false
# mfvideosrc device-index=0 ! videoconvert n-threads = 8 ! video/x-raw,format=I420,width=720,height=480 ! x264enc tune=zerolatency ! rtph264pay ! udpsink host = 10.246.58.64 port=5001 sync=false
# gst-launch-1.0 -v rtspsrc location=rtsp://127.0.0.1:8554/test latency=0 ! decodebin ! autovideosink

backsub = cv2.createBackgroundSubtractorKNN()
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

        self.setGeometry(300, 300, 250, 100)
        self.setWindowTitle('Receiver')
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.show()

        #self.proc = None
        self.cap = None
        self.thread = None


    def groupStreaming(self):
        groupBox = QGroupBox('Reception of Stream')
        self.btn_stream = QPushButton('Start')
        self.btn_stop = QPushButton('Stop')
        self.btn_stop.setEnabled(False)
        self.edit_port = QLineEdit(self)
        self.edit_port.setText("5000")
        self.label_port = QLabel('Port Number:', self)
        self.edit_jitterlatency = QLineEdit(self)
        self.edit_jitterlatency.setText("10")
        self.label_jitterlatency = QLabel('Jitter:', self)
        self.edit_res = QComboBox(self)
        self.edit_res.addItems(['SD','HD','FHD','QHD'])
        self.edit_res.setCurrentIndex(0)
        self.label_res = QLabel('Resolution:', self)
        #self.label_rtsp = QLabel('RTSP:', self)
        self.rtsp = QCheckBox(self)

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
        grid.addWidget(self.label_jitterlatency,*(3,0),*(1,2))
        grid.addWidget(self.edit_jitterlatency,*(3,2),*(1,4))
        grid.addWidget(self.label_res,*(4,0),*(1,3))
        grid.addWidget(self.edit_res,*(4,3),*(1,3))
        #grid.addWidget(self.label_rtsp,*(5,0),*(1,3))
        #grid.addWidget(self.rtsp,*(5,3),*(1,3))
        groupBox.setLayout(grid)
        
        return groupBox

    def btn_stream_clicked(self):
        feature = 0
        global stop_threads
        if self.btn_stream.isEnabled() == True:
            self.btn_stream.setEnabled(False)
            #self.btn_stop.setEnabled(True)

        stop_threads = False
        self.thread = Thread(target=self.start_capture, args=(self.edit_port.text(),self.edit_jitterlatency.text(),self.edit_res.currentIndex()))
        self.thread.daemon = True
        self.thread.start()

    def btn_stop_clicked(self):
        feature = 0
        global stop_threads
        if self.btn_stop.isEnabled() == True:
            self.btn_stop.setEnabled(False)
            self.btn_stream.setEnabled(True)
            stop_threads = True
            self.thread.join()
            self.cap.release()

    def closeEvent(self, event):
        global stop_threads
        close = QMessageBox(text='Quitting...')
        #close.setText("You sure?")
        #close.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        if self.thread != None:
            stop_threads = True
            self.thread.join()
        if self.cap != None:
            self.cap.release()
        close = close.exec()

    def start_capture(self,port,latency,resolution):
        '''
        if self.rtsp.isChecked():
            pipeline = "rtsp://127.0.0.1:8554/test"
        else:
            pipeline = "udpsrc port="+port+" ! application/x-rtp,payload=96,encoding-name=H264 ! rtpjitterbuffer mode=1 latency="+latency+" ! rtph264depay ! h264parse ! decodebin ! videoconvert ! appsink"
        '''
        pipeline = "udpsrc port="+port+" ! application/x-rtp,payload=96,encoding-name=H264 ! rtpjitterbuffer mode=1 latency="+latency+" ! rtph264depay ! h264parse ! decodebin ! videoconvert ! appsink"
        print('Waiting for packets...')

        if resolution == 0:
            w,h  = 640, 480
        elif resolution == 1:
            w, h = 1280, 720
        elif resolution == 2:
            w, h = 1920, 1080
        else:
            w, h = 2560, 1440

        self.cap = cv2.VideoCapture(pipeline)
        #self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            print('Error: Unable to open pipeline')
            exit()

        self.btn_stop.setEnabled(True)

        while True:
            global stop_threads
            if stop_threads:
                break
            # Read a frame from the pipeline
            print('Reciving packets ...')
            ret, frame = self.cap.read()
            if not ret:
                print('Error: Unable to read frame')
                break
            if ret:
                frame_proc = cv2.resize(frame,(w,h),interpolation = cv2.INTER_AREA)
                newframe = frame_proc #cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)

                # Display the frame
                cv2.imshow('Frame', newframe)
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
