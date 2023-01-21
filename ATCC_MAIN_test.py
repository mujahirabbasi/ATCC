from PyQt4 import QtGui, QtCore
import cv2
import sys
import csv
from multiprocessing import Process, Queue, Pipe, Value, Array
from threading import Thread
from ATCC import *
import MySQLdb
from sshtunnel import SSHTunnelForwarder
from uuid import getnode

class Emitter(QtCore.QObject, Thread):

    def __init__(self, transport, parent=None):
        QtCore.QObject.__init__(self,parent)
        Thread.__init__(self)
        self.transport = transport
        self.daemon = True

    def _emit(self, signature, args=None):
        if args:
            self.emit(QtCore.SIGNAL(signature), args)
        else:
            self.emit(QtCore.SIGNAL(signature))

    def run(self):
        while True:
            try:
                signature = self.transport.recv()
            except EOFError:
                break
            else:
                self._emit(*signature)

class ATCCProc(Process):

    def __init__(self, transport, queue, showcase, link, daemon=True):
        Process.__init__(self)
        self.link = link
        self.daemon = False
        self.transport = transport
        self.data_to_mother = queue
        self.showcase = showcase
        self.jId = 1
        if 'Mayo' in self.link: self.jId = 2

    def emit_to_mother(self, signature, args=None):
        signature = (signature, )
        if args:
            signature += (args, )
        self.transport.send(signature)

    def run(self):
        infile, jId, speed, classes = self.link, self.jId, 3, ['SMV', 'HMV']
        cap = cv2.VideoCapture(infile)
        ret, frame = cap.read()
        if ret:
            if frame.shape[1] > 1300:
                dim = (16*70,9*70)
                if jId in [5,6,7]: dim = (4*240,3*240)
            else:
                dim = (frame.shape[1],frame.shape[0])
            
            ATCC_counter = AutoCounter(frame,0,classes=classes)
        else:
            print 'Cannot read Video Capture: Check Link'
            cap.release()
        
        stream = Value('i', 0)
        xywh = Array('i', [0,0,0,0])
        capBuff = Queue()
        detBuff = Queue()
        outBuff = Queue()
            
        procs = []
        procs.append(Process(target=readCapture, args=(infile, speed, stream, capBuff)))
        procs.append(Process(target=preProcess, args=(xywh, dim, stream, capBuff, detBuff)))
        procs.append(Process(target=detect, args=(xywh, stream, detBuff, outBuff)))
        
        for p in procs:
            p.start()
            
        while stream.value == 0: continue
        print 'x'
        xywh[2] = abs(ATCC_counter.DElements[0][1][0] - ATCC_counter.DElements[0][0][0])
        xywh[0] = min(ATCC_counter.DElements[0][0][0],ATCC_counter.DElements[0][1][0])
        xywh[3] = abs(ATCC_counter.DElements[0][1][1] - ATCC_counter.DElements[0][0][1])
        xywh[1] = min(ATCC_counter.DElements[0][0][1],ATCC_counter.DElements[0][1][1])
        while stream.value >= 1:
            try:
                frame, outputs = outBuff.get(timeout=0.1)
                ATCC_counter.processNextFrame(frame, outputs)
                img = ATCC_counter.drawElements(frame.copy())
                img = ATCC_counter.displayCount(img)
                if self.showcase.value == 1:
                    self.data_to_mother.put((True, img))
                    self.emit_to_mother('data(PyQt_PyObject)', QtCore.QString("test"))
                if self.showcase.value == -1:
                    stream.value = 0
                    break
            except:
                if stream.value == 4:
                    stream.value = 5
        while True:
            try:
                ret, frame = self.data_to_mother.get(timeout=0.1)
            except:
                break
#        for p in procs:
#            p.join()
        #===============================================================
        
#        cap = cv2.VideoCapture(self.link)
#        ret, frame = cap.read()
#        tic = time.time()
#        while ret:
#            if time.time() - tic < 0.01: continue
#            if self.data_to_mother.qsize() > 10:
#                continue
#            if self.showcase.value == 1:
#                self.data_to_mother.put((ret, frame))
#                self.emit_to_mother('data(PyQt_PyObject)', QtCore.QString("test"))
#            ret, frame = cap.read()
#            if self.showcase.value == -1: break
#            tic = time.time()
#        cap.release()
#        while True:
#            try:
#                ret, frame = self.data_to_mother.get(timeout=0.1)
#            except:
#                break
                    
class ATCCMainWindow(QtGui.QMainWindow):
    
    def __init__(self):
        super(ATCCMainWindow, self).__init__()
        
        self.initUI()
        
    def initUI(self):
        #Geometry
        self.setGeometry(100, 100, 1020, 550)
        self.setFixedSize(1150, 570)
        self.setWindowFlags(QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint)
        
        #Title
        self.setWindowTitle('AiSee: ATCC Solution (Client Application)')
        
        #Menubar
        self.initMenu()
        
        #CentralWidget
        cwid = QtGui.QWidget(self)
        self.setCentralWidget(cwid)
        
        cameras = CameraRoster(cwid)
        display = DisplayWidget(cwid)
        
        grid = QtGui.QGridLayout(cwid)
        grid.setSpacing(10)
        grid.addWidget(QtGui.QLabel('Camera List',cwid),0,0,1,1)
        grid.addWidget(QtGui.QLabel('Displaying Result for: ',cwid),0,1,1,1)
        grid.addWidget(cameras, 1, 0, 1, 1)
        grid.addWidget(display, 1, 1, 1, 1)
        grid.setColumnStretch(0, 35)
        grid.setColumnStretch(1, 100)
        cwid.setLayout(grid)
        
        #StatusBar
        self.statusBar().showMessage('Ready')
        
        self.center()
        self.show()
        
    def initMenu(self):
        menubar = self.menuBar()
    
    def center(self):
        ag = QtGui.QDesktopWidget().availableGeometry()
        self.resize(ag.width()/1.5, ag.height()/1.5)
        qr = self.frameGeometry()
        qr.moveCenter(ag.center())
        self.move(qr.topLeft())
    
class CameraRoster(QtGui.QScrollArea):
    def __init__(self, parent):
        super(CameraRoster, self).__init__(parent)
        self.n =1
        self.initList()
        
    def initList(self):
        camcsv = csv.reader(open('camList.csv','rb'))
        next(camcsv, None)
        wid = QtGui.QWidget(self)
        vbox = QtGui.QVBoxLayout(wid)
        idx = 1
        for row in camcsv:
            try:
                name = row[1].split('.')
                name = row[0]+' ('+'.'.join(name[2:])+')'
                vbox.addWidget(Camera(wid, idx, name, row[2]))
                idx += 1
            except:
                continue
        wid.setLayout(vbox)
        self.setWidget(wid)
        
class Camera(QtGui.QWidget):
    def __init__(self, parent, idx, name, link):
        super(Camera, self).__init__(parent)
        self.link = link
        self.idx = idx
        
        hbox = QtGui.QHBoxLayout(self)
        hbox.setSpacing(10)
        hbox.setMargin(0)
        label = CameraLabel(self, name)
        activate = QtGui.QCheckBox(self)
        hbox.addWidget(label)
        hbox.addWidget(activate)
        hbox.setStretch(0,90)
        hbox.setStretch(1,10)
        self.setLayout(hbox)
        
        self.isShowcasing = Value('i', 0)
        activate.stateChanged.connect(self.togglePlay)
        
    def togglePlay(self):
        label = self.findChildren(CameraLabel)[0]
        state = self.findChildren(QtGui.QCheckBox)[0].checkState()
        if state > 0:
            label.toggleActive()
            mother_pipe, child_pipe = Pipe()
            queue = Queue()
            self.data_from_child = queue
            emitter = Emitter(mother_pipe)
            ATCCProc(child_pipe, queue, self.isShowcasing, self.link).start()
            emitter.start()
            self.connect(emitter,QtCore.SIGNAL('data(PyQt_PyObject)'), self.getNextFrame)
        else:
            label.toggleActive()
            self.isShowcasing.value = -1
            
    def getNextFrame(self, text):
        label = self.findChildren(CameraLabel)[0]
        state = self.findChildren(QtGui.QCheckBox)[0].checkState()
        if state > 0:
            try:
                ret, frame = self.data_from_child.get()
            except:
                ret = False
            if ret and label.isActive:
                display = self.parent().parent().parent().parent().findChildren(DisplayWidget)[0]
                display.setDisplay(frame)
    
    def mousePressEvent(self, event):
        event.button()

class CameraLabel(QtGui.QLabel):
    def __init__(self, parent, name):
        if len(name) <= 20:
            super(CameraLabel, self).__init__(name, parent)
        else:
            super(CameraLabel, self).__init__(name[0:17]+'...', parent)
        self.setToolTip(name)
        self.setWordWrap(True)
        self.isActive = False
        self.setFixedSize(160,30)
        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.setLineWidth(3)
        
    def mousePressEvent(self, event):
        if event.button() == 1 and not self.isActive:
            self.toggleActive()
            
    def toggleActive(self):
        self.isActive = not self.isActive
        checkBox = self.parent().findChildren(QtGui.QCheckBox)[0]
        if checkBox.checkState() == 0: self.isActive = False
        if self.isActive:
            self.parent().isShowcasing.value = 1
            self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
            self.setLineWidth(3)
            cameras = self.parent().parent().findChildren(Camera)
            for cam in cameras:
                label = cam.findChildren(CameraLabel)[0]
                if cam.idx is not self.parent().idx and label.isActive:
                    label.toggleActive()
        else:
            self.parent().isShowcasing.value = 0
            self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
            self.setLineWidth(3)
        
class DisplayWidget(QtGui.QLabel):
    def __init__(self, parent):
        super(DisplayWidget, self).__init__(parent)
        self.setFixedSize(890,510)
        img = cv2.imread("AiSEE_ATCC.jpg")
        self.setDisplay(img)
        
    def setDisplay(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sz = self.size()
        w, h = sz.width(), sz.height()
        img = cv2.resize(img, (w,h))
        self.setPixmap(QtGui.QPixmap(QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format_RGB888)))
        
def main():
    app = QtGui.QApplication(sys.argv)
    atcc_gui = ATCCMainWindow()
    sys.exit(app.exec_())
        
if __name__ == '__main__':
    main()
