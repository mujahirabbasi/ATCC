from PyQt4 import QtGui, QtCore
import sys
import uuid
import MySQLdb
from sshtunnel import SSHTunnelForwarder
import cv2
import time
from ATCCThreads import *
import Queue

class ATCCMainWindow(QtGui.QMainWindow):
    
    def __init__(self):
        super(ATCCMainWindow, self).__init__()
        self.initUI()
        
    def initUI(self):
        #Geometry
        self.setGeometry(100, 100, 1150, 600)
        self.setFixedSize(1150, 600)
        self.setWindowFlags(QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint)
        
        #Title
        self.setWindowTitle('AiSee: ATCC Solution (Client Application)')
        
        #Menubar
        self.initMenu()
        
        #CentralWidget
        cwid = QtGui.QWidget(self)
        self.setCentralWidget(cwid)
        
        #Connection
        self.server = SSHTunnelForwarder(('139.59.86.80', 22),
                                     ssh_password="giri@123",
                                     ssh_username="root",
                                     remote_bind_address=('127.0.0.1', 3306))
        self.server.start()
        self.DBconn = MySQLdb.connect(host='127.0.0.1',
                               port=self.server.local_bind_port,
                               user='root',
                               passwd='ASSpeed123@',
                               db='ATCC')       
        
        display = DisplayWidget(cwid)                               
        cameras = CameraRoster(cwid)
        
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
        self.statusBar().showMessage('Please login to continue...')
        
        self.center()
        self.show()
        
        self.checkLogin()
        #self.timer = QtCore.QTimer(self)
        #self.timer.timeout.connect(self.pseudoQuery)
        #self.timer.start(30*1000)
        self.statusBar().showMessage('Welcome! Select Cameras to start, Switch between cameras to view')
    
    def pseudoQuery(self):
        cur = self.DBconn.cursor()
        cur.execute("select 1;")
        cur.close()
            
    def checkLogin(self):
        cur = self.DBconn.cursor()
        cur.execute("select ID, Login, Password from Machines;")# where MacID = "+str(uuid.getnode())+";")
        self.users = cur.fetchall();
        login = ATCCLogin(self.users, self)
        if login.exec_() != QtGui.QDialog.Accepted:
            self.DBconn.close()
            cur.close()
            self.server.stop()
            sys.exit()
        self.users = self.findChildren(ATCCLogin)[0].users
        try:
            cur.execute('update Machines set Status=1 where ID='+str(self.users[0])+';')
            self.DBconn.commit()
        except:
            self.DBconn.rollback()
        cur.close()
        
    def initMenu(self):
        menubar = self.menuBar()
    
    def center(self):
        ag = QtGui.QDesktopWidget().availableGeometry()
        self.resize(ag.width()/1.5, ag.height()/1.5)
        qr = self.frameGeometry()
        qr.moveCenter(ag.center())
        self.move(qr.topLeft())
    
    def closeEvent(self, event):
        reply = QtGui.QMessageBox.question(self, 'Alert!!!', "Are you sure to quit?", QtGui.QMessageBox.Yes | QtGui.QMessageBox.No,
                QtGui.QMessageBox.No)
        if reply == QtGui.QMessageBox.Yes:
            #self.timer.stop()
            cameras = self.findChildren(CameraRoster)[0].findChildren(QtGui.QWidget)[0].findChildren(Camera)
            for cam in cameras:
                cbox = cam.findChildren(QtGui.QCheckBox)[0]
                if cbox.checkState() > 0: cbox.setCheckState(0)
            cur = self.DBconn.cursor()
            try:
                cur.execute('update Machines set Status=0 where ID='+str(self.users[0])+';')
                self.DBconn.commit()
            except:
                self.DBconn.rollback()
            cur.close()
            self.DBconn.close()
            self.server.stop()
            event.accept()
            QtGui.QApplication.exit(0)
        else:
            event.ignore()
            
class CameraRoster(QtGui.QScrollArea):
    def __init__(self, parent):
        super(CameraRoster, self).__init__(parent)
        self.initList()
        
    def initList(self):
        cur = self.parent().parent().DBconn.cursor()
        cur.execute('select ID, IP, Description, BoxX1, BoxY1, BoxX2, BoxY2, LineX1, LineY1, LineX2, LineY2, RTSPlink, ProcSpeed from Cameras;')
        cameras = cur.fetchall()
        cur.close()
        
        wid = QtGui.QWidget(self)
        vbox = QtGui.QVBoxLayout(wid)
        for cam in cameras:
            vbox.addWidget(Camera(wid, cam))
        wid.setLayout(vbox)
        self.setWidget(wid)
        
class Camera(QtGui.QWidget):
    def __init__(self, parent, cam):
        super(Camera, self).__init__(parent)
        self.params = (cam[11], cam[12], [(cam[3],cam[4]),(cam[5],cam[6])], [(cam[7],cam[8]),(cam[9],cam[10])])
        self.idx = cam[0]
        
        name = cam[1].split('.')
        name = '('+'.'.join(name[2:])+') '+cam[2]
        
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
        
        self.isShowcasing = 0
        self.name = name
        activate.stateChanged.connect(self.togglePlay)
        
    def togglePlay(self):
        label = self.findChildren(CameraLabel)[0]
        state = self.findChildren(QtGui.QCheckBox)[0].checkState()
        label.toggleActive()
        if state > 0:
            self.start()
        else:
            self.stop()
        
    def start(self):
        cur = self.parent().parent().parent().parent().parent().DBconn.cursor()
        mid = self.parent().parent().parent().parent().parent().users[0]
        try:
            cur.execute('update Cameras set Status=1, MachineID='+str(mid)+' where ID='+str(self.idx)+';')
            self.parent().parent().parent().parent().parent().DBconn.commit()
        except:
            self.parent().parent().parent().parent().parent().DBconn.rollback()
        cur.close()
        
        cap_q, out_q, res_q = Queue.Queue(), Queue.Queue(), Queue.Queue()
        if self.idx < 6: gpu = 0
        else: gpu = 1
        self.pool = [ATCCThread(cap_q, res_q, self.params, gpu),
                     ReaderThread(cap_q, self.params)]
        
        self.res_q = res_q
        for thread in self.pool:
            thread.start()
        self.connect(self.pool[0],QtCore.SIGNAL('data(PyQt_PyObject)'), self.getNextFrame)

    def stop(self):
        self.isShowcasing = -1
        self.findChildren(QtGui.QCheckBox)[0].setCheckState(0)
        cur = self.parent().parent().parent().parent().parent().DBconn.cursor()
        mid = self.parent().parent().parent().parent().parent().users[0]
        try:
            #cur.execute("update Cameras set Status=0, LastSyncMachine="+str(uuid.getnode())+", LastSyncTime='"+time.strftime('%Y-%m-%d %H:%M:%S')+"' where ID="+str(self.idx)+";")
            cur.execute("update Cameras set Status=0, LastSyncMachine="+str(mid)+", LastSyncTime='"+time.strftime('%Y-%m-%d %H:%M:%S')+"' where ID="+str(self.idx)+";")
            self.parent().parent().parent().parent().parent().DBconn.commit()
        except:
            self.parent().parent().parent().parent().parent().DBconn.rollback()
        cur.close()
        
        for thread in self.pool:
            thread.join()
        self.pool, self.res_q = None, None
    
    def getNextFrame(self, text):
        state = self.findChildren(QtGui.QCheckBox)[0].checkState()
        if state > 0:
            try:
                ret, frame, strtime = self.res_q.get(timeout=0.1)
            except:
                ret = False
            if ret and self.isShowcasing == 1:
                print strtime
                display = self.parent().parent().parent().parent().findChildren(DisplayWidget)[0]
                display.setDisplay(frame)
            elif not ret:
                if self.isShowcasing != -1: QtGui.QMessageBox.warning(self, 'Stopped Streaming', 'Camera '+str(self.idx)+': '+self.name)
                self.findChildren(QtGui.QCheckBox)[0].setCheckState(0)

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
            self.parent().isShowcasing = 1
            self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
            self.setLineWidth(3)
            cameras = self.parent().parent().findChildren(Camera)
            for cam in cameras:
                label = cam.findChildren(CameraLabel)[0]
                if cam.idx != self.parent().idx and label.isActive:
                    label.toggleActive()
        else:
            self.parent().isShowcasing = 0
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

class ATCCLogin(QtGui.QDialog):
    def __init__(self, users, parent=None):
        super(ATCCLogin, self).__init__(parent)
        self.users = users
        self.setWindowTitle("Login Panel")
        self.user, self.passwd = 'Username', 'Password'
        self.textName = QtGui.QLineEdit(self.user,self)
        pal, font = self.textName.palette(), self.textName.font()
        pal.setColor(6, QtGui.QColor(QtCore.Qt.gray)); font.setBold(True)
        self.textName.setPalette(pal), self.textName.setFont(font)
        self.textPass = QtGui.QLineEdit(self.passwd,self)
        self.textPass.setPalette(pal), self.textPass.setFont(font)
        self.textPass.setEchoMode(2)
        self.buttonLogin = QtGui.QPushButton('Login', self)
        self.buttonLogin.clicked.connect(self.handleLogin)
        layout = QtGui.QVBoxLayout(self)
        layout.addWidget(QtGui.QLabel("        AiSEE Tech: ATCC       ", self))
        layout.addWidget(self.textName)
        layout.addWidget(self.textPass)
        layout.addWidget(self.buttonLogin)
        self.textName.textChanged.connect(self.getName)
        self.textPass.textChanged.connect(self.getPass)
        self.textName.setFocus(), self.textName.selectAll()
        self.setFixedSize(200,150)
        
    def getName(self):
        if self.user != self.textName.text():
            pal = self.textName.palette()
            pal.setColor(6, QtGui.QColor(QtCore.Qt.black))
            self.textName.setPalette(pal)
            self.user = self.textName.text()
    
    def getPass(self):
        if self.passwd != self.textPass.text():
            pal = self.textPass.palette()
            pal.setColor(6, QtGui.QColor(QtCore.Qt.black))
            self.textPass.setPalette(pal)
            self.passwd = self.textPass.text()
            
    def handleLogin(self):
        upair = (str(self.textName.text()),str(self.textPass.text()))
        for user in self.users:
            if user[1] == upair[0] and user[2] == upair[1]:
                self.users = user
                self.accept()
                return
        QtGui.QMessageBox.warning(self, 'Error', 'Username or Password does not match our database')

def main():
    app = QtGui.QApplication(sys.argv)
    atcc_gui = ATCCMainWindow()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
