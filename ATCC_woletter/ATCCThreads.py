import os, time
import threading, Queue
import cv2
import pyyolo
from PyQt4 import QtCore
from ATCCTracker import *

class ReaderThread(threading.Thread):
    def __init__(self, cap_q, para):
        super(ReaderThread, self).__init__()
        self.link = 'test1.mp4'#para[0]
        self.speed = para[1]
        self.cap_q = cap_q
        self.stopRequest = threading.Event()
        
    def run(self):
        try:
            cap = cv2.VideoCapture(self.link)
            ret, frame = cap.read()
            if ret:
                if frame.shape[1] > 1300:
                    dim = (16*70,9*70)
                else:
                    dim = (frame.shape[1],frame.shape[0])
        except:
            pass
        
        i = 0
        while not self.stopRequest.isSet():
            if self.cap_q.qsize() > 5:
                continue
            tic = time.time()
            try:
                ret, frame = cap.read()
                i += 1
            except:
                continue
            if ret and i%self.speed == 0:
                frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
                self.cap_q.put((ret, frame))
            elif not ret:
                self.cap_q.put((ret, []))
            print 'A', time.time() - tic
    
    def join(self, timeout=None):
        self.stopRequest.set()
        super(ReaderThread, self).join(timeout)
        
class PreprocThread(threading.Thread):
    def __init__(self, cap_q, det_q, para):
        super(PreprocThread, self).__init__()
        self.cap_q = cap_q
        self.det_q = det_q
        self.xywh = [para[2][0][0], para[2][0][1], abs(para[2][1][0] - para[2][0][0]), abs(para[2][1][1] - para[2][0][1])]
        self.stopRequest = threading.Event()
        
    def run(self):
        x, y, w, h = self.xywh
        
        while not self.stopRequest.isSet():
            if self.det_q.qsize() > 5:
                continue
            tic = time.time()
            try:
                ret, frame = self.cap_q.get(timeout=0.01)
            except Queue.Empty:
                continue
            if ret:    
                data = self.letterbox(frame[y:y+h,x:x+w])
                data = data.ravel()/255.0
                data = np.ascontiguousarray(data, dtype=np.float32)
                self.det_q.put((ret, frame, data))
            else:
                self.det_q.put((ret, [], []))
            print 'B', time.time() - tic
                
    def join(self, timeout=None):
        self.stopRequest.set()
        super(PreprocThread, self).join(timeout)
        
    def letterbox(self, frame):
        w, h = 416, 416
        imh, imw, imc = frame.shape
        if float(w)/imw < float(h)/imh:
            new_w = w;
            new_h = (imh * w)/imw;
        else:
            new_h = h;
            new_w = (imw * h)/imh;
        resized = cv2.resize(frame, (new_w,new_h))#, interpolation = cv2.INTER_AREA)
        boxed = 0.5*np.ones((h,w,imc), dtype=np.uint8)
        fh, fw = (h-new_h)/2, (w-new_w)/2
        boxed[fh:fh+new_h,fw:fw+new_w] = resized
        return boxed.transpose(2,0,1)
        
class DetectionThread(threading.Thread):
    def __init__(self, det_q, out_q, para, gpudev):
        super(DetectionThread, self).__init__()
        self.det_q = det_q
        self.out_q = out_q
        self.w, self.h = abs(para[2][1][0] - para[2][0][0]), abs(para[2][1][1] - para[2][0][1])
        self.gpu = gpudev
        self.stopRequest = threading.Event()
        
    def run(self):
        cwd = os.getcwd()
        os.chdir(os.path.expanduser('~')+'/Software/pyyolo/darknet')
        datacfg = 'cfg/coco.data'
        cfgfile = 'cfg/yolo.cfg'
        weightfile = 'models/yolo.weights'
        thresh = 0.1
        hier_thresh = 0.7
        pyyolo.init(datacfg, cfgfile, weightfile, self.gpu)
        os.chdir(cwd)
        
        while not self.stopRequest.isSet():
            if self.out_q.qsize() > 5:
                continue
            tic = time.time()
            try:
                ret, frame, data = self.det_q.get(timeout=0.01)
            except Queue.Empty:
                continue
            
            if ret:
                outputs = pyyolo.detect(self.w, self.h, 3, data, thresh, hier_thresh)
                self.out_q.put((ret, frame, outputs))
            else:
                self.out_q.put((ret, [], []))
            print 'C', time.time() - tic
                
    def join(self, timeout=None):
        self.stopRequest.set()
        super(DetectionThread, self).join(timeout)
        
class ATCCThread(threading.Thread, QtCore.QObject):
    def __init__(self, out_q, res_q, para, parent=None):
        #super(ATCCThread, self).__init__()
        QtCore.QObject.__init__(self,parent)
        threading.Thread.__init__(self)
        self.out_q = out_q
        self.res_q = res_q
        self.stopRequest = threading.Event()
        self.counter = AutoCounter(classes=['SMV','HMV','TWB'],box=para[2],line=para[3],speed=para[1])
            
    def run(self):
        while not self.stopRequest.isSet():
            tic = time.time()
            try:
                ret, frame, outputs = self.out_q.get(timeout=0.01)
            except Queue.Empty:
                continue
            if ret:
                self.counter.processNextFrame(frame, outputs)
                img = self.counter.drawElements(frame.copy())
                img = self.counter.displayCount(img)
                self.res_q.put((ret, img))
            else:
                self.res_q.put((ret, []))
                self.stopRequest.set()
            self.emit(QtCore.SIGNAL('data(PyQt_PyObject)'), QtCore.QString("test"))
            print 'D', time.time() - tic
            
    def join(self, timeout=None):
        self.stopRequest.set()
        super(ATCCThread, self).join(timeout)
