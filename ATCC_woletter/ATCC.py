import pyyolo
import numpy as np
import multiprocessing as mp
import math
from munkres import Munkres
import cv2
import sys, os
import time
from datetime import date
import csv

def readCapture(infile, speed, stream, capBuff):
    i = 0
    while stream.value == 0: continue
    tot = time.time()
    cap = cv2.VideoCapture(infile)
    while stream.value == 1:
        if capBuff.qsize() >= 10:
            continue
        start = time.time()
        ret, frame = cap.read()
        if ret:
            if i%speed == 0:
                capBuff.put(frame)
            i += 1
        else:
            capBuff.close()
            stream.value = 2
    cap.release()
    if stream.value == 0:
        while True:
            try:
                frame = capBuff.get(timeout=0.1)
            except:
                return
    while stream.value < 3:
        pass
        
def preProcess(xywh , dim, stream, capBuff, detBuff):
    def letterbox(frame):
        w, h = 416, 416
        imh, imw, imc = frame.shape
        if float(w)/imw < float(h)/imh:
            new_w = w;
            new_h = (imh * w)/imw;
        else:
            new_h = h;
            new_w = (imw * h)/imh;
        resized = cv2.resize(frame, (new_w,new_h), interpolation = cv2.INTER_AREA)
        boxed = 0.5*np.ones((h,w,imc), dtype=np.uint8)
        fh, fw = (h-new_h)/2, (w-new_w)/2
        boxed[fh:fh+new_h,fw:fw+new_w] = resized
        return boxed.transpose(2,0,1)
        
    while stream.value == 0: continue
    while stream.value >= 1:
        x, y, w, h = xywh
        if detBuff.qsize() >= 10:
            continue
        try:
            frame = cv2.resize(capBuff.get(timeout=0.1),dim,interpolation=cv2.INTER_AREA)
            data = letterbox(frame[y:y+h,x:x+w])
            data = data.ravel()/255.0
            data = np.ascontiguousarray(data, dtype=np.float32)
            detBuff.put((frame, data))
        except:
            if stream.value == 2:
                detBuff.close()
                stream.value = 3
                break
    if stream.value == 0:
        while True:
            try:
                frame = capBuff.get(timeout=0.1)
            except:
                while True:
                    try:
                        frame, data = detBuff.get(timeout=0.1)
                    except:
                        return
    while stream.value < 4:
        pass
        
def detect(xywh, stream, detBuff, outBuff):
    cwd = os.getcwd()
    os.chdir(os.path.expanduser('~')+'/Software/pyyolo/darknet')
    datacfg = 'cfg/coco.data'
    cfgfile = 'cfg/yolo.cfg'
    weightfile = 'models/yolo.weights'
    thresh = 0.4
    hier_thresh = 0.5
    pyyolo.init(datacfg, cfgfile, weightfile)
    os.chdir(cwd)

    stream.value = 1

    while stream.value >= 1:
        w, h = xywh[2:]
        c = 3
        if outBuff.qsize() >= 10:
            continue
        try:
            frame, data = detBuff.get(timeout=0.1)
            start = time.time()
            outputs = pyyolo.detect(w, h, c, data, thresh, hier_thresh)
            outBuff.put((frame, outputs))
        except:
            if stream.value == 3:
                outBuff.close()
                stream.value = 4
                break
    if stream.value == 0:
        while True:
            try:
                frame, data = detBuff.get(timeout=0.1)
            except:
                while True:
                    try:
                        frame, outputs = outBuff.get(timeout=0.1)
                    except:
                        return
    while stream.value < 5:
        pass

class Detector(object):
    def __init__(self,classes=['car'],fw=0.5,fh=0.5):
        self.fw = fw
        self.fh = fh
        self.classes = classes
        self.probs = {'SMV': 0.3, 'TWB': 0.2, 'HMV': 0.3, 'PED': 0.2}
        self.output = []

    def detectObjects(self, outputs, x, y,w,h):
        outputs = [o for o in outputs if o['class'] in ['car', 'person', 'bicycle', 'motorbike', 'bus', 'truck']]
        self.output = []
        for o in outputs:
            o['left'], o['right'] = o['left'] + x, o['right'] + x
            o['top'], o['bottom'] = o['top'] + y, o['bottom'] + y
            o['width'] = o['right'] - o['left']
            o['height'] = o['bottom'] - o['top']
            if o['class'] == 'car': o['class'] = 'SMV'
            elif o['class'] in ['motorbike', 'bicycle']: o['class'] = 'TWB'
            elif o['class'] in ['bus', 'truck']: o['class'] = 'HMV'
            elif o['class'] == 'person': o['class'] = 'PED'
            if o['class'] in self.classes and o['width'] < self.fw*w and o['height'] < self.fh*h and o['prob'] > self.probs[o['class']]:
                self.output.append(o)
    
    def getClasses(self):
        return [o['class'] for o in self.output]

    def getProbs(self,tag='car'):
        return [o['prob'] for o in self.output if o['class'] == tag]
         
    def getBboxes(self,tag='car'):
        return [[(o['left'],o['top']),(o['right'],o['bottom'])] for o in self.output if o['class'] == tag]
        
    def getCenters(self,tag='car'):
        return [((o['left']+o['right'])/2,(o['top']+o['bottom'])/2) for o in self.output if o['class'] == tag]
        
    def getSizes(self,tag='car'):
        return [(o['width'],o['height']) for o in self.output if o['class'] == tag]

class Tracker(object):
    def __init__(self,tag='car',speed=3):
        self.tracks = []
        self.nextId = 1
        self.tag = tag
        self.costOfNonAssignment = 0.7
        
        if speed in range(1,3):
            self.goneTooLong = 15
            self.ageT = 8
            self.vis = 0.6
            self.costOfNonAssignment = 0.6
        elif speed in range(3,5):
            self.goneTooLong = 10
            self.ageT = 5
            self.vis = 0.6
            self.costOfNonAssignment = 0.75
        else:
            self.goneTooLong = 8
            self.ageT = 4
            self.vis = 0.5
            self.costOfNonAssignment = 0.9
        
    def updateTracker(self, bboxes, centers):
        self.predictNewLocationsOfTracks()
        assignments, unassignedTracks, unassignedDetections = self.detectionToTrackAssignment(bboxes, centers)
        self.updateAssignedTracks(bboxes, centers, assignments)
        self.updateUnassignedTracks(unassignedTracks)
        self.deleteLostTracks()
        self.createNewTracks(bboxes, centers, unassignedDetections)
    
    def initializeTrack(self, bbox, KF, hist):
        if self.nextId >= 1000:
            self.nextId = 1
            for track in self.tracks:
                track['id'] = self.nextId
                self.nextId += 1
        track = {'id': self.nextId, 'bbox': bbox, 'history': [hist], 'isCounted': False,\
                 'KF': KF, 'age': 1, 'totalVisibleCount': 1, 'conseqInvisibleCount': 0}
        self.tracks.append(track)
        self.nextId += 1
                            
    def predictNewLocationsOfTracks(self):
        for track in self.tracks:
            kfp = track['KF'].predict()
            bbox = track['bbox']
            w, h = bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1]
            track['bbox'] = [(int(kfp[0][0]-w/2.0),int(kfp[1][0]-h/2.0)), \
                             (int(kfp[0][0]+w/2.0),int(kfp[1][0]+h/2.0))]
            track['history'].append((int(kfp[0][0]),int(kfp[1][0])))
                             
    def detectionToTrackAssignment(self, bboxes, centers):
        cost = []
        if self.tag in ['TWB', 'PED']:
            self.costOfNonAssignment = 30
            for track in self.tracks:
                tCenter = track['KF'].statePost[0:2]
                P, H, R = track['KF'].errorCovPost, track['KF'].measurementMatrix, track['KF'].measurementNoiseCov
                S = (H.dot(P)).dot(np.transpose(H)) + R
                if centers: cost.append([])
                for c in centers:
                    dCenter = np.array([[c[0]],[c[1]]])
                    X = dCenter - tCenter
                    dist = ((np.transpose(X)).dot(np.linalg.inv(S))).dot(X) + math.log(np.linalg.det(S))
                    cost[-1].append(math.sqrt(dist))
        else:
            self.costOfNonAssignment = 0.7
            for track in self.tracks:
                if bboxes: cost.append([])
                tbox = track['bbox']
                for dbox in bboxes:
                    St = (tbox[1][0] - tbox[0][0])*(tbox[1][1] - tbox[0][1])
                    Sd = (dbox[1][0] - dbox[0][0])*(dbox[1][1] - dbox[0][1])
                    dx = float(min(tbox[1][0],dbox[1][0]) - max(tbox[0][0],dbox[0][0]))
                    dy = float(min(tbox[1][1],dbox[1][1]) - max(tbox[0][1],dbox[0][1]))
                    if dx > 0. and dy > 0.:
                        Si = dx*dy
                    else:
                        Si = 0.
                    Su = St + Sd - Si
                    cost[-1].append(1. - (Si/Su))
        if cost:
            m = Munkres()
            assignments = m.compute(cost)
            assignments = [(i,j) for i,j in assignments if cost[i][j] < self.costOfNonAssignment]
            asgnTracks = [a[0] for a in assignments]
            asgnDetections = [a[1] for a in assignments]
        else:
            assignments = asgnTracks = asgnDetections = []
        unassignedTracks = [i for i in range(len(self.tracks)) if i not in asgnTracks]
        unassignedDetections = [i for i in range(len(bboxes)) if i not in asgnDetections]
        return assignments, unassignedTracks, unassignedDetections
        
    def updateAssignedTracks(self, bboxes, centers, assignments):
        for t, d in assignments:
            self.tracks[t]['KF'].correct(np.array([[centers[d][0]],[centers[d][1]]],np.float32))
            self.tracks[t]['bbox'] = bboxes[d]
            self.tracks[t]['age'] += 1
            self.tracks[t]['totalVisibleCount'] += 1
            self.tracks[t]['conseqInvisibleCount'] = 0
            self.tracks[t]['history'][-1] = (centers[d][0],centers[d][1])
            
    def updateUnassignedTracks(self, unassignedTracks):
        for i in unassignedTracks:
            self.tracks[i]['age'] += 1
            self.tracks[i]['conseqInvisibleCount'] += 1
            
    def deleteLostTracks(self):
        Ntracks = []
        for track in self.tracks:
            visibility = track['totalVisibleCount'] / float(track['age'])
            if (track['age'] > self.ageT or visibility > self.vis) and track['conseqInvisibleCount'] < self.goneTooLong:
                Ntracks.append(track)
        self.tracks = Ntracks
        
    def createNewTracks(self, bboxes, centers, unassignedDetections):
        for new in unassignedDetections:
            centroid = centers[new]
            bbox = bboxes[new]
            
            kalman = cv2.KalmanFilter(4, 2)
            kalman.transitionMatrix = np.array([[1.,0.,1.,0.], [0.,1.,0.,1.],\
                                                [0.,0.,1.,0.], [0.,0.,0.,1.]],np.float32)
            kalman.measurementMatrix = np.array([[1.,0.,0.,0.], [0.,1.,0.,0.]],np.float32)
            kalman.processNoiseCov = 10*np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],\
                                                  [0.,0.,.25,0.],[0.,0.,0.,.25]],np.float32)
            kalman.measurementNoiseCov = 10*np.array([[1.,0.],[0.,1.]],np.float32)
            kalman.errorCovPost = 20*np.array([[1.,0.,0.,0.], [0.,1.,0.,0.],\
                                           [0.,0.,.25,0.], [0.,0.,0.,.25]],np.float32)
            kalman.statePost = np.array([[centroid[0]],[centroid[1]],[0],[0]],np.float32)
             
            hist = (centroid[0],centroid[1])
            self.initializeTrack(bbox, kalman, hist)

class AutoCounter(object):
    def __init__(self,frame,mode=0,classes=['car'],box=[],line=[],speed=3):
        self.detector = Detector(classes)
        
        self.Trackers, self.count = {}, {}
        for tag in classes:
            self.Trackers[tag] = Tracker(tag,speed)
            self.count[tag] = [0, 0]
        
        if len(box) == 0:
            wp, hp = int(0.1*frame.shape[1]), int(0.1*frame.shape[0])
            box = [(wp,hp),(shape[1]-wp,shape[0]-hp)]
            line = [(0,shape[0]/2),(shape[1],shape[0]/2)]
            
        self.DElements = [box,line,False,[0,0,0],(255,0,0),(0,255,255)]
        self.getLineEq()
        self.mode = mode
        
    def getDElements(self,event,x,y,flag,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.DElements[param][0] = (x,y)
            self.DElements[param][1] = (x,y)
            self.DElements[2] = True
            for tag in self.Trackers:
                self.Trackers[tag] = Tracker(tag)
                self.count[tag] = [0, 0]
        elif event == cv2.EVENT_MOUSEMOVE and self.DElements[2]:
            self.DElements[param][1] = (x,y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.DElements[2] = False
            self.DElements[param][1] = (x,y)
            print self.DElements[param]
        if param == 0:
            w = self.DElements[0][1][0] - self.DElements[0][0][0]
            h = self.DElements[0][1][1] - self.DElements[0][0][1]
            if abs(w) >= abs(h):
                i = 0
                if w != 0: i = abs(w)/w
                hy = self.DElements[0][0][1] + h/2
                lx = self.DElements[0][0][0] - i*20
                rx = self.DElements[0][1][0] + i*20
                self.DElements[1] = [(lx,hy),(rx,hy)]
            else:
                i = 0
                if h != 0: i = abs(h)/h
                wx = self.DElements[0][0][0] + w/2
                ty = self.DElements[0][0][1] - i*10
                by = self.DElements[0][1][1] + i*10
                self.DElements[1] = [(wx,ty),(wx,by)]
        self.getLineEq()
    
    def getLineEq(self):
        a = self.DElements[1][1][1] - self.DElements[1][0][1]
        b = self.DElements[1][0][0] - self.DElements[1][1][0]
        c = self.DElements[1][1][0]*self.DElements[1][0][1] - self.DElements[1][0][0]*self.DElements[1][1][1]
        self.DElements[3] = [a, b, c]
        
    def drawElements(self,frame):
        col = self.DElements[4]
        cv2.rectangle(frame,self.DElements[0][0],self.DElements[0][1],col,4)
        col = self.DElements[5]
        cv2.line(frame,self.DElements[1][0],self.DElements[1][1],col,4)
        return frame
        
    def processNextFrame(self, frame, outputs):
        font = cv2.FONT_HERSHEY_SIMPLEX
        w = abs(self.DElements[0][1][0] - self.DElements[0][0][0])
        x = min(self.DElements[0][0][0],self.DElements[0][1][0])
        h = abs(self.DElements[0][1][1] - self.DElements[0][0][1])
        y = min(self.DElements[0][0][1],self.DElements[0][1][1])
        self.detector.detectObjects(outputs,x,y,w,h)
        counted = False
        tStamp = time.strftime('%H:%M:%S', time.localtime())
        for tag in self.Trackers:
            bboxes = self.detector.getBboxes(tag)
            centers = self.detector.getCenters(tag)
            self.Trackers[tag].updateTracker(bboxes, centers)
            for box in bboxes:
                cv2.rectangle(frame,box[0],box[1],(255,255,255),2)
            for track in self.Trackers[tag].tracks:
                if track['age'] > self.Trackers[tag].ageT:
                    cv2.rectangle(frame,track['bbox'][0],track['bbox'][1],(0,0,255),1)
                    for pt in track['history'][-15:]:
                        cv2.circle(frame,pt,3,(0,255,0),2)
                    #cv2.putText(frame,str(track['id']),track['history'][-1],font,0.7,(0,0,0),2,cv2.LINE_AA)
                    if self.mode == 0 and not track['isCounted']:
                        check = self.hasCrossedLine(track)
                        if check != -1:
                            self.count[tag][check] += 1
                            counted = True
                            track['isCounted'] = True
                    elif self.mode == 1 and not track['isCounted']:
                        check = self.leftDetBox(track)
                        if check != -1:
                            self.count[tag][check] += 1
                            counted = True
                            track['isCounted'] = True
        if counted:
            if self.mode == 1: self.DElements[4] = (0,0,0)
            else: self.DElements[5] = (0,0,0)
        else:
            self.DElements[4] = (255,0,0)
            self.DElements[5] = (0,255,255)
    
    def hasCrossedLine(self,track):
        l = self.DElements[3]
        pt1 = track['history'][0]
        pt2 = track['history'][-1]
        sg1 = pt1[0]*l[0] + pt1[1]*l[1] + l[2]
        sg2 = pt2[0]*l[0] + pt2[1]*l[1] + l[2]
        if sg1*sg2 <= 0:
            if l[2]*sg1 <= 0: return 1
            else: return 0
        return -1
        
    def leftDetBox(self,track):
        box = self.DElements[0]
        pt1 = track['history'][-2]
        pt2 = track['history'][-1]
        a = (pt1[0] > min(box[0][0],box[1][0])) & (pt1[0] < max(box[0][0],box[1][0]))
        a &= (pt1[1] > min(box[0][1],box[1][1])) & (pt1[1] < max(box[0][1],box[1][1]))
        b = (pt2[0] > min(box[0][0],box[1][0])) & (pt2[0] < max(box[0][0],box[1][0]))
        b &= (pt2[1] > min(box[0][1],box[1][1])) & (pt2[1] < max(box[0][1],box[1][1]))
        if a and not b:
            return 0
        if not a and b:
            return 1
        return -1
        
    def displayCount(self,frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
    
        h, w, c = frame.shape[0], int(0.15*frame.shape[1]), 3
        img = np.zeros((h,w,c),np.uint8)
    
        fh, th = 0, int(0.1*h)
        img[fh:fh+th,0:w] = np.concatenate((255*np.ones((th,w,1),np.uint8),255*np.ones((th,w,1),np.uint8),0*np.ones((th,w,1),np.uint8)),2)
        cv2.putText(img,'ATCC',(int(0.2*w),fh+int(0.8*th)),font,1.5,(0,0,0),3,cv2.LINE_AA)
        
        fh, th = fh+th, int(0.1*h)
        img[fh:fh+th,0:w] = 50*np.ones((th,w,3),np.uint8)
        cv2.putText(img,'Dir 1',(int(0.3*w),fh+int(0.7*th)),font,1,(255,255,255),3,cv2.LINE_AA)
        
        for tag in self.Trackers:
            fh, th = fh+th, int(0.1*h)
            cv2.putText(img,tag+':  '+str(self.count[tag][0]),(int(0.1*w),fh+int(0.8*th)),font,0.7,(255,255,255),2,cv2.LINE_AA)
            
        fh, th = fh+th, int(0.1*h)
        img[fh:fh+th,0:w] = 50*np.ones((th,w,3),np.uint8)
        cv2.putText(img,'Dir 2',(int(0.3*w),fh+int(0.7*th)),font,1,(255,255,255),3,cv2.LINE_AA)
        
        for tag in self.Trackers:
            fh, th = fh+th, int(0.1*h)
            cv2.putText(img,tag+':  '+str(self.count[tag][1]),(int(0.1*w),fh+int(0.8*th)),font,0.7,(255,255,255),2,cv2.LINE_AA)
            
        frame = np.concatenate((frame,img),axis=1)
        return frame
        
def emptyCallback(event,x,y,flag,param):
    pass
       
def writeCount(wsx, AC, classes, row):
    global closexl
    row += 1
    if closexl:
        return
    else:
        Timer(10, writeCount, (wsx, AC, classes, row)).start()
    
    wsx['A'+str(row)] = time.strftime('%H:%M:%S', time.localtime())
    col = 2
    for i in range(2):
        sumc = 0
        for tag in classes:
            sumc += AC.count[tag][i]
            wsx[getCell(row,col)] = AC.count[tag][i]
            if row >= 8:
                wsx[getCell(row,col+1)] = "=AVERAGE("+getCell(row-4,col)+":"+getCell(row,col)+")"
            if row >= 13:
                wsx[getCell(row,col+2)] = "=AVERAGE("+getCell(row-9,col)+":"+getCell(row,col)+")"
            col += 3
        wsx[getCell(row,col)] = sumc
        if row >= 8:
            wsx[getCell(row,col+1)] = "=AVERAGE("+getCell(row-4,col)+":"+getCell(row,col)+")"
        if row >= 13:
            wsx[getCell(row,col+2)] = "=AVERAGE("+getCell(row-9,col)+":"+getCell(row,col)+")"
        col += 3
         
if __name__ == '__main__':
    #infile = '../videos/ParkJL_test.mp4'
    #infile = 'rtsp://172.16.24.26/media/video1'
    #infile = 'rtsp://172.16.18.2/media/video1'
    infile, jId, speed, classes = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
    record, out, closexl = False, False, False
    isPlaying = False
    
    ####################File output###########################################
    outFilen = 'demo.xlsx'
    if jId == 1: #ParkJL
        outFilen = 'ParkJL'+'1'
    elif jId == 2: #AJC
        outFilen = 'AJC_skp'
    elif jId == 3: #Mullick
        outFilen = 'Mullick'
    elif jId == 4: #Birla
        outFilen = 'BirlaPlanet'
    elif jId == 5: #DLNKB
        outFilen = 'DLNKB'
    elif jId == 6: #Mayo
        outFilen = 'MayoRd'
    elif jId == 7: #HOSPRD
        outFilen = 'HOSPRD'
    outFilen = outFilen+'_'+date.today().strftime('%d-%b-%Y')+'.csv'
    if os.path.isfile(outFilen):
        outFilen = open(outFilen,'a+b')
        logFile = csv.writer(outFilen,delimiter=',',quoting=csv.QUOTE_MINIMAL)
    else:
        outFilen = open(outFilen,'w+b')
        logFile = csv.writer(outFilen,delimiter=',',quoting=csv.QUOTE_MINIMAL)
        logFile.writerow(['TimeStamp','Type','Direction','ImgFile','Color','X','Y'])
    ####################File output###########################################
    
    name = 'ATCC'
    cv2.namedWindow(name,cv2.WINDOW_AUTOSIZE)
    cap = cv2.VideoCapture(infile)
    ret, frame = cap.read()
    if ret:
        if frame.shape[1] > 1300:
            dim = (16*70,9*70)
            if jId in [5,6,7]: dim = (4*240,3*240)
        else:
            dim = (frame.shape[1],frame.shape[0])
        
        frame = cv2.resize(frame,dim,interpolation=cv2.INTER_AREA)
        cv2.imshow(name,frame)
        cv2.waitKey(50)
        ATCC_counter = AutoCounter(frame,0,classes=classes)
    else:
        print 'Cannot read video file'
    cap.release()
    
    stream = mp.Value('i', 0)
    xywh = mp.Array('i', [0,0,0,0])
    capBuff = mp.Queue()
    detBuff = mp.Queue()
    outBuff = mp.Queue()
        
    procs = []
    procs.append(mp.Process(target=readCapture, args=(infile, speed, stream, capBuff)))
    procs.append(mp.Process(target=preProcess, args=(xywh, dim, stream, capBuff, detBuff)))
    procs.append(mp.Process(target=detect, args=(xywh, stream, detBuff, outBuff)))
    #procs.append(mp.Process(target=display, args=(stream, outBuff)))
    
    for p in procs:
        p.start()
    #===============================================================
    while stream.value == 0: continue
    tot = time.time()
    i = 0
    while stream.value >= 1:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('b'):
            if not isPlaying:
                cv2.setMouseCallback(name,ATCC_counter.getDElements,0)
        elif key == ord('l'):
            if not isPlaying:
                cv2.setMouseCallback(name,ATCC_counter.getDElements,1)
        elif key == ord(' '):
            isPlaying = not isPlaying
            cv2.setMouseCallback(name,emptyCallback)
        elif key == ord('q'):
            closexl = True
            stream.value = 0
        if isPlaying:
            try:
                start = time.time()
                frame, outputs = outBuff.get(timeout=0.1)
                ATCC_counter.processNextFrame(frame, outputs, logFile)
                img = ATCC_counter.drawElements(frame.copy())
                img = ATCC_counter.displayCount(img)
                cv2.imshow(name,img)
                if record: out.write(img)
                i += 1
                print 'D', time.time() - tot, time.time() - start, i
            except:
                if stream.value == 4:
                    stream.value = 5
                    break
        else:
            xywh[2] = abs(ATCC_counter.DElements[0][1][0] - ATCC_counter.DElements[0][0][0])
            xywh[0] = min(ATCC_counter.DElements[0][0][0],ATCC_counter.DElements[0][1][0])
            xywh[3] = abs(ATCC_counter.DElements[0][1][1] - ATCC_counter.DElements[0][0][1])
            xywh[1] = min(ATCC_counter.DElements[0][0][1],ATCC_counter.DElements[0][1][1])
            img = ATCC_counter.drawElements(frame.copy())
            img = ATCC_counter.displayCount(img)
            cv2.imshow(name,img)
            if record and not out:
                fourcc = cv2.VideoWriter_fourcc(*'H264')
                out = infile.split('.')
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps < 20.: fps = 12.5
                if fps > 100.: fps = 25.
                out = cv2.VideoWriter(out[0]+'_atcc_x'+str(speed)+'.mkv',fourcc, fps, (img.shape[1],img.shape[0]))
    cv2.destroyAllWindows()
    closexl = True
    outFilen.close()
    
    if record: out.release()
    if stream.value == 0:
        while True: 
            try:
                frame, outputs = outBuff.get(timeout=0.1)
            except:
                break
    #===============================================================
    for p in procs:
        p.join()
    
    print time.time() - tot
