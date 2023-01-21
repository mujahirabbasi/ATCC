import cv2
import numpy as np
import math
from munkres import Munkres

class Detector(object):
    def __init__(self,classes=['car'],fw=0.8,fh=0.8):
        self.fw = fw
        self.fh = fh
        self.classes = classes
        self.probs = {'SMV': 0.3, 'TWB': 0.1, 'HMV': 0.3, 'PED': 0.1}
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
        
        if speed in range(1,3):
            self.goneTooLong = 15
            self.ageT = 8
            self.vis = 0.6
            if self.tag not in ['TWB', 'PED']: self.costOfNonAssignment = 0.7
            else: self.costOfNonAssignment = 30
        elif speed in range(3,5):
            self.goneTooLong = 10
            self.ageT = 4
            self.vis = 0.5
            if self.tag not in ['TWB', 'PED']: self.costOfNonAssignment = 0.8
            else: self.costOfNonAssignment = 50
        else:
            self.goneTooLong = 8
            self.ageT = 2
            self.vis = 0.5
            if self.tag not in ['TWB', 'PED']: self.costOfNonAssignment = 0.9
            else: self.costOfNonAssignment = 80
        
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
    def __init__(self,classes=['car'],box=[],line=[],speed=3):
        self.detector = Detector(classes)
        
        self.Trackers, self.count = {}, {}
        for tag in classes:
            self.Trackers[tag] = Tracker(tag,speed)
            self.count[tag] = [0, 0]
            
        self.DElements = [box,line,False,[0,0,0],(255,0,0),(0,255,255)]
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
        w = abs(self.DElements[0][1][0] - self.DElements[0][0][0])
        x = min(self.DElements[0][0][0],self.DElements[0][1][0])
        h = abs(self.DElements[0][1][1] - self.DElements[0][0][1])
        y = min(self.DElements[0][0][1],self.DElements[0][1][1])
        self.detector.detectObjects(outputs,x,y,w,h)
        counted = False
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
                    if not track['isCounted']:
                        check = self.hasCrossedLine(track)
                        if check != -1:
                            self.count[tag][check] += 1
                            counted = True
                            track['isCounted'] = True
        if counted:
            self.DElements[5] = (0,0,0)
        else:
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
