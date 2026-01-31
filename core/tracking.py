# -*- coding: utf-8 -*-
import numpy as np
from collections import OrderedDict
import time
from scipy.spatial import distance as dist

class CentroidTracker:
    def __init__(self, max_disappeared=30):
        self.next_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
            del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for tid in list(self.disappeared.keys()):
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared: self.deregister(tid)
            return self.objects
        input_centroids = np.array([(int((r[0]+r[2])/2), int((r[1]+r[3])/2)) for r in rects])
        if len(self.objects) == 0:
            for i in range(len(input_centroids)): self.register(input_centroids[i])
        else:
            oids = list(self.objects.keys()); ocs = list(self.objects.values())
            D = dist.cdist(np.array(ocs), input_centroids)
            rows = D.min(axis=1).argsort(); cols = D.argmin(axis=1)[D.min(axis=1).argsort()]
            ur, uc = set(), set()
            for (r, c) in zip(rows, cols):
                if r in ur or c in uc: continue
                self.objects[oids[r]] = input_centroids[c]; self.disappeared[oids[r]] = 0
                ur.add(r); uc.add(c)
            for r in set(range(D.shape[0])) - ur:
                self.disappeared[oids[r]] += 1
                if self.disappeared[oids[r]] > self.max_disappeared: self.deregister(oids[r])
            for c in set(range(D.shape[1])) - uc: self.register(input_centroids[c])
        return self.objects

class PedestrianFlowManager:
    def __init__(self, line_pts=None, interval=60):
        self.line_pts = line_pts if line_pts else [(100, 400), (500, 400)]
        self.in_side_sign = 1 
        self.interval, self.start_time = interval, time.time()
        self.in_total = 0
        self.out_total = 0
        self.track_history = {}          
        self.last_in_time = {}           
        self.crossing_time = {}          

    def set_line(self, p1, p2):
        self.line_pts = [p1, p2]
        self.track_history.clear()
        self.last_in_time.clear()
        self.crossing_time.clear()

    def set_in_side(self, click_pos):
        (x1, y1), (x2, y2) = self.line_pts
        self.in_side_sign = 1 if (x2 - x1) * (click_pos[1] - y1) - (y2 - y1) * (click_pos[0] - x1) > 0 else -1

    def check_crossing(self, tid, pos):
        now = time.time()
        if tid in self.crossing_time and (now - self.crossing_time[tid] < 1.0):
            return None
        (x1, y1), (x2, y2) = self.line_pts
        side = (x2 - x1) * (pos[1] - y1) - (y2 - y1) * (pos[0] - x1)
        curr_sign = 1 if side > 0 else -1
        if tid not in self.track_history:
            self.track_history[tid] = curr_sign
            return None
        prev_sign = self.track_history[tid]
        if prev_sign == curr_sign:
            self.track_history[tid] = curr_sign
            return None
        direction = "IN" if curr_sign == self.in_side_sign else "OUT"
        if direction == "OUT":
            if tid not in self.last_in_time:
                self.track_history[tid] = curr_sign
                return None
            if now - self.last_in_time[tid] < 3.0:
                self.track_history[tid] = curr_sign
                return None
        if direction == "IN":
            self.in_total += 1
            self.last_in_time[tid] = now  
        else:  
            self.out_total += 1
        self.crossing_time[tid] = now
        self.track_history[tid] = curr_sign
        return direction

    def get_status(self):
        elapsed = time.time() - self.start_time
        reset = elapsed >= self.interval
        data = {
            "in": self.in_total,
            "out": self.out_total,
            "reset": reset,
            "elapsed": int(elapsed)
        }
        if reset:
            self.in_total = 0
            self.out_total = 0
            self.start_time = time.time()
        return data