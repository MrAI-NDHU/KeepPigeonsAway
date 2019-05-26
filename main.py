from argparse import ArgumentParser
from copy import copy
from enum import auto, Enum, IntEnum
from queue import Empty as QueueEmpty
from queue import Full as QueueFull
from queue import Queue
from threading import Thread
from typing import Dict, List, Tuple
import logging
import math
import numpy
import pickle
import random
import sys
import time

import cv2
import darknet
import keyboard

IS_TEST_NEEDED = False
IS_DECIDE_ONLY = False
IS_DETECT_ONLY = False

try:
    from servo import Servo
    from servo.controller import ControllerForPCA9685
    from Jetson import GPIO
    import termios
except:
    IS_DECIDE_ONLY = True
if IS_DETECT_ONLY:
    IS_DECIDE_ONLY = True

X, Y = "x", "y"
C_RED = (0x00, 0x00, 0xFF)
C_GREEN = (0x00, 0xFF, 0x00)
C_BLUE = (0xFF, 0x00, 0x00)
C_YELLOW = (0x00, 0xFF, 0xFF)
C_MAGENTA = (0xFF, 0x00, 0xFF)
C_CYAN = (0xFF, 0xFF, 0x00)


class Rect:
    def __init__(self):
        self.x1, self.x2, self.y1, self.y2, self.cx, self.cy = \
            -1, -1, -1, -1, -1.0, -1.0
    
    def __copy__(self):
        r = Rect()
        r.x1, r.x2, r.y1, r.y2, r.cx, r.cy = \
            self.x1, self.x2, self.y1, self.y2, self.cx, self.cy
        return r


class Angle:
    def __init__(self):
        self.x, self.y = -1.0, -1.0
    
    def __copy__(self):
        a = Angle()
        a.x, a.y = self.x, self.y
        return a
    
    def dict(self) -> Dict[str, float]:
        if self.x < 0 or self.y < 0:
            return {}
        return {X: self.x, Y: self.y}


class Status(Enum):
    detecting = auto()
    confirming = auto()
    detected = auto()
    sweeping = auto()
    abandoning = auto()


class Count:
    def __init__(self):
        self.confirmation, self.sweep, self.error, self.abandon = 0, 0, 0, 0
    
    def __copy__(self):
        c = Count()
        c.confirmation, c.sweep, c.error, c.abandon = \
            self.confirmation, self.sweep, self.error, self.abandon
        return c
    
    def clear(self):
        self.confirmation, self.sweep, self.error, self.abandon = 0, 0, 0, 0


class Limit(IntEnum):
    confirmation = 6
    sweep = 72
    error = 12
    abandon = 72


class Area:
    def __init__(self):
        self.rect = Rect()
        self.angle = Angle()
        self.status = Status.detecting
        self.count = Count()


class Detection:
    def __init__(self):
        self.x1, self.x2, self.y1, self.y2 = -1, -1, -1, -1
        self.cx, self.cy, self.ax, self.ay = -1, -1, -1, -1
        self.rate = 0.0


class KeepPigeonsAway:
    
    def __init__(self, vi: str, vo: str, sw: int, sh: int):
        self.split_w, self.split_h = sw, sh
        self.laser_pin, self.servo_x_ch, self.servo_y_ch = 18, 1, 0
        self.cap_ratio = 1920 / 1080
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.showing_w, self.showing_h = 854, 480
        
        self.detecting_color = C_GREEN
        self.detected_color = C_CYAN
        self.sweeping_color = C_MAGENTA
        self.count1_color = C_BLUE
        self.count2_color = C_RED
        self.others_color = C_YELLOW
        
        self.areas = self.make_areas()
        self.init_areas_rect()
        self.arm = self.get_arm()
        self.init_laser()
        
        self.thd_deciding = \
            Thread(name="DecidingThd", target=self.thd_deciding_func)
        self.thd_sweeping = \
            Thread(name="SweepingThd", target=self.thd_sweeping_func)
        self.thd_showing = \
            Thread(name="ShowingThd", target=self.thd_showing_func)
        self.que_deciding = Queue(1)
        self.que_sweeping = Queue(1)
        self.que_showing = Queue(1)
        
        self.is_started_detecting = False
        self.is_terminated = False
        self.has_video_in = vi != ""
        self.has_video_out = vo != ""
        self.video_in, self.video_out = self.get_video_io(vi, vo)
        self.thd_showing.start()
        
        self.area_angle_spacing = Angle()
        self.areas_canter_angle = Angle()
        self.init_areas_angle()
        
        self.darknet_net, self.darknet_net_w, self.darknet_net_h = None, 0, 0
        self.darknet_meta, self.darknet_img = None, None
        self.init_darknet()
        
        if not IS_DETECT_ONLY:
            self.thd_deciding.start()
        if not IS_DECIDE_ONLY:
            self.thd_sweeping.start()
        try:
            self.loop_detecting()
        except KeyboardInterrupt:
            self.is_terminated = True
    
    def __del__(self):
        if not IS_DECIDE_ONLY:
            GPIO.cleanup()
        self.close_video_io()
        cv2.destroyAllWindows()
    
    def make_areas(self) -> List[List[Area]]:
        if IS_DETECT_ONLY:
            return None
        return [[Area() for _ in range(self.split_w)]
                for _ in range(self.split_h)]
    
    def init_darknet(self):
        config_path = "./darknet/pigeons_cfg/yolov3-tiny-pigeons.cfg"
        weight_path = \
            "./darknet/pigeons_weights/yolov3-tiny-pigeons_last.weights"
        meta_path = "./darknet/pigeons_cfg/pigeons.data"
        self.darknet_net = darknet.load_net_custom(config_path.encode(
            "ascii"), weight_path.encode("ascii"), 0, 1)
        self.darknet_net_w = darknet.network_width(self.darknet_net)
        self.darknet_net_h = darknet.network_height(self.darknet_net)
        self.darknet_meta = darknet.load_meta(meta_path.encode("ascii"))
        self.darknet_img = darknet.make_image(
            self.darknet_net_w, self.darknet_net_h, 3)
    
    def init_areas_rect(self):
        if IS_DETECT_ONLY:
            return
        
        aw, ah = self.showing_w / self.split_w, self.showing_h / self.split_h
        for ay in range(self.split_h):
            for ax in range(self.split_w):
                x1, y1 = aw * ax, ah * ay
                x2, y2 = aw * (ax + 1), ah * (ay + 1)
                a = self.areas[ay][ax]
                a.rect.x1 = int(round(x1))
                a.rect.y1 = int(round(y1))
                a.rect.x2 = int(round(x2)) - 1
                a.rect.y2 = int(round(y2)) - 1
                a.rect.cx = (x1 + x2 - 1) / 2
                a.rect.cy = (y1 + y2 - 1) / 2
    
    def init_areas_angle(self):
        if IS_DECIDE_ONLY:
            return
        
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)
        filepath = input("Load areas angle: ").strip()
        if filepath != "":
            with open(filepath, "rb") as f:
                areas: List[List[Area]] = pickle.load(f)
                for ay in range(self.split_h):
                    for ax in range(self.split_w):
                        self.areas[ay][ax].angle.x = areas[ay][ax].angle.x
                        self.areas[ay][ax].angle.y = areas[ay][ax].angle.y
            
            self.init_area_angle_spacing()
            self.init_areas_center_angle()
            self.check_areas_angle()
        else:
            self.open_laser()
            while True:
                if keyboard.is_pressed("e"):
                    break
                elif keyboard.is_pressed("up"):
                    self.arm.rotate({Y: -0.4}, True)
                    time.sleep(0.5)
                elif keyboard.is_pressed("down"):
                    self.arm.rotate({Y: 0.4}, True)
                    time.sleep(0.5)
                elif keyboard.is_pressed("left"):
                    self.arm.rotate({X: 0.5}, True)
                    time.sleep(0.5)
                elif keyboard.is_pressed("right"):
                    self.arm.rotate({X: -0.5}, True)
                    time.sleep(0.5)
                elif keyboard.is_pressed("s"):
                    termios.tcflush(sys.stdin, termios.TCIOFLUSH)
                    i, is_ok = -1, False
                    while not is_ok:
                        s = input("Input area number in [{},{}]: ").strip()
                        if s.isdecimal():
                            i = int(s)
                            if 0 <= i < self.split_w * self.split_h:
                                is_ok = True
                    a = self.areas[i // self.split_w][i % self.split_w]
                    a.angle.x = self.arm.current_angles[X]
                    a.angle.y = self.arm.current_angles[Y]
            self.close_laser()
            
            self.init_area_angle_spacing()
            self.init_areas_center_angle()
            self.check_areas_angle()
            termios.tcflush(sys.stdin, termios.TCIOFLUSH)
            filepath = input("Save areas angle: ").strip()
            if filepath != "":
                with open(filepath, "wb") as f:
                    pickle.dump(self.areas, f)
    
    def init_area_angle_spacing(self):
        for i in range(1, self.split_w * self.split_h):
            ax1, ay1 = i % self.split_w, i // self.split_w
            ax0, ay0 = (i - 1) % self.split_w, (i - 1) // self.split_w
            if ax1 > ax0:
                area1_x = self.areas[ay1][ax1].angle.x
                area0_x = self.areas[ay0][ax0].angle.x
                self.area_angle_spacing.x = max(
                    abs(area1_x - area0_x), self.area_angle_spacing.x)
            if ay1 > ay0:
                area1_y = self.areas[ay1][ax1].angle.y
                area0_y = self.areas[ay0][ax0].angle.y
                self.area_angle_spacing.y = max(
                    abs(area1_y - area0_y), self.area_angle_spacing.y)
    
    def init_areas_center_angle(self):
        if self.split_w % 2 == 1:
            ax1 = ax2 = (self.split_w - 1) // 2
        else:
            ax1, ax2 = self.split_w // 2 - 1, self.split_w // 2
        if self.split_h % 2 == 1:
            ay1 = ay2 = (self.split_h - 1) // 2
        else:
            ay1, ay2 = self.split_h // 2 - 1, self.split_h // 2
        ay1_ax = (self.areas[ay1][ax1].angle.x +
                  self.areas[ay1][ax2].angle.x) / 2
        ay1_ay = (self.areas[ay1][ax1].angle.y +
                  self.areas[ay1][ax2].angle.y) / 2
        ay2_ax = (self.areas[ay2][ax1].angle.x +
                  self.areas[ay2][ax2].angle.x) / 2
        ay2_ay = (self.areas[ay2][ax1].angle.y +
                  self.areas[ay2][ax2].angle.y) / 2
        self.areas_canter_angle.x = (ay1_ax + ay2_ax) / 2
        self.areas_canter_angle.y = (ay1_ay + ay2_ay) / 2
    
    def init_laser(self):
        if IS_DECIDE_ONLY:
            return
        
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.laser_pin, GPIO.OUT, initial=GPIO.LOW)
        if IS_TEST_NEEDED:
            logging.debug("test start")
            for i in range(0):
                time.sleep(2)
                self.open_laser()
                time.sleep(2)
                self.close_laser()
            logging.debug("test done")
    
    def open_laser(self):
        GPIO.output(self.laser_pin, GPIO.HIGH)
    
    def close_laser(self):
        GPIO.output(self.laser_pin, GPIO.LOW)
    
    def check_areas_angle(self):
        for ay in range(self.split_h):
            for ax in range(self.split_w):
                d = self.areas[ay][ax].angle.dict()
                if not d:
                    raise Exception("failed to check areas angle")
                if IS_TEST_NEEDED:
                    n = ay * self.split_w + ax
                    logging.debug("check {}: {}".format(n, d))
                    self.arm.rotate(d, False)
                    self.open_laser()
                    time.sleep(1)
                    for i in range(100):
                        self.sweep_area(ax, ay)
                    self.close_laser()
        self.arm.rotate(self.areas_canter_angle.dict(), False)
    
    def get_arm(self) -> ControllerForPCA9685:
        if IS_DECIDE_ONLY:
            return None
        
        mg995_sec_per_angle = \
            ((0.16 - 0.2) / (6.0 - 4.8) * (5.0 - 4.8) + 0.2) / 60.0
        mg995_pan = Servo(0.0, 180.0, 180.0, 630.0, 50.0, mg995_sec_per_angle)
        mg995_tilt = Servo(0.0, 180.0, 150.0, 510.0, 50.0, mg995_sec_per_angle)
        servos = {X: mg995_pan, Y: mg995_tilt}
        chs = {X: self.servo_x_ch, Y: self.servo_y_ch}
        return ControllerForPCA9685(servos, chs, 60.0)
    
    def get_video_io(self, vi: str, vo: str) -> \
            (cv2.VideoCapture, cv2.VideoWriter):
        video_in = cv2.VideoCapture(0 if not vi else vi)
        video_out = None
        if self.has_video_out:
            if not vo.lower().endswith(".mp4"):
                vo += ".mp4"
            video_out = cv2.VideoWriter(vo, cv2.VideoWriter_fourcc(*"mp4v"),
                                        10, (self.showing_w, self.showing_h))
        return video_in, video_out
    
    def get_cap_img(self, w: int, h: int) -> numpy.ndarray:
        ret, img = self.video_in.read()
        if not ret:
            return None
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    
    def close_video_io(self):
        self.is_terminated = True
        if self.video_in.isOpened():
            self.video_in.release()
        if self.has_video_out and self.video_out.isOpened():
            self.video_out.release()
    
    def sweep_area(self, ax: int, ay: int):
        area_angle = self.areas[ay][ax].angle
        x = random.uniform(
            area_angle.x - self.area_angle_spacing.x / 2,
            area_angle.x + self.area_angle_spacing.x / 2)
        y = random.uniform(
            area_angle.y - self.area_angle_spacing.y / 2,
            area_angle.y + self.area_angle_spacing.y / 2)
        self.arm.rotate({X: x, Y: y}, False)
        time.sleep(1 / 20)
    
    def trans_detections(
            self, detections_raw: List[Tuple[str, float, Tuple[
                float, float, float, float]]]) -> List[Detection]:
        detections = [Detection() for _ in range(len(detections_raw))]
        for i in range(len(detections_raw)):
            r, d = detections_raw[i], detections[i]
            wr = self.showing_w / self.darknet_net_w
            hr = self.showing_h / self.darknet_net_h
            x, y, w, h = r[2][0] * wr, r[2][1] * hr, r[2][2] * wr, r[2][3] * hr
            d.rate = r[1]
            d.x1, d.y1 = int(round(x - (w / 2))), int(round(y - (h / 2)))
            d.x2, d.y2 = int(round(x + (w / 2))), int(round(y + (h / 2)))
            d.cx, d.cy = int(round(x)), int(round(y))
            
            if IS_DETECT_ONLY:
                continue
            dist2 = sys.maxsize
            for ay in range(0, self.split_h):
                for ax in range(0, self.split_w):
                    rect = self.areas[ay][ax].rect
                    d2 = math.pow(rect.cx - x, 2) + math.pow(rect.cy - y, 2)
                    if d2 < dist2:
                        d.ax, d.ay = ax, ay
                        dist2 = d2
        return detections
    
    def draw_text(self, img: numpy.ndarray, text: str, x: int, y: int,
                  size: float, color: Tuple[int, int, int], align: int = 0):
        if not 0 <= align < 9:
            raise OverflowError
        (text_w, text_h), base_line = \
            cv2.getTextSize(text, self.font, size, 1)
        w, h = text_w + base_line / 2, text_h + base_line
        if align % 3 == 0:
            x = int(math.floor(x + base_line / 4))
        elif align % 3 == 1:
            x = int(round(x - w / 2 + 1 / 2 + base_line / 4))
        else:
            x = int(math.ceil(x - w + 1 + base_line / 4))
        if align // 3 == 0:
            y = int(round(y + text_h + base_line / 2 - 1))
        elif align // 3 == 1:
            y = int(round(y - h / 2 + 1 / 2 + text_h + base_line / 2 - 1))
        else:
            y = int(round(y - h + 1 + text_h + base_line / 2 - 1))
        cv2.putText(img, text, (x, y), self.font, size, color, 1, cv2.LINE_AA)
    
    def draw_detections(self, img: numpy.ndarray, detections: List[Detection]):
        color, padding, size = self.others_color, 1, 10
        for d in detections:
            cv2.rectangle(img, (d.x1, d.y1), (d.x2, d.y2), color, 1)
            self.draw_text(img, "{:5.2f}%".format(d.rate * 100),
                           d.cx, d.y1 - 1, 1 / 4, self.others_color, 7)
            
            if IS_DETECT_ONLY:
                continue
            a = self.areas[d.ay][d.ax]
            cx, cy = int(round(a.rect.cx)), int(round(a.rect.cy))
            cv2.arrowedLine(img, (d.cx, d.cy), (cx, cy), color, 1, cv2.LINE_AA)
    
    def draw_fps(self, img: numpy.ndarray, fps: float):
        self.draw_text(img, "FPS:{:05.2f}".format(fps),
                       self.showing_w - 1, 0, 1 / 2, self.others_color, 2)
    
    def draw_areas(self, img: numpy.ndarray,
                   areas: List[List[Area]], is_detecting: bool):
        if areas is None:
            return
        
        for ay in range(self.split_h):
            for ax in range(self.split_w):
                a = areas[ay][ax]
                color = self.detecting_color
                
                if is_detecting and a.status != Status.detecting:
                    if a.status == Status.sweeping:
                        color = self.sweeping_color
                        c = (int(round(a.rect.cx)), int(round(a.rect.cy)))
                        cv2.circle(img, c, 4, color, 1, cv2.LINE_AA)
                        self.draw_text(img, "SWP:{:2d}".format(a.count.sweep),
                                       a.rect.x1 + 1, a.rect.y2 - 1, 1 / 2,
                                       self.count1_color, 6)
                        if a.count.error > 0:
                            self.draw_text(
                                img, "ERR:{:1d}".format(a.count.error),
                                a.rect.x2 - 1, a.rect.y2 - 1, 1 / 2,
                                self.count2_color, 8)
                    else:
                        color = self.detected_color
                        if a.status == Status.abandoning:
                            self.draw_text(
                                img, "ABD:{:2d}".format(a.count.abandon),
                                a.rect.x1 + 1, a.rect.y2 - 1, 1 / 2,
                                self.count1_color, 6)
                        if a.status == Status.confirming:
                            self.draw_text(
                                img, "CNF:{:1d}".format(a.count.confirmation),
                                a.rect.x2 - 1, a.rect.y2 - 1, 1 / 2,
                                self.count2_color, 8)
                
                cv2.rectangle(img, (a.rect.x1, a.rect.y1),
                              (a.rect.x2, a.rect.y2), color, 1)
                self.draw_text(img, "{:02d}".format(ay * self.split_w + ax),
                               a.rect.x1 + 1, a.rect.y1 + 1, 1, color, 0)
                if not is_detecting:
                    cv2.circle(
                        img, (int(round(a.rect.cx)), int(round(a.rect.cy))),
                        4, self.sweeping_color, 1, cv2.LINE_AA)
    
    def copy_areas(self) -> List[List[Area]]:
        copied_areas = self.make_areas()
        for ay in range(self.split_h):
            for ax in range(self.split_w):
                a, ca = self.areas[ay][ax], copied_areas[ay][ax]
                ca.rect = copy(a.rect)
                ca.angle = copy(a.angle)
                ca.status = a.status
                ca.count = copy(a.count)
        return copied_areas
    
    def loop_detecting(self):
        self.is_started_detecting = True
        
        while not self.is_terminated:
            begin_time = time.time()
            img = self.get_cap_img(self.darknet_net_w, self.darknet_net_h)
            if img is None:
                self.close_video_io()
                return
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            darknet.copy_image_from_bytes(self.darknet_img, img.tobytes())
            detections_raw = darknet.detect_image(
                self.darknet_net, self.darknet_meta,
                self.darknet_img, thresh=0.3)
            detections = self.trans_detections(detections_raw)
            
            fps = 1 / (time.time() - begin_time)
            try:
                if IS_DETECT_ONLY:
                    self.que_showing.put((None, detections, fps), timeout=1)
                else:
                    self.que_deciding.put((detections, fps), timeout=1)
            except (QueueEmpty, QueueFull) as e:
                if self.is_terminated:
                    return
                raise e
    
    def thd_deciding_func(self):
        while not self.is_terminated:
            begin_time = time.time()
            try:
                detections, fps = self.que_deciding.get(timeout=1)
            except (QueueEmpty, QueueFull) as e:
                if self.is_terminated:
                    return
                raise e
            detected_areas = set()
            for d in detections:
                detected_areas.add((d.ax, d.ay))
            
            for ay in range(self.split_h):
                for ax in range(self.split_w):
                    a = self.areas[ay][ax]
                    if (ax, ay) in detected_areas:
                        if a.status == Status.detecting:
                            a.count.clear()
                            a.status = Status.confirming
                            a.count.confirmation = 1
                        elif a.status == Status.confirming:
                            a.count.confirmation += 1
                            if a.count.confirmation > Limit.confirmation:
                                a.count.clear()
                                a.status = Status.detected
                        elif a.status == Status.sweeping:
                            if a.count.error > 0:
                                a.count.error = 0
                            a.count.sweep += 1
                            if a.count.sweep > Limit.sweep:
                                a.count.clear()
                                a.status = Status.abandoning
                        elif a.status == Status.abandoning:
                            a.count.abandon += 1
                            if a.count.abandon > Limit.abandon:
                                a.count.clear()
                                a.status = Status.detected
                    else:
                        if a.status == Status.sweeping:
                            a.count.error += 1
                            if a.count.error > Limit.error:
                                a.count.clear()
                                a.status = Status.detecting
                        elif a.status != Status.detecting:
                            a.count.clear()
                            a.status = Status.detecting
            
            sweeping_ax, sweeping_ay, sweeping_count, ai = -1, -1, 0, 0
            candidate_areas = [(-1, -1)] * self.split_w * self.split_h
            for ay in range(self.split_h):
                for ax in range(self.split_w):
                    a = self.areas[ay][ax]
                    if a.status == Status.sweeping:
                        sweeping_ax, sweeping_ay = ax, ay
                        sweeping_count += 1
                    elif a.status == Status.detected:
                        candidate_areas[ai] = (ax, ay)
                        ai += 1
            if sweeping_count > 1:
                logging.error("sweeping status must be 1")
                exit(1)
            elif sweeping_count == 0 and ai > 0:
                ax, ay = random.choice(candidate_areas[:ai])
                a = self.areas[ay][ax]
                a.count.clear()
                a.status = Status.sweeping
                a.count.sweep = 1
                sweeping_ax, sweeping_ay = ax, ay
            
            logging.info("use {:.3}s".format(time.time() - begin_time))
            try:
                if not IS_DECIDE_ONLY:
                    self.que_sweeping.put(
                        (sweeping_ax, sweeping_ay), timeout=1)
                self.que_showing.put(
                    (self.copy_areas(), detections, fps), timeout=1)
            except (QueueEmpty, QueueFull) as e:
                if self.is_terminated:
                    return
                raise e
    
    def thd_sweeping_func(self):
        sweeping_ax, sweeping_ay = -1, -1
        
        while not self.is_terminated:
            if self.que_sweeping.full():
                ax, ay = self.que_sweeping.get(timeout=1)
                if sweeping_ax != ax or sweeping_ay != ay:
                    self.close_laser()
                    if ax >= 0 and ay >= 0:
                        self.arm.rotate(self.areas[ay][ax].angle.dict(), False)
                    else:
                        self.arm.rotate(self.areas_canter_angle.dict(), False)
                    sweeping_ax, sweeping_ay = ax, ay
            if sweeping_ax >= 0 and sweeping_ay >= 0:
                self.open_laser()
                self.sweep_area(sweeping_ax, sweeping_ay)
            else:
                time.sleep(1 / 20)  # avoid high CPU usage
    
    def thd_showing_func(self):
        areas, detections, fps = self.areas, None, 10.0
        
        while not self.is_terminated:
            begin_time = time.time()
            img = self.get_cap_img(self.showing_w, self.showing_h)
            if img is None:
                self.close_video_io()
                return
            if self.que_showing.full():
                areas, detections, fps = self.que_showing.get(timeout=1)
            if self.is_started_detecting:
                if detections is not None:
                    self.draw_detections(img, detections)
                    self.draw_fps(img, fps)
                self.draw_areas(img, areas, True)
            else:
                self.draw_areas(img, areas, False)
            if self.has_video_out:
                self.video_out.write(img)
            cv2.imshow("image", img)
            cv2.waitKey(1)
            
            logging.info("use {:.3}s".format(time.time() - begin_time))
            if self.has_video_in:
                time.sleep(1 / fps)


def main():
    random.seed()
    logging.basicConfig(
        stream=sys.stdout, level=logging.ERROR, datefmt="%H:%M:%S",
        format="%(asctime)s.%(msecs)03d | %(levelname)-5s | "
               "%(threadName)12s -> %(funcName)s: %(message)s")
    
    p = ArgumentParser(prog="sudo python3 main.py")
    p.add_argument("-vi", "--video-input", dest="vi", type=str, default="",
                   help="input video's path, ignore it to using videostream")
    p.add_argument("-vo", "--video-output", dest="vo", type=str, default="",
                   help="output mp4 video's path, ignore it when no output")
    p.add_argument("-sw", "--spilt-width", dest="sw", type=int, default=4,
                   help="spilt width of sweeping areas")
    p.add_argument("-sh", "--spilt-height", dest="sh", type=int, default=3,
                   help="spilt height of sweeping areas")
    args = p.parse_args()
    
    KeepPigeonsAway(args.vi, args.vo, args.sw, args.sh)


if __name__ == '__main__':
    main()
