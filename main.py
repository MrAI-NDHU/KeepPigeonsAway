from copy import copy
from enum import auto, IntEnum
from queue import Queue
from threading import Thread
from typing import Any, Dict, List, Tuple
import logging
import math
import numpy
# import os
import pickle
import random
# import select
import sys
import time
import termios

from Jetson import GPIO
import cv2
import keyboard

from servo import Servo
from servo.controller import ControllerForPCA9685
import darknet

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


class Status(IntEnum):
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


T_COPY_RECT = type("T_COPY_RECT", Rect.__bases__, dict(Rect.__dict__))
T_COPY_ANGLE = type("T_COPY_ANGLE", Angle.__bases__, dict(Angle.__dict__))
T_COPY_COUNT = type("T_COPY_COUNT", Count.__bases__, dict(Count.__dict__))


class Area:
    rect = Rect()
    angle = Angle()
    status = Status.detecting
    count = Count()


class Detection:
    x1, x2, y1, y2, cx, cy, ax, ay, r = -1, -1, -1, -1, -1, -1, -1, -1, 0.0


class DriveAwayPigeons:
    
    def __init__(self, split_w: int, split_h: int):
        self.video_path = ""
        self.split_w, self.split_h = split_w, split_h
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
        self.cap = \
            cv2.VideoCapture(0 if not self.video_path else self.video_path)
        self.thd_showing.start()
        
        self.area_angle_spacing = Angle()
        self.areas_canter_angle = Angle()
        self.init_areas_angle()
        
        self.darknet_net, self.darknet_net_w, self.darknet_net_h = None, 0, 0
        self.darknet_meta, self.darknet_img = None, None
        self.init_darknet()
        
        self.thd_deciding.start()
        self.thd_sweeping.start()
        self.loop_detecting()
    
    def __del__(self):
        GPIO.cleanup()
    
    def make_areas(self) -> List[List[Area]]:
        return [[Area()] * self.split_w for _ in range(self.split_h)]
    
    def init_darknet(self):
        config_path = "./cfg/yolov3-tiny.cfg"
        weight_path = "./yolov3-tiny_pigeon.weights"
        meta_path = "./cfg/pigeon.data"
        self.darknet_net = darknet.load_net_custom(config_path.encode(
            "ascii"), weight_path.encode("ascii"), 0, 1)
        self.darknet_net_w = darknet.network_width(self.darknet_net)
        self.darknet_net_h = darknet.network_height(self.darknet_net)
        self.darknet_meta = darknet.load_meta(meta_path.encode("ascii"))
        self.darknet_img = darknet.make_image(
            self.darknet_net_w, self.darknet_net_h, 3)
    
    def init_areas_rect(self):
        aw, ah = self.showing_w / self.split_w, self.showing_h / self.split_h
        for ay in range(self.split_h):
            for ax in range(self.split_w):
                x1, y1 = aw * ax, ah * ay
                x2, y2 = aw * (ax + 1), ah * (ay + 1)
                self.areas[ay][ax].rect.x1 = int(round(x1))
                self.areas[ay][ax].rect.y1 = int(round(y1))
                self.areas[ay][ax].rect.x2 = int(round(x2)) - 1
                self.areas[ay][ax].rect.y2 = int(round(y2)) - 1
                self.areas[ay][ax].rect.cx = (x1 + x2 - 1) / 2
                self.areas[ay][ax].rect.cy = (y1 + y2 - 1) / 2
    
    def init_areas_angle(self):
        if TEST_DETECT_ONLY:
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
                            i, is_ok = int(s), True
                    self.areas[i // self.split_w][i % self.split_w].angle.x = \
                        self.arm.current_angles[X]
                    self.areas[i // self.split_w][i % self.split_w].angle.y = \
                        self.arm.current_angles[Y]
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
        if TEST_DETECT_ONLY:
            return
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
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.laser_pin, GPIO.OUT, initial=GPIO.LOW)
        if TEST_DEVICE:
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
                if TEST_DEVICE:
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
        mg995_sec_per_angle = \
            ((0.16 - 0.2) / (6.0 - 4.8) * (5.0 - 4.8) + 0.2) / 60.0
        mg995_pan = Servo(0.0, 180.0, 180.0, 630.0, 50.0, mg995_sec_per_angle)
        mg995_tilt = Servo(0.0, 180.0, 150.0, 510.0, 50.0, mg995_sec_per_angle)
        servos = {X: mg995_pan, Y: mg995_tilt}
        chs = {X: self.servo_x_ch, Y: self.servo_y_ch}
        return ControllerForPCA9685(servos, chs, 60.0)
    
    def get_cap_img(self, w: int, h: int) -> numpy.ndarray:
        _, img = self.cap.read()
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    
    def sweep_area(self, ax: int, ay: int):
        if TEST_DETECT_ONLY:
            time.sleep(1 / 20)
            return
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
            d = detections_raw[i]
            detections[i].r = d[1]
            wr = self.showing_w / self.darknet_net_w
            hr = self.showing_h / self.darknet_net_h
            x, y, w, h = d[2][0] * wr, d[2][1] * hr, d[2][2] * wr, d[2][3] * hr
            detections[i].x1 = int(round(x - (w / 2)))
            detections[i].y1 = int(round(y - (h / 2)))
            detections[i].x2 = int(round(x + (w / 2)))
            detections[i].y2 = int(round(y + (h / 2)))
            detections[i].cx = int(round(x))
            detections[i].cy = int(round(y))
            dist = sys.maxsize
            for ay in range(0, self.split_h):
                for ax in range(0, self.split_w):
                    rect = self.areas[ay][ax].rect
                    d2 = math.pow(rect.cx - x, 2) + math.pow(rect.cy - y, 2)
                    if d2 < dist:
                        detections[i].ax = ax
                        detections[i].ay = ay
                        dist = d2
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
            cx = int(round(self.areas[d.ay][d.ax].rect.cx))
            cy = int(round(self.areas[d.ay][d.ax].rect.cy))
            cv2.line(img, (d.cx, d.cy), (cx, cy), color, 1, cv2.LINE_AA)
            self.draw_text(img, "{:5.2f}%".format(d.r * 100),
                           d.cx, d.y1 - 1, 1 / 4, self.others_color, 7)
    
    def draw_fps(self, img: numpy.ndarray, fps: float):
        self.draw_text(img, "FPS:{:05.2f}".format(fps),
                       self.showing_w - 1, 0, 1 / 2, self.others_color, 2)
    
    def draw_areas(self, img: numpy.ndarray,
                   areas: List[List[Area]], is_detecting: bool):
        for ay in range(self.split_h):
            for ax in range(self.split_w):
                i_str = str(ay * self.split_w + ax)
                rect = self.areas[ay][ax].rect
                color = self.detecting_color
                if is_detecting and areas[ay][ax].status != Status.detecting:
                    if areas[ay][ax].status == Status.sweeping:
                        color = self.sweeping_color
                        c = (int(round(rect.cx)), int(round(rect.cy)))
                        cv2.circle(img, c, 4, color, 1, cv2.LINE_AA)
                        self.draw_text(
                            img, "SWP:{:2d}".format(areas[ay][ax].count.sweep),
                            rect.x1 + 1, rect.y2 - 1, 1 / 2,
                            self.count1_color, 6)
                        if areas[ay][ax].count.error > 0:
                            self.draw_text(
                                img, "ERR:{:1d}".format(
                                    areas[ay][ax].count.error),
                                rect.x2 - 1, rect.y2 - 1, 1 / 2,
                                self.count2_color, 8)
                    else:
                        color = self.detected_color
                        if areas[ay][ax].status == Status.abandoning:
                            self.draw_text(
                                img, "ABD:{:2d}".format(
                                    areas[ay][ax].count.abandon),
                                rect.x1 + 1, rect.y2 - 1, 1 / 2,
                                self.count1_color, 6)
                        if areas[ay][ax].status == Status.confirming:
                            self.draw_text(
                                img, "CNF:{:1d}".format(
                                    areas[ay][ax].count.confirmation),
                                rect.x2 - 1, rect.y2 - 1, 1 / 2,
                                self.count2_color, 8)
                cv2.rectangle(img, (rect.x1, rect.y1),
                              (rect.x2, rect.y2), color, 1)
                self.draw_text(
                    img, i_str, rect.x1 + 1, rect.y1 + 1, 1, color, 0)
                if not is_detecting:
                    cv2.circle(img, (int(round(rect.cx)), int(round(rect.cy))),
                               4, self.sweeping_color, 1, cv2.LINE_AA)
    
    def copy_areas(self) -> List[List[Area]]:
        copied_areas = self.make_areas()
        for ay in range(self.split_h):
            for ax in range(self.split_w):
                copied_areas[ay][ax].rect = copy(self.areas[ay][ax].rect)
                copied_areas[ay][ax].angle = copy(self.areas[ay][ax].angle)
                copied_areas[ay][ax].status = self.areas[ay][ax].status
                copied_areas[ay][ax].count = copy(self.areas[ay][ax].count)
        return copied_areas
    
    def loop_detecting(self):
        self.is_started_detecting = True
        while True:
            begin_time = time.time()
            img = self.get_cap_img(self.darknet_net_w, self.darknet_net_h)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            darknet.copy_image_from_bytes(self.darknet_img, img.tobytes())
            detections_raw = darknet.detect_image(
                self.darknet_net, self.darknet_meta,
                self.darknet_img, thresh=0.3)
            fps = 1 / (time.time() - begin_time)
            self.que_deciding.put((detections_raw, fps))
    
    def thd_deciding_func(self):  # TODO
        sweeping_ax, sweeping_ay, abandoning_ax, abandoning_ay = -1, -1, -1, -1
        while True:
            begin_time = time.time()
            detections_raw, fps = self.que_deciding.get()
            detections = self.trans_detections(detections_raw)
            can_areas, areas = set(), set()
            for d in detections:
                can_areas.add((d.ax, d.ay))
            
            def random_choice(old_ax: int = -1,
                              old_ay: int = -1) -> (int, int):
                candidate = list(areas)
                if 0 < self.areas[abandoning_ay][abandoning_ax].count.sweep \
                        <= Limit.abandon:
                    old_ax, old_ay = abandoning_ax, abandoning_ay
                if len(candidate) > 0 and (old_ax, old_ay) in candidate:
                    candidate.remove((old_ax, old_ay))
                if len(candidate) == 0:
                    return -1, -1
                new_ax, new_ay = random.choice(candidate)
                self.areas[new_ay][new_ax].status = Status.sweeping
                self.areas[new_ay][new_ax].count.sweep = 1
                self.areas[new_ay][new_ax].count.error = 0
                return new_ax, new_ay
            
            if sweeping_ax < 0 or sweeping_ay < 0:
                sweeping_ax, sweeping_ay = random_choice()
                self.areas[abandoning_ay][abandoning_ax].count.sweep += 1
                if sweeping_ax >= 0 and sweeping_ay >= 0:
                    abandoning_ax, abandoning_ay = -1, -1
            else:
                if (sweeping_ax, sweeping_ay) in areas:
                    if self.areas_errors[sweeping_ay][sweeping_ax] > 0:
                        self.areas_errors[sweeping_ay][sweeping_ax] = 0
                    self.areas_sweeps[sweeping_ay][sweeping_ax] += 1
                    if self.areas_sweeps[sweeping_ay][sweeping_ax] > \
                            self.sweeps_limit:
                        ax, ay = sweeping_ax, sweeping_ay
                        sweeping_ax, sweeping_ay = \
                            random_choice(sweeping_ax, sweeping_ay)
                        if sweeping_ax < 0 or sweeping_ay < 0:
                            self.areas_status[ay][ax] = A
                            self.areas_sweeps[ay][ax] = 1
                            abandoning_ax, abandoning_ay = ax, ay
                else:
                    self.areas_errors[sweeping_ay][sweeping_ax] += 1
                    if self.areas_errors[sweeping_ay][sweeping_ax] > \
                            self.errors_limit:
                        sweeping_ax, sweeping_ay = random_choice()
            for ay in range(self.split_h):
                for ax in range(self.split_w):
                    if (ax != sweeping_ax or ay != sweeping_ay) and \
                            (ax != abandoning_ax or ay != abandoning_ay):
                        if (ax, ay) in areas:
                            self.areas_status[ay][ax] = D
                        else:
                            self.areas_status[ay][ax] = N
                        self.areas_sweeps[ay][ax] = 0
                        self.areas_errors[ay][ax] = 0
            logging.info("use {:.3}s".format(time.time() - begin_time))
            self.que_sweeping.put((sweeping_ax, sweeping_ay))
            self.que_showing.put((self.copy_areas(), detections, fps))
    
    def thd_sweeping_func(self):
        sweeping_ax, sweeping_ay = -1, -1
        while True:
            if not self.que_sweeping.empty():
                ax, ay = self.que_sweeping.get()
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
        while True:
            begin_time = time.time()
            img = self.get_cap_img(self.showing_w, self.showing_h)
            if not self.que_showing.empty():
                areas, detections, fps = self.que_showing.get()
            if self.is_started_detecting:
                if detections is not None:
                    self.draw_detections(img, detections)
                    self.draw_fps(img, fps)
                self.draw_areas(img, areas, True)
            else:
                self.draw_areas(img, areas, False)
            cv2.imshow("image", img)
            cv2.waitKey(1)
            logging.info("use {:.3}s".format(time.time() - begin_time))
            if self.video_path != "":
                time.sleep(1 / fps)


TEST_DEVICE = True
TEST_DETECT_ONLY = False


def main():
    random.seed()
    logging.basicConfig(
        stream=sys.stdout, level=logging.ERROR, datefmt="%H:%M:%S",
        format="%(asctime)s.%(msecs)03d | %(levelname)-5s | "
               "%(threadName)12s -> %(funcName)s: %(message)s")
    
    split_w, split_h = 4, 3
    if len(sys.argv) == 2:
        split_w, split_h = int(sys.argv[1]), int(sys.argv[1])
    elif len(sys.argv) >= 3:
        split_w, split_h = int(sys.argv[1]), int(sys.argv[2])
    d = None
    try:
        d = DriveAwayPigeons(split_w, split_h)
    except KeyboardInterrupt:
        del d


if __name__ == '__main__':
    main()
