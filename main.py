from queue import Queue
from threading import Thread
from typing import Any, Dict, TypeVar, Tuple
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

Number = TypeVar("Number", int, float)

X, Y = "x", "y"
X1, Y1, X2, Y2 = "x1", "y1", "x2", "y2",
CX, CY, AX, AY, R = "cx", "cy", "ax", "ay", "r"
N, D, S = 0, 1, 2


class DriveAwayPigeons:
    
    def __init__(self, split_w: int, split_h: int):
        self.split_w, self.split_h = split_w, split_h
        self.angle_prec = 1 / 40
        self.laser_pin, self.servo_x_ch, self.servo_y_ch = 18, 1, 0
        self.sweeps_limit, self.errors_limit, self.abandons_limit = 36, 6, 36
        self.cap_ratio = 1920 / 1080
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.showing_w, self.showing_h = 854, 480
        self.area_normal_color = (0x00, 0xFF, 0x00)  # green
        self.area_detected_color = (0xFF, 0x00, 0x00)  # blue
        self.area_sweeping_color = (0xFF, 0x00, 0xFF)  # purple
        self.area_error_color = (0x00, 0x00, 0xFF)  # red
        self.detection_color = (0xFF, 0xFF, 0x00)  # cyan
        self.others_color = (0x00, 0xFF, 0xFF)  # yellow
        
        self.areas_rect = self.get_areas_rect()
        self.areas_status = self.make_areas_int(N)
        self.areas_sweeps = self.make_areas_int(0)
        self.areas_errors = self.make_areas_int(0)
        self.arm = self.get_arm()
        self.init_laser()
        
        self.thd_detecting = Thread(target=self.thd_detecting_func)
        self.thd_deciding = Thread(target=self.thd_deciding_func)
        self.thd_sweeping = Thread(target=self.thd_sweeping_func)
        self.thd_showing = Thread(target=self.thd_showing_func)
        self.que_deciding = Queue(1)
        self.que_sweeping = Queue(1)
        self.que_showing = Queue(1)
        
        self.is_started_detecting = False
        self.cap = cv2.VideoCapture(0)
        self.thd_showing.start()
        
        self.areas_angle = [[{}]]
        self.area_angle_spacing = {}
        self.init_areas_angle()
        
        self.darknet_net, self.darknet_net_w, self.darknet_net_h = None, 0, 0
        self.darknet_meta, self.darknet_img = None, None
        self.init_darknet()
        
        self.thd_detecting.start()
        self.thd_deciding.start()
        self.thd_sweeping.start()
        
        self.thd_detecting.join()
    
    def __del__(self):
        GPIO.cleanup()
    
    def make_areas_int(self, n: int):
        return [[n] * self.split_w for _ in range(self.split_h)]
    
    def make_areas_dict(self):
        return [[{} for _ in range(self.split_w)] for _ in range(self.split_h)]
    
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
    
    def init_laser(self):
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.laser_pin, GPIO.OUT, initial=GPIO.LOW)
        if TEST_MODE:
            logging.info("init_laser: test start")
            for i in range(0):
                time.sleep(2)
                self.open_laser()
                time.sleep(2)
                self.close_laser()
            logging.info("init_laser: test finish")
    
    def open_laser(self):
        GPIO.output(self.laser_pin, GPIO.HIGH)
    
    def close_laser(self):
        GPIO.output(self.laser_pin, GPIO.LOW)
    
    def get_arm(self) -> ControllerForPCA9685:
        mg995_sec_per_angle = \
            ((0.16 - 0.2) / (6.0 - 4.8) * (5.0 - 4.8) + 0.2) / 60.0
        mg995_pan = Servo(0.0, 180.0, 180.0, 630.0, 50.0, mg995_sec_per_angle)
        mg995_tilt = Servo(0.0, 180.0, 150.0, 510.0, 50.0, mg995_sec_per_angle)
        servos = {X: mg995_pan, Y: mg995_tilt}
        chs = {X: self.servo_x_ch, Y: self.servo_y_ch}
        return ControllerForPCA9685(servos, chs, 60.0)
    
    def get_areas_rect(self) -> [[Dict[str, Number]]]:
        areas_rect = self.make_areas_dict()
        aw, ah = self.showing_w / self.split_w, self.showing_h / self.split_h
        for ay in range(self.split_h):
            for ax in range(self.split_w):
                x1, y1 = aw * ax, ah * ay
                x2, y2 = aw * (ax + 1), ah * (ay + 1)
                areas_rect[ay][ax][X1] = int(round(x1))
                areas_rect[ay][ax][Y1] = int(round(y1))
                areas_rect[ay][ax][X2] = int(round(x2)) - 1
                areas_rect[ay][ax][Y2] = int(round(y2)) - 1
                areas_rect[ay][ax][CX] = (x1 + x2) / 2
                areas_rect[ay][ax][CY] = (y1 + y2) / 2
        return areas_rect
    
    def check_areas_angle(self):
        for ay in range(self.split_h):
            for ax in range(self.split_w):
                if self.areas_angle[ay][ax] is None:
                    raise Exception("failed to check areas angle")
                if TEST_MODE:
                    n = ay * self.split_w + ax
                    logging.info("check_areas_angle: check {}: {}"
                                 .format(n, self.areas_angle[ay][ax]))
                    self.arm.rotate(self.areas_angle[ay][ax], False)
                    self.open_laser()
                    time.sleep(1)
                    for i in range(100):
                        self.sweep_area(ax, ay)
                    self.close_laser()
    
    def init_areas_angle(self):
        self.areas_angle = self.make_areas_dict()
        if TEST_DETECT_ONLY:
            return
        flush_stdin()
        filepath = input("Load areas angle: ").strip()
        if filepath != "":
            with open(filepath, "rb") as f:
                self.areas_angle = pickle.load(f)
            self.init_sweep_attrs()
            self.check_areas_angle()
        else:
            self.open_laser()
            while True:
                if keyboard.is_pressed("e"):
                    break
                elif keyboard.is_pressed("up"):
                    self.arm.rotate({Y: -self.angle_prec}, True)
                elif keyboard.is_pressed("down"):
                    self.arm.rotate({Y: self.angle_prec}, True)
                elif keyboard.is_pressed("left"):
                    self.arm.rotate({X: self.angle_prec}, True)
                elif keyboard.is_pressed("right"):
                    self.arm.rotate({X: -self.angle_prec}, True)
                elif keyboard.is_pressed("s"):
                    flush_stdin()
                    i = int(input("Input area number in range [{},{}]: "
                                  .format(0, self.split_w * self.split_h - 1)))
                    self.areas_angle[i // self.split_w][i % self.split_w] = \
                        self.arm.current_angles.copy()
            self.close_laser()
            self.init_sweep_attrs()
            self.check_areas_angle()
            flush_stdin()
            filepath = input("Save areas angle: ").strip()
            if filepath != "":
                with open(filepath, "wb") as f:
                    pickle.dump(self.areas_angle, f)
    
    def init_sweep_attrs(self):
        self.area_angle_spacing = {X: 0.0, Y: 0.0}
        if TEST_DETECT_ONLY:
            return
        for i in range(1, self.split_w * self.split_h):
            ax1, ay1 = i % self.split_w, i // self.split_w
            ax0, ay0 = (i - 1) % self.split_w, (i - 1) // self.split_w
            if ax1 > ax0:
                area1_x = self.areas_angle[ay1][ax1][X]
                area0_x = self.areas_angle[ay0][ax0][X]
                self.area_angle_spacing[X] = max(
                    abs(area1_x - area0_x), self.area_angle_spacing[X])
            if ay1 > ay0:
                area1_y = self.areas_angle[ay1][ax1][Y]
                area0_y = self.areas_angle[ay0][ax0][Y]
                self.area_angle_spacing[Y] = max(
                    abs(area1_y - area0_y), self.area_angle_spacing[Y])
    
    def sweep_area(self, ax: int, ay: int):
        if TEST_DETECT_ONLY:
            time.sleep(0.05)
            return
        area_angle = self.areas_angle[ay][ax]
        x = random.uniform(
            area_angle[X] - self.area_angle_spacing[X] / 2,
            area_angle[X] + self.area_angle_spacing[X] / 2)
        y = random.uniform(
            area_angle[Y] - self.area_angle_spacing[Y] / 2,
            area_angle[Y] + self.area_angle_spacing[Y] / 2)
        self.arm.rotate({X: x, Y: y}, False)
        time.sleep(0.05)
    
    def get_cap_img(self, w, h) -> numpy.ndarray:
        _, img = self.cap.read()
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    
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
    
    def draw_areas(self, img: numpy.ndarray, areas_status: [[int]] = None,
                   areas_sweeps: [[int]] = None, areas_errors: [[int]] = None,
                   has_canter: bool = False):
        for ay in range(self.split_h):
            for ax in range(self.split_w):
                i_str = str(ay * self.split_w + ax)
                rect = self.areas_rect[ay][ax]
                if areas_status is None:
                    areas_status = self.areas_status
                if areas_status[ay][ax] == N:
                    color = self.area_normal_color
                else:
                    color = self.area_detected_color
                    if areas_status[ay][ax] == S:
                        color = self.area_sweeping_color
                        c = (int(round(rect[CX])), int(round(rect[CY])))
                        cv2.circle(img, c, 4, color, 1, cv2.LINE_AA)
                        self.draw_text(
                            img, "{:2d}".format(areas_sweeps[ay][ax]),
                            rect[X1] + 1, rect[Y2] - 1, 1 / 2, color, 6)
                        if areas_errors[ay][ax] > 0:
                            color = self.area_error_color
                            self.draw_text(
                                img, "{:2d}".format(areas_errors[ay][ax]),
                                rect[X2] - 1, rect[Y2] - 1, 1 / 2, color, 8)
                cv2.rectangle(img, (rect[X1], rect[Y1]),
                              (rect[X2], rect[Y2]), color, 1)
                self.draw_text(
                    img, i_str, rect[X1] + 1, rect[Y1] + 1, 1, color, 0)
                if has_canter:
                    c = (int(round(rect[CX])), int(round(rect[CY])))
                    cv2.circle(
                        img, c, 4, self.area_sweeping_color, 1, cv2.LINE_AA)
    
    def draw_fps(self, img: numpy.ndarray, fps: float):
        self.draw_text(img, "FPS:{:6.2f}".format(fps),
                       self.showing_w - 1, 0, 1 / 2, self.others_color, 2)
    
    def draw_detections(self, img: numpy.ndarray,
                        detections: [Dict[str, Number]]):
        color, padding, size = self.detection_color, 1, 10
        for d in detections:
            cv2.rectangle(img, (d[X1], d[Y1]), (d[X2], d[Y2]), color, 1)
            cx = int(round(self.areas_rect[d[AY]][d[AX]][CX]))
            cy = int(round(self.areas_rect[d[AY]][d[AX]][CY]))
            cv2.line(img, (d[CX], d[CY]), (cx, cy), color, 1, cv2.LINE_AA)
            self.draw_text(img, "{:6.2f}%".format(d[R] * 100),
                           d[CX], d[Y1] - 1, 1 / 4, self.detection_color, 7)
    
    def trans_detections(self,
                         detections_raw: [[[Any]]]) -> [Dict[str, Number]]:
        detections = [{} for _ in range(len(detections_raw))]
        for i in range(len(detections_raw)):
            d = detections_raw[i]
            detections[i][R] = d[1]
            wr = self.showing_w / self.darknet_net_w
            hr = self.showing_h / self.darknet_net_h
            x, y, w, h = d[2][0] * wr, d[2][1] * hr, d[2][2] * wr, d[2][3] * hr
            detections[i][X1] = int(round(x - (w / 2)))
            detections[i][Y1] = int(round(y - (h / 2)))
            detections[i][X2] = int(round(x + (w / 2)))
            detections[i][Y2] = int(round(y + (h / 2)))
            detections[i][CX] = int(round(x))
            detections[i][CY] = int(round(y))
            dist = sys.maxsize
            for ay in range(0, self.split_h):
                for ax in range(0, self.split_w):
                    rect = self.areas_rect[ay][ax]
                    d2 = math.pow(rect[CX] - x, 2) + math.pow(rect[CY] - y, 2)
                    if d2 < dist:
                        detections[i][AX] = ax
                        detections[i][AY] = ay
                        dist = d2
        return detections
    
    def thd_detecting_func(self):
        self.is_started_detecting = True
        while True:
            begin_time = time.time()
            img = self.get_cap_img(self.darknet_net_w, self.darknet_net_h)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            darknet.copy_image_from_bytes(self.darknet_img, img.tobytes())
            detections_raw = darknet.detect_image(
                self.darknet_net, self.darknet_meta,
                self.darknet_img, thresh=0.25)
            fps = 1 / (time.time() - begin_time)
            self.que_deciding.put((detections_raw, fps))
    
    def thd_deciding_func(self):
        sweeping_ax, sweeping_ay = -1, -1
        abandoning_ax, abandoning_ay, abandons = -1, -1, 0
        while True:
            detections_raw, fps = self.que_deciding.get()
            detections = self.trans_detections(detections_raw)
            areas = set()
            for d in detections:
                areas.add((d[AX], d[AY]))
            
            def random_choice(old_ax: int = -1,
                              old_ay: int = -1) -> (int, int):
                candidate = list(areas)
                if 0 < abandons <= self.abandons_limit:
                    old_ax, old_ay = abandoning_ax, abandoning_ay
                if len(candidate) > 0 and (old_ax, old_ay) in candidate:
                    candidate.remove((old_ax, old_ay))
                if len(candidate) == 0:
                    return -1, -1
                new_ax, new_ay = random.choice(candidate)
                self.areas_status[new_ay][new_ax] = S
                self.areas_sweeps[new_ay][new_ax] = 1
                self.areas_errors[new_ay][new_ax] = 0
                return new_ax, new_ay
            
            if sweeping_ax < 0 or sweeping_ay < 0:
                sweeping_ax, sweeping_ay = random_choice()
                abandons += 1
                if sweeping_ax >= 0 and sweeping_ay >= 0:
                    abandoning_ax, abandoning_ay, abandons = -1, -1, 0
            else:
                if (sweeping_ax, sweeping_ay) in areas:
                    if self.areas_errors[sweeping_ay][sweeping_ax] > 0:
                        self.areas_errors[sweeping_ay][sweeping_ax] = 0
                    self.areas_sweeps[sweeping_ay][sweeping_ax] += 1
                    if self.areas_sweeps[sweeping_ay][sweeping_ax] > \
                            self.sweeps_limit:
                        abandoning_ax, abandoning_ay = sweeping_ax, sweeping_ay
                        sweeping_ax, sweeping_ay = \
                            random_choice(sweeping_ax, sweeping_ay)
                        if sweeping_ax < 0 or sweeping_ay < 0:
                            abandons = 1
                else:
                    self.areas_errors[sweeping_ay][sweeping_ax] += 1
                    if self.areas_errors[sweeping_ay][sweeping_ax] > \
                            self.errors_limit:
                        sweeping_ax, sweeping_ay = random_choice()
            for ay in range(self.split_h):
                for ax in range(self.split_w):
                    if ax != sweeping_ax or ay != sweeping_ay:
                        if (ax, ay) in areas:
                            self.areas_status[ay][ax] = D
                        else:
                            self.areas_status[ay][ax] = N
                        self.areas_sweeps[ay][ax] = 0
                        self.areas_errors[ay][ax] = 0
            self.que_sweeping.put((sweeping_ax, sweeping_ay))
            self.que_showing.put(
                (self.areas_status.copy(), self.areas_sweeps.copy(),
                 self.areas_errors.copy(), detections, fps))
    
    def thd_sweeping_func(self):
        sweeping_ax, sweeping_ay = -1, -1
        while True:
            if not self.que_sweeping.empty():
                ax, ay = self.que_sweeping.get()
                if sweeping_ax != ax or sweeping_ay != ay:
                    self.close_laser()
                    if ax >= 0 and ay >= 0:
                        self.arm.rotate(self.areas_angle[ay][ax], False)
                    else:
                        self.arm.rotate({X: 90.0, Y: 90.0}, False)
                    sweeping_ax, sweeping_ay = ax, ay
            if sweeping_ax >= 0 and sweeping_ay >= 0:
                self.open_laser()
                self.sweep_area(sweeping_ax, sweeping_ay)
            # else:
            #     time.sleep(1 / 60)  # avoid high CPU usage
            print(sweeping_ax, sweeping_ay)
    
    def thd_showing_func(self):
        areas_status, areas_sweeps, areas_errors, detections, fps = \
            self.areas_status, self.areas_sweeps, self.areas_errors, None, 0.0
        while True:
            img = self.get_cap_img(self.showing_w, self.showing_h)
            if not self.que_showing.empty():
                areas_status, areas_sweeps, areas_errors, detections, fps = \
                    self.que_showing.get()
            if self.is_started_detecting:
                if detections is not None:
                    self.draw_detections(img, detections)
                    self.draw_fps(img, fps)
                self.draw_areas(img, areas_status, areas_sweeps, areas_errors)
            else:
                self.draw_areas(img, has_canter=True)
            cv2.imshow("image", img)
            cv2.waitKey(1)
            # time.sleep(1 / 60)  # avoid high CPU usage


TEST_MODE = True
TEST_DETECT_ONLY = False


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    random.seed()
    
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


def flush_stdin():
    termios.tcflush(sys.stdin, termios.TCIOFLUSH)
    # while len(select.select([sys.stdin.fileno()], [], [], None)[0]) > 0:
    #     os.read(sys.stdin.fileno(), 4096)


if __name__ == '__main__':
    main()
