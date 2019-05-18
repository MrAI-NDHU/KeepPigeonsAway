from queue import Queue
from threading import Thread
from typing import Any
import logging
import pickle
import random
import sys
import time
import termios

from Jetson import GPIO
import cv2
import keyboard

from servo import Servo
from servo.controller import ControllerForPCA9685
import darknet


class DriveAwayPigeons:
    Y, X = "y", "x"
    
    def __init__(self, split_w: int, split_h: int, angle_prec: float):
        self.split_w, self.split_h = split_w, split_h
        self.areas_cnt = self.split_w * self.split_h
        self.angle_prec = angle_prec
        self.laser_pin = 18
        self.sweep_around_times = 3
        self.can_detect, self.can_sweep = False, False
        self.cap = cv2.VideoCapture(0)
        self.cap_ratio = 16 / 9
        self.split_line_color = (0x00, 0xFF, 0x00)
        self.areas_canter_color = (0x00, 0x00, 0xFF)
        
        self.darknet_net, self.darknet_net_w, self.darknet_net_h = None, 0, 0
        self.darknet_meta, self.darknet_img = None, None
        
        self.showing_w, self.showing_h = 0, 0
        self.areas_angle = [[{}]]
        self.area_angle_spacing = {}
        
        self.arm = self.set_arm()
        self.init_darknet()
        self.init_laser()
        
        self.thd_detecting = Thread(target=self.thd_detecting_func)
        self.thd_deciding = Thread(target=self.thd_deciding_func)
        self.thd_sweeping = Thread(target=self.thd_sweeping_func)
        self.thd_showing = Thread(target=self.thd_showing_func)
        self.que_deciding = Queue(1)
        self.que_sweeping = Queue(1)
        self.que_showing = Queue(1)
        self.thd_showing.start()
        
        self.init_areas_angle()
        
        self.thd_detecting.start()
        self.thd_deciding.start()
        self.thd_sweeping.start()
        
        self.thd_detecting.join()
    
    def __del__(self):
        GPIO.cleanup()
    
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
        self.showing_h = min(self.darknet_net_w, self.darknet_net_h)
        self.showing_w = int(round(self.showing_h * self.cap_ratio))
    
    def init_laser(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.laser_pin, GPIO.OUT, initial=GPIO.HIGH)
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
    
    def set_arm(self) -> ControllerForPCA9685:
        mg995_sec_per_angle = ((0.16 - 0.2) / (6.0 - 4.8) * (5.0 - 4.8) + 0.2) / 60.0
        mg995_tilt = Servo(0.0, 180.0, 150.0, 510.0, 50.0, mg995_sec_per_angle)
        mg995_pan = Servo(0.0, 180.0, 180.0, 630.0, 50.0, mg995_sec_per_angle)
        servos = {self.Y: mg995_tilt, self.X: mg995_pan}
        chs = {self.Y: 0, self.X: 1}
        return ControllerForPCA9685(servos, chs, 60.0)
    
    def check_areas_angle(self):
        for y in range(self.split_h):
            for x in range(self.split_w):
                if self.areas_angle[y][x] is None:
                    raise Exception("failed to check areas angle")
                if TEST_MODE:
                    n = y * self.split_w + x
                    logging.info("check_areas_angle: check {}: {}".format(n, self.areas_angle[y][x]))
                    self.arm.rotate(self.areas_angle[y][x], False)
                    self.open_laser()
                    time.sleep(2)
                    self.can_sweep = True
                    for i in range(100):
                        self.sweep_area(x, y)
                    self.can_sweep = False
                    self.close_laser()
    
    def init_areas_angle(self):
        self.areas_angle = [[None] * self.split_w for i in range(self.split_h)]
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
                    self.arm.rotate({self.Y: -self.angle_prec}, True)
                elif keyboard.is_pressed("down"):
                    self.arm.rotate({self.Y: self.angle_prec}, True)
                elif keyboard.is_pressed("left"):
                    self.arm.rotate({self.X: self.angle_prec}, True)
                elif keyboard.is_pressed("right"):
                    self.arm.rotate({self.X: -self.angle_prec}, True)
                elif keyboard.is_pressed("s"):
                    termios.tcflush(sys.stdin, termios.TCIOFLUSH)
                    i = int(input("Input area number in range [{},{}]: "
                                  .format(0, self.split_w * self.split_h - 1)))
                    self.areas_angle[i // self.split_w][i % self.split_w] = \
                        self.arm.current_angles.copy()
            self.close_laser()
            self.init_sweep_attrs()
            self.check_areas_angle()
            filepath = input("Save areas angle: ").strip()
            if filepath != "":
                with open(filepath, "wb") as f:
                    pickle.dump(self.areas_angle, f)
    
    def init_sweep_attrs(self):
        self.area_angle_spacing = {self.X: 0.0, self.Y: 0.0}
        for i in range(1, self.areas_cnt):
            ax1, ay1 = i % self.split_w, i // self.split_w
            ax0, ay0 = (i - 1) % self.split_w, (i - 1) // self.split_w
            if ax1 > ax0:
                area1_x = self.areas_angle[ay1][ax1][self.X]
                area0_x = self.areas_angle[ay0][ax0][self.X]
                self.area_angle_spacing[self.X] = \
                    max(abs(area1_x - area0_x), self.area_angle_spacing[self.X])
            if ay1 > ay0:
                area1_y = self.areas_angle[ay1][ax1][self.Y]
                area0_y = self.areas_angle[ay0][ax0][self.Y]
                self.area_angle_spacing[self.Y] = \
                    max(abs(area1_y - area0_y), self.area_angle_spacing[self.Y])
    
    def sweep_area(self, area_x: int, area_y: int):
        if not self.can_sweep:
            return
        area_angle = self.areas_angle[area_y][area_x]
        x = random.uniform(area_angle[self.X] - self.area_angle_spacing[self.X] / 2, \
            area_angle[self.X] + self.area_angle_spacing[self.X] / 2)
        y = random.uniform(area_angle[self.Y] - self.area_angle_spacing[self.Y] / 2, \
            area_angle[self.Y] + self.area_angle_spacing[self.Y] / 2)
        time.sleep(0.05)
        self.arm.rotate({self.X: x, self.Y: y}, False)
    
    def get_cap_img(self) -> Any:
        _, img = self.cap.read()
        return cv2.resize(img, (self.showing_w, self.showing_h),
                          interpolation=cv2.INTER_LINEAR)
    
    def draw_split_lines_and_areas_canter(self, img: Any) -> Any:
        for sx in range(1, self.split_w):
            x = int(round(self.showing_w / self.split_w * sx))
            img = cv2.line(img, (x, 0), (x, self.showing_h - 1),
                           self.split_line_color, 1)
        for sy in range(1, self.split_h):
            y = int(round(self.showing_h / self.split_h * sy))
            img = cv2.line(img, (0, y), (self.showing_w - 1, y),
                           self.split_line_color, 1)
        for sx in range(1, self.split_w * 2 + 1, 2):
            for sy in range(1, self.split_h * 2 + 1, 2):
                x = int(round(self.showing_w / (self.split_w * 2) * sx))
                y = int(round(self.showing_h / (self.split_h * 2) * sy))
                img = cv2.circle(img, (x, y), 4, self.areas_canter_color, 1)
        return img
    
    def thd_detecting_func(self):
        self.can_detect = True
        while True:
            detections = None
            # TODO
            self.que_deciding.put(detections)
            pass
    
    def thd_deciding_func(self):
        while True:
            detections = self.que_deciding.get()
            # TODO
            # time.sleep(0.01)  # if high cpu usage
            area_x, area_y = None, None
            detected_areas, sweeping_area, ignored_areas, fps = \
                None, None, None, None
            self.que_sweeping.put((area_x, area_y))
            self.que_showing.put(
                (detected_areas, sweeping_area, ignored_areas, fps))
            pass
    
    def thd_sweeping_func(self):
        self.can_sweep = True
        while True:
            if not self.can_sweep:
                time.sleep(0.01)  # avoid high CPU usage
                continue
            area_x, area_y = self.que_sweeping.get()
            self.sweep_area(area_x, area_y)
    
    def thd_showing_func(self):
        while True:
            img = None
            if not self.can_detect:
                img = self.get_cap_img()
                img = self.draw_split_lines_and_areas_canter(img)
            else:
                detected_areas, sweeping_area, ignored_areas, fps = \
                    self.que_showing.get()
                # TODO
            cv2.imshow("image", img)
            cv2.waitKey(1)


TEST_MODE = True


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
        d = DriveAwayPigeons(split_w, split_h, 0.05)
    except KeyboardInterrupt:
        del d


if __name__ == '__main__':
    main()
