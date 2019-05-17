from typing import Any
import logging
import pickle
import queue
import sys
import threading
import time

from Jetson import GPIO
import cv2
import keyboard

from servo import Servo
from servo.controller import ControllerForPCA9685
import darknet

TEST_MODE = True


class DriveAwayPigeons:
    Y, X = "y", "x"
    
    def __init__(self, split_w: int, split_h: int, angle_prec: float):
        self.split_w, self.split_h = split_w, split_h
        self.areas_cnt = self.split_w * self.split_h
        self.angle_prec = angle_prec
        self.laser_pin = 18
        self.can_sweep = False
        self.cap = cv2.VideoCapture(0)
        self.cap_ratio = \
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) \
            / self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.split_line_color = (0x00, 0xFF, 0x00)
        self.areas_canter_color = (0x00, 0x00, 0xFF)
        
        self.darknet_net, self.darknet_meta = None, None
        self.darknet_net_w, self.darknet_net_h = 0, 0
        self.showing_w, self.showing_h = 0, 0
        self.areas_angle = [[{}]]
        self.sweep_around_times = 0
        self.sweep_angle_amp = {}
        
        self.arm = self.set_arm()
        self.init_darknet()
        
        self.thd_delecting = threading.Thread(target=self.thd_delecting_func)
        self.thd_deciding = threading.Thread(target=self.thd_deciding_func)
        self.thd_sweeping = threading.Thread(target=self.thd_sweeping_func)
        self.thd_showing = threading.Thread(target=self.thd_showing_func)
        self.que_deciding = queue.Queue(1)
        self.que_sweeping = queue.Queue(1)
        self.que_showing = queue.Queue(1)
        self.thd_delecting.start()
        self.thd_deciding.start()
        self.thd_sweeping.start()
        self.thd_showing.start()
        
        self.init_laser()
        self.init_areas_angle()
    
    def __del__(self):
        GPIO.cleanup()
    
    def init_darknet(self):
        config_path = "./cfg/yolov3-tiny.cfg"
        weight_path = "./yolov3-tiny_pigeon.weights"
        meta_path = "./cfg/pigeon.data"
        self.darknet_net = darknet.load_net_custom(config_path.encode(
            "ascii"), weight_path.encode("ascii"), 0, 1)
        self.darknet_meta = darknet.load_meta(meta_path.encode("ascii"))
        self.darknet_net_w = darknet.network_width(self.darknet_net)
        self.darknet_net_h = darknet.network_height(self.darknet_net)
        self.showing_h = min(self.darknet_net_w, self.darknet_net_h)
        self.showing_w = self.showing_h * self.cap_ratio
    
    def init_laser(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.laser_pin, GPIO.OUT, initial=GPIO.LOW)
        if TEST_MODE:
            logging.info("init_laser: test start")
            for i in range(4):
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
        mg995_sec_per_angle = (0.16 - 0.2) / (6.0 - 4.8) * (5.0 - 4.8) + 0.2
        mg995_tilt = Servo(0.0, 180.0, 150.0, 510.0, 50.0, mg995_sec_per_angle)
        mg995_pan = Servo(0.0, 180.0, 180.0, 630.0, 50.0, mg995_sec_per_angle)
        servos = {self.Y: mg995_tilt, self.X: mg995_pan}
        chs = {self.Y: 0, self.X: 1}
        return ControllerForPCA9685(servos, chs, 60.0)
    
    def check_areas_angle(self):
        for y in range(len(self.areas_angle)):
            for x in range(y):
                if self.areas_angle[y][x] is None:
                    raise Exception("failed to check areas angle")
                if TEST_MODE:
                    n = y * len(self.areas_angle) + x
                    logging.info("check_areas_angle: check {}".format(n))
                    self.arm.rotate(self.areas_angle[y][x], False)
                    self.open_laser()
                    time.sleep(2)
                    self.sweep_area(x, y)
                    self.close_laser()
    
    def init_areas_angle(self):
        self.areas_angle = [[None] * self.split_w] * self.split_h
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
                    self.arm.rotate({self.X: -self.angle_prec}, True)
                elif keyboard.is_pressed("right"):
                    self.arm.rotate({self.X: self.angle_prec}, True)
                elif keyboard.is_pressed("s"):
                    i = int(input("Input area number in range [{},{}]: "
                                  .format(0, self.split_w * self.split_h - 1)))
                    self.areas_angle[i // self.split_h][i % self.split_w] = \
                        self.arm.current_angles
            self.close_laser()
            self.init_sweep_attrs()
            self.check_areas_angle()
            filepath = input("Save areas angle: ").strip()
            if filepath != "":
                with open(filepath, "wb") as f:
                    pickle.dump(self.areas_angle, f)
    
    def init_sweep_attrs(self):
        area_angle_spacing = {self.X: 0.0, self.Y: 0.0}
        for i in self.areas_angle:
            for j in range(1, len(i)):
                area_angle_spacing[self.X] = max(i[j][self.X] - i[
                    j - 1][self.X], area_angle_spacing[self.X])
                area_angle_spacing[self.Y] = max(i[j][self.Y] - i[
                    j - 1][self.Y], area_angle_spacing[self.Y])
        t = int(round(min(area_angle_spacing[self.X],
                          area_angle_spacing[self.Y]) / 2 / self.angle_prec))
        self.sweep_angle_amp[self.X] = area_angle_spacing[self.X] / t
        self.sweep_angle_amp[self.Y] = area_angle_spacing[self.Y] / t
        self.sweep_around_times = t
    
    def sweep_area(self, area_x: int, area_y: int):
        area_angle = self.areas_angle[area_y][area_x]
        self.arm.rotate(area_angle, False)
        for i in range(self.sweep_around_times):
            if not self.can_sweep:
                return
            area_angle[self.X] -= self.sweep_angle_amp[self.X] / 2
            area_angle[self.Y] -= self.sweep_angle_amp[self.Y] / 2
            self.arm.rotate(area_angle, False)
            if not self.can_sweep:
                return
            self.arm.rotate({self.X: self.sweep_angle_amp[self.X] * i}, True)
            if not self.can_sweep:
                return
            self.arm.rotate({self.Y: self.sweep_angle_amp[self.Y] * i}, True)
            if not self.can_sweep:
                return
            self.arm.rotate({self.X: self.sweep_angle_amp[self.X] * -i}, True)
            if not self.can_sweep:
                return
            self.arm.rotate({self.Y: self.sweep_angle_amp[self.Y] * -i}, True)
    
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
                img = cv2.circle(img, (x, y), 2, self.areas_canter_color, 1)
        return img
    
    def thd_delecting_func(self):
        while True:
            pass
    
    def thd_deciding_func(self):
        while True:
            pass
    
    def thd_sweeping_func(self):
        while True:
            if not self.can_sweep:
                continue
            area_x, area_y = self.que_sweeping.get(True)
            self.sweep_area(area_x, area_y)
    
    def thd_showing_func(self):
        while True:
            img = self.get_cap_img()
            img = self.draw_split_lines_and_areas_canter(img)
            cv2.imshow("image", img)
            cv2.waitKey(1)


def main():
    split_w, split_h = 4, 3
    if len(sys.argv) == 2:
        split_w, split_h = int(sys.argv[1]), int(sys.argv[1])
    elif len(sys.argv) >= 3:
        split_w, split_h = int(sys.argv[1]), int(sys.argv[2])
    DriveAwayPigeons(split_w, split_h, 1.0)


if __name__ == '__main__':
    main()
