import sys
import keyboard
from typing import Dict

import Jetson.GPIO

from servo import Servo
from servo.controller import ControllerForPCA9685


class DriveAwayPigeons:
    Y, X = "y", "x"
    
    def __init__(self, split_w: int, split_h: int, angle_precision: float):
        self.split_w, self.split_h = split_w, split_h
        self.areas_cnt = self.split_w * self.split_h
        
        self.arm = self.set_arm()
        self.areas_angle = self.set_areas_angle(angle_precision)
        self.areas_spacing_w, self.areas_spacing_h = self.set_areas_spacing()
    
    def set_arm(self) -> ControllerForPCA9685:
        mg995_sec_per_angle = (0.16 - 0.2) / (6.0 - 4.8) * (5.0 - 4.8) + 0.2
        mg995_tilt = Servo(0.0, 180.0, 150.0, 510.0, 50.0, mg995_sec_per_angle)
        mg995_pan = Servo(0.0, 180.0, 180.0, 630.0, 50.0, mg995_sec_per_angle)
        servos = {self.Y: mg995_tilt, self.X: mg995_pan}
        chs = {self.Y: 0, self.X: 1}
        return ControllerForPCA9685(servos, chs, 60.0)
    
    def set_areas_angle(self, angle_precision: float) -> [[Dict[object, int]]]:
        areas_angle = [[None] * self.split_w] * self.split_h
        while True:
            if keyboard.is_pressed("e"):
                break
            elif keyboard.is_pressed("up"):
                self.arm.rotate({self.Y: -angle_precision}, True)
            elif keyboard.is_pressed("down"):
                self.arm.rotate({self.Y: angle_precision}, True)
            elif keyboard.is_pressed("left"):
                self.arm.rotate({self.X: -angle_precision}, True)
            elif keyboard.is_pressed("right"):
                self.arm.rotate({self.X: angle_precision}, True)
            elif keyboard.is_pressed("s"):
                i = int(input("Input area number in range [{},{}]: "
                              .format(0, self.split_w * self.split_h - 1)))
                areas_angle[i // self.split_h][i % self.split_w] = \
                    self.arm.current_angles
        for i in areas_angle:
            for j in i:
                if j is None:
                    raise Exception("failed to set areas angle")
        return areas_angle
    
    def set_areas_spacing(self) -> (float, float):
        areas_spacing_w, areas_spacing_h = 0, 0
        for i in self.areas_angle:
            for j in range(1, len(i)):
                areas_spacing_w = \
                    max(i[j][self.X] - i[j - 1][self.X], areas_spacing_w)
                areas_spacing_h = \
                    max(i[j][self.Y] - i[j - 1][self.Y], areas_spacing_h)
        return areas_spacing_w, areas_spacing_h


def main():
    split_w, split_h = 4, 3
    if len(sys.argv) == 2:
        split_w, split_h = int(sys.argv[1]), int(sys.argv[1])
    elif len(sys.argv) >= 3:
        split_w, split_h = int(sys.argv[1]), int(sys.argv[2])
    DriveAwayPigeons(split_w, split_h, 1.0)


if __name__ == '__main__':
    main()
