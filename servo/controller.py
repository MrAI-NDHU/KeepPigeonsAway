import logging
import math
import time
from typing import Dict, List

from servo import Servo


class ControllerForPCA9685:
    def __init__(self, servos: Dict[object, Servo], chs: Dict[object, int],
                 pwm_freq: float, init_angles: Dict[object, float] = None):
        import Adafruit_PCA9685 as PCA9685
        
        if list(servos.keys()).sort() != list(chs.keys()).sort():
            raise ValueError
        if init_angles is None:
            init_angles = {k: (v.angle_min_deg + v.angle_max_deg) / 2
                           for k, v in servos.items()}
        elif list(servos.keys()).sort() != list(init_angles.keys()).sort():
            raise ValueError
        
        self.servos = servos
        self.chs = chs
        self.pwm_freq = pwm_freq
        self.init_angles = {k: servos[k].fix_angle(v)
                            for k, v in init_angles.items()}
        self.current_angles = self.init_angles.copy()
        
        PCA9685.software_reset()
        self.pca9685 = PCA9685.PCA9685()
        self.pca9685.set_pwm_freq(self.pwm_freq)
        for k in servos:
            self.pca9685.set_pwm(self.chs[k], 0, int(round(
                self.servos[k].angle_to_pwm_val(self.init_angles[k]))))
            time.sleep(self.servos[k].wait_time(self.servos[k].angle_max_deg))
    
    def rotate(self, angles: Dict[object, float], is_relative: bool):
        for k, angle in angles.items():
            angle = self.servos[k].fix_angle(
                angle + (self.current_angles[k] if is_relative else 0.0))
            if math.isclose(self.current_angles[k], angle):
                continue
            angle_diff = abs(self.current_angles[k] - angle)
            
            logging.info("Controller: rotating %s from %f to %f",
                         k, self.current_angles[k], angle)
            self.pca9685.set_pwm(self.chs[k], 0, int(round(
                self.servos[k].angle_to_pwm_val(angle))))
            time.sleep(self.servos[k].wait_time(angle_diff))
            self.current_angles[k] = angle


class ControllerForRPi:
    def __init__(self, servos: Dict[object, Servo], pins: Dict[object, int],
                 init_angles: Dict[object, float] = None):
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BOARD)
        
        if list(servos.keys()).sort() != list(pins.keys()).sort():
            raise ValueError
        if init_angles is None:
            init_angles = {k: (v.angle_min_deg + v.angle_max_deg) / 2
                           for k, v in servos.items()}
        elif list(servos.keys()).sort() != list(init_angles.keys()).sort():
            raise ValueError
        
        self.servos = servos
        self.pins = pins
        self.active_servos = set()
        self.init_angles = {k: servos[k].fix_angle(v)
                            for k, v in init_angles.items()}
        self.current_angles = self.init_angles.copy()
        
        self.pwms = {}
        self.gpio = GPIO
        for k in servos:
            self.gpio.setup(self.pins[k], GPIO.OUT)
            self.pwms[k] = self.gpio.PWM(pins[k], servos[k].pwm_freq)
        
        logging.info("Controller: initialized, set pins %s", pins)
    
    def __del__(self):
        self.gpio.cleanup()
        logging.info("Controller: deleted")
    
    def start(self, servos_key: List[object]):
        started = set()
        for k in servos_key:
            if k in self.servos and not (k in self.active_servos):
                init_duty_cycle \
                    = self.servos[k].angle_to_pwm_val(self.init_angles[k])
                self.pwms[k].start(init_duty_cycle)
                time.sleep(
                    self.servos[k].wait_time(self.servos[k].angle_max_deg))
                self.active_servos.add(k)
                started.add(k)
        if len(started) > 0:
            logging.info("Controller: started %s", started)
    
    def stop(self, servos_key: List[object]):
        stopped = set()
        for k in servos_key:
            if k in self.servos and k in self.active_servos:
                self.pwms[k].stop()
                self.active_servos.remove(k)
                stopped.add(k)
        if len(stopped) > 0:
            logging.info("Controller: stopped %s", stopped)
    
    def rotate(self, angles: Dict[object, float], is_relative: bool):
        for k, angle in angles.items():
            if not (k in self.active_servos):
                continue
            
            angle = self.servos[k].fix_angle(
                angle + (self.current_angles[k] if is_relative else 0.0))
            if math.isclose(self.current_angles[k], angle):
                continue
            angle_diff = abs(self.current_angles[k] - angle)
            
            logging.info("Controller: rotating %s from %f to %f",
                         k, self.current_angles[k], angle)
            self.pwms[k].ChangeDutyCycle(
                self.servos[k].angle_to_pwm_val(angle))
            time.sleep(self.servos[k].wait_time(angle_diff))
            self.pwms[k].ChangeDutyCycle(0.0)
            self.current_angles[k] = angle
