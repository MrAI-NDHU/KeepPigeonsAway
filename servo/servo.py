class Servo:
    def __init__(self, angle_min_deg: float, angle_max_deg: float,
                 pwm_val_min: float, pwm_val_max: float, pwm_freq: float,
                 sec_per_deg: float):
        self._angle_min_deg = angle_min_deg
        self._angle_max_deg = angle_max_deg
        self._pwm_val_min = pwm_val_min
        self._pwm_val_max = pwm_val_max
        self._pwm_freq = pwm_freq
        self._sec_per_deg = sec_per_deg
    
    @property
    def angle_min_deg(self) -> float:
        return self._angle_min_deg
    
    @property
    def angle_max_deg(self) -> float:
        return self._angle_max_deg
    
    @property
    def pwm_val_min(self) -> float:
        return self._pwm_val_min
    
    @property
    def pwm_val_max(self) -> float:
        return self._pwm_val_max
    
    @property
    def pwm_freq(self) -> float:
        return self._pwm_freq
    
    @property
    def sec_per_deg(self) -> float:
        return self._sec_per_deg
    
    def fix_angle(self, angle: float) -> float:
        if angle < self.angle_min_deg:
            return self.angle_min_deg
        elif angle > self.angle_max_deg:
            return self.angle_max_deg
        else:
            return angle
    
    def wait_time(self, rel_angle: float) -> float:
        return self.sec_per_deg * self.fix_angle(abs(rel_angle))
    
    def angle_to_pwm_val(self, angle: float) -> float:
        return self.pwm_val_min \
               + (self.fix_angle(angle) - self.angle_min_deg) \
               / (self.angle_max_deg - self.angle_min_deg) \
               * (self.pwm_val_max - self.pwm_val_min)
