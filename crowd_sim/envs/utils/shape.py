import math

import shapely.geometry
import shapely.affinity
import matplotlib.pyplot as plt

class HeadingRect:
    def __init__(self, mx, my, agent_width, length, angle):
        self.mx = mx
        self.my = my
        self.cx = mx + (length/2)*math.cos(angle/180 * math.pi)
        self.cy = my + (length/2)*math.sin(angle/180 * math.pi)
        self.w = length
        self.h = agent_width
        self.angle = angle
        self.color = 'yellow'

    def get_shapely_contour(self):
        w = self.w
        h = self.h
        c = shapely.geometry.box(-w/2.0, -h/2.0, w/2.0, h/2.0)
        rc = shapely.affinity.rotate(c, self.angle)
        return shapely.affinity.translate(rc, self.cx, self.cy)

    def intersection(self, other):
        return self.get_shapely_contour().intersection(other.get_shapely_contour())

    def intersects(self, other):
        return not self.intersection(other).is_empty

    def get_pyplot_rect(self):
        return plt.Rectangle(
            (
                self.mx + (self.h/2)*math.sin(self.angle/180 * math.pi),
                self.my - (self.h/2)*math.cos(self.angle/180 * math.pi)
            ),
            self.w, self.h, self.angle, color=self.color, alpha=0.3, fill=True)


class AgentHeadingRect(HeadingRect):
    RECT_EXTEND_TIME_STEPS = 3

    def __init__(self, px, py, radius, vx, vy, kinematics, action=None):
        if action and kinematics == 'holonomic':
            assert type(action).__name__ == 'ActionXY'
            v = math.sqrt(action.vx**2 + action.vy**2)
            degree = math.atan2(action.vy, action.vx) / math.pi * 180
        elif action: 
            assert type(action).__name__ == 'ActionRot'
            v = action.v
            degree = action.r + math.atan2(action.vy, action.vx) / math.pi * 180
        else:
            v =  math.sqrt(vx**2 + vy**2)
            degree = math.atan2(vy, vx) / math.pi * 180
        super().__init__(px, py, radius*2, v*self.RECT_EXTEND_TIME_STEPS, degree)
