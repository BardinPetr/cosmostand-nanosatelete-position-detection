# -*- coding: utf-8 -*-

from operator import itemgetter
from math import atan2,degrees
import sys, math, pygame
import numpy as np
import cv2

def getAngle(p1, p2):
    return degrees(atan2(p2[0]-p1[0], p2[1]-p2[0])) - 90

#Credit to codentronix.com
class Point3D:
    def __init__(self, x = 0, y = 0, z = 0):
        self.x, self.y, self.z = float(x), float(y), float(z)
 
    def rotateX(self, angle):
        """ Rotates the point around the X axis by the given angle in degrees. """
        rad = angle * math.pi / 180
        cosa = math.cos(rad)
        sina = math.sin(rad)
        y = self.y * cosa - self.z * sina
        z = self.y * sina + self.z * cosa
        return Point3D(self.x, y, z)
 
    def rotateY(self, angle):
        """ Rotates the point around the Y axis by the given angle in degrees. """
        rad = angle * math.pi / 180
        cosa = math.cos(rad)
        sina = math.sin(rad)
        z = self.z * cosa - self.x * sina
        x = self.z * sina + self.x * cosa
        return Point3D(x, self.y, z)
 
    def rotateZ(self, angle):
        """ Rotates the point around the Z axis by the given angle in degrees. """
        rad = angle * math.pi / 180
        cosa = math.cos(rad)
        sina = math.sin(rad)
        x = self.x * cosa - self.y * sina
        y = self.x * sina + self.y * cosa
        return Point3D(x, y, self.z)
 
    def project(self, win_width, win_height, fov, viewer_distance):
        """ Transforms this 3D point to 2D using a perspective projection. """
        factor = fov / (viewer_distance + self.z)
        x = self.x * factor + win_width / 2
        y = -self.y * factor + win_height / 2
        return Point3D(x, y, self.z)
 
class Simulation:
    def __init__(self):
        pygame.init()
 
        self.screen = pygame.display.set_mode((640, 480))
        pygame.display.set_caption("Nano-satelite position demonstration")
 
        self.clock = pygame.time.Clock()
 
        self.vertices = [
            Point3D(-1,1,-1),
            Point3D(1,1,-1),
            Point3D(1,-1,-1),
            Point3D(-1,-1,-1),
            Point3D(-1,1,1),
            Point3D(1,1,1),
            Point3D(1,-1,1),
            Point3D(-1,-1,1)
        ]
 
        self.faces  = [(0,1,2,3),(1,5,6,2),(5,4,7,6),(4,0,3,7),(0,4,5,1),(3,2,6,7)]
        self.colors = [(255,0,255),(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,255,0)]
 
        self.pitch = 0
        self.roll = 0
        self.yaw = 0

        self.r0 = 0
        self.r1 = 0
        self.r2 = 0
        self.cnt = 0

        """Углы виртуального кубика при нулевых углах спутника"""
        self.corr_pitch = 0  
        self.corr_roll = 0        
 
    def run(self):
        def color_params(R = None, G = None, B = None):
            hsv_color = cv2.cvtColor(np.uint8([[[R,G,B]]]),cv2.COLOR_BGR2HSV)
            color_f = np.array(np.array(hsv_color)[0][0][0])
            hsv_max = np.array((color_f + 10, 255, 255), np.uint8)
            hsv_min = np.array((color_f - 10, 100, 100), np.uint8)
            if (color_f<10):
                hsv_min = np.array((160, 100, 100), np.uint8)
                hsv_max = np.array((179, 255, 255), np.uint8)
            color = (R,G,B)
            return hsv_min, hsv_max, color

        def tresh(hsv_min = None, hsv_max = None, color = None):
            thresh = cv2.inRange(hsv, hsv_min, hsv_max)
            moments = cv2.moments(thresh, 1)
            dM01 = moments['m01']
            dM10 = moments['m10']
            dArea = moments['m00']
            x = None
            y = None
            if dArea > 100:
                x = int(dM10 / dArea)
                y = int(dM01 / dArea)
                cv2.circle(img, (x, y), 5, color, 2)
                cv2.putText(img, "%d-%d" % (x, y), (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            return (x if x else 0, y if y else 0)

        hsv_y, hsv_yx, color_yellow = color_params(0,255,255)
        hsv_r, hsv_rx, color_red = color_params(0,0,255)
        hsv_g, hsv_gx, color_green = color_params(0,255,0)
        hsv_b, hsv_bx, color_blue = color_params(255,0,0)
    
        cap = cv2.VideoCapture(0)
        while 1:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    cap.release()
                    cv2.destroyAllWindows()
                    pygame.quit()
                    sys.exit()
 
            self.clock.tick(50)
            self.screen.fill((0,0,0))
 
            mass = np.zeros((4,2))
            flag, img = cap.read()
            img = cv2.flip(img, 1)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            a = tresh(hsv_y, hsv_yx, color_yellow)
            b = tresh(hsv_r, hsv_rx, color_red)
            c = tresh(hsv_g, hsv_gx, color_green)
            #d = tresh(hsv_b, hsv_bx, color_blue) #We do not need it

            self.r0 += getAngle(a, b)
            self.r1 += getAngle(a, c)
            self.cnt += 1

            if self.cnt == 5:
                self.pitch = self.r0 / self.cnt
                self.roll = self.r1 / self.cnt
                self.r0 = self.r1 = self.cnt = 0
                print(self.pitch, self.roll)

            t = []
            for v in self.vertices:
                r = v.rotateX(self.corr_roll+self.roll).rotateY(self.corr_pitch+self.pitch).rotateZ(45)
                p = r.project(self.screen.get_width(), self.screen.get_height(), 256, 4)
                t.append(p)
 
            avg_z = []
            i = 0
            for f in self.faces:
                z = (t[f[0]].z + t[f[1]].z + t[f[2]].z + t[f[3]].z) / 4.0
                avg_z.append([i,z])
                i = i + 1
 
            for tmp in sorted(avg_z,key=itemgetter(1),reverse=True):
                face_index = tmp[0]
                f = self.faces[face_index]
                pointlist = [(t[f[0]].x, t[f[0]].y), (t[f[1]].x, t[f[1]].y),
                             (t[f[1]].x, t[f[1]].y), (t[f[2]].x, t[f[2]].y),
                             (t[f[2]].x, t[f[2]].y), (t[f[3]].x, t[f[3]].y),
                             (t[f[3]].x, t[f[3]].y), (t[f[0]].x, t[f[0]].y)]
                pygame.draw.polygon(self.screen,self.colors[face_index],pointlist)
 
            pygame.display.flip()            
            cv2.imshow('pre-result', img)

            ch = cv2.waitKey(1)
            if ch == 27:
                break
 
if __name__ == "__main__":
    Simulation().run()
