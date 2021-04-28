import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

class Poly():
    
    def __init__(self, path):
        self.fname = path
        self.img = cv2.imread(self.fname)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.point = []
        self.fig = plt.figure()

    def getCoord(self):
        ax = self.fig.add_subplot(111)
        plt.imshow(self.img)
        cid = self.fig.canvas.mpl_connect('button_press_event', self.__onclick)
        plt.show()
        return self.point

    def __onclick(self,event):
        self.point.append((event.xdata,event.ydata))
        plt.plot(event.xdata, event.ydata, '.', linewidth=2, color='red', markersize=16)
        self.fig.canvas.draw()
        return self.point


class Mask():

    def __init__(self, path,pts):
        self.fname = path
        self.pts = np.array(pts, np.dtype('int'))
        self.img = cv2.imread(self.fname)
        self.dir = os.path.dirname(self.fname)
    
    def mask(self):
        ## (1) Crop the bounding rect
        rect = cv2.boundingRect(self.pts)
        x,y,w,h = rect
        croped = self.img[y:y+h, x:x+w].copy()
        ## (2) make mask
        self.pts = self.pts - self.pts.min(axis=0)
        mask = np.zeros(croped.shape[:2], np.uint8)
        cv2.drawContours(mask, [self.pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
        ## (3) do bit-op
        dst = cv2.bitwise_and(croped, croped, mask=mask)
        
        ## (4) add the white background
        bg = np.ones_like(croped, np.uint8)*255
        cv2.bitwise_not(bg,bg, mask=mask)
        res = bg + dst
        cv2.imwrite(f"{self.dir}\\dst2.png", res)
        
        return res


path = 'c:\\temp\\K_09.24.PNG'
t = Poly(path)
points = t.getCoord()
#print(points)
m = Mask(path,points)
#print(m.pts)
masked = m.mask()

cv2.imshow('image',masked)
cv2.waitKey(0)   
#closing all open windows 
cv2.destroyAllWindows() 