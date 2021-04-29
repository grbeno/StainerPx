import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import math

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


class Stainer():
    " STAINER FOLTOZÓ MÓDSZER/METHOD "

    def __init__(self, path,colors):
        self.path = path
        self.colors = colors
        self.stColorInts = [(0,254,255)]
        self.stColors = []
        self.stData = []
        self.dir = os.path.dirname(self.path)

    def __createStColors(self):
        " Szürkeárnyalatok intervallumai + átfestő/foltozó színek -- grayscaled intervals + colors to stain - min:2, max:16"
        decrease = [0,0,127,84,63,50,42,36,31,28,25,23,21,19,18,16,15] # decrease from 254&('from' values) till lowest, with [i]
        blueColors = [
            (240,248,255),(230,230,250),(135,206,250),(0,191,255),(176,195,222),
            (30,144,255),(70,130,180),(95,158,160),(72,61,139),(65,105,225),
            (138,43,226),(0,109,176),(75,0,130),(0,0,255),(0,0,128),(0,0,57)
        ]
        for i in range(self.colors):
            intvs_len = len(self.stColorInts)-1
            start = self.stColorInts[intvs_len][1]-decrease[self.colors]
            end = self.stColorInts[intvs_len][1]-1
            index = intvs_len
            add_intv = tuple([index,start,end])
            self.stColorInts.append(add_intv)
        
        bright_blues = math.ceil(self.colors/2)
        dark_blues = self.colors - bright_blues
        self.stColors = blueColors[:bright_blues] + blueColors[-dark_blues:]
        self.stColors.insert(0,(255,255,255)) # 0. background


    def stain(self):
        " Stainer folt/színbecslő módszer - Stainer method for colour measuring "
        gray = cv2.imread(self.path)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        new = cv2.imread(self.path)
        height, width = new.shape[1], new.shape[0]
        size = height * width

        self.__createStColors()
        self.stData = [0 for x in range(len(self.stColors))]
        for i in range(width):
            for j in range(height):
                for(index,start,end) in self.stColorInts:
                    if start <= gray[i,j] and gray[i,j] <= end:
                        new[i,j]= self.stColors[index]    
                        count=self.stData[index]
                        self.stData[index]=count+1
                        break
        
        new = cv2.cvtColor(new, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'{self.dir}\\stain.png',new)
        self.stData = [float(format(data/(size-self.stData[0])*100,'.2f')) for data in self.stData]
        return self.stData


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

img = "c:\\temp\\dst2.png"
colors = 12
s = Stainer(img,colors)
print(s.stain())

