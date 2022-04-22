from yolodetect import ODBranch
from lanedetect import LDBranch

import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
height = 590
width = 1640

from convex_polygon_intersection import intersect

def change_poly_to_array(polygon):
    polygon = list(polygon)    
    #polygon.append(polygon[0])
    x,y = zip(*polygon)
    p = []
    for i in range(len(x)):
        p.append([int(x[i]), int(y[i])])
    p = np.array(p)
    return(p)


if '__main__' == __name__:

    yoloinfo = ODBranch()
    laneinfo = LDBranch()
    assert len(yoloinfo) == len(laneinfo)


    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    split = 'firsttry'
    print(split[:-3]+'avi')
    vout = cv2.VideoWriter(split[:-3]+'avi', fourcc , 30.0, (width, height))

    for key, yolores in yoloinfo.items():
        
        lanei = laneinfo[key]
        laneres = []
        for x in lanei:
            if x:
                laneres.append(x)


        imgpath = os.path.join('samples/', key+'.jpg')
        print(imgpath)
        # load raw image
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        vout.write(img)

        # process obstacle detection branch results
        yoloarea = []
        for _, x, y, w, h in yolores:
            x = float(x)*width
            y = float(y)*height
            w = float(w)*width
            h = float(h)*height
            x1 = (x-w/2)
            y1 = (y-h/2)
            x2 = (x+w/2)
            y2 = (y+h/2)
            yoloarea.append(np.array([[x1,y1], [x2,y1], [x2,y2], [x1,y2]]))

        # process lane detection branch result
        linelen = len(laneres)
        c = []
        if laneres[0][0][1] < height:
            c = [[laneres[0][0][0], height]]
        c += laneres[0]
        c += laneres[linelen-1][::-1]
        if laneres[linelen-1][0][1] < height:
            c += [[laneres[linelen-1][0][0], height]]
        lanearea = np.array(c)

        
        c = lanearea
        cv2.fillPoly(img, [c], (0, 150, 0))

        # area intersection algorithm
        c = c.tolist()
        for i in range(len(yoloarea)):
            d = yoloarea[i].tolist()
            #d.append(d[0])
        
            polygon3 = intersect(d, c)
            if polygon3:
                p = change_poly_to_array(polygon3)
                cv2.fillPoly(img, [p], (0, 0, 0))

        plt.figure()
        fig, ax = plt.subplots()
        ax.imshow(img)
        for i in range(len(laneres)):
            x = []
            y = []
            for j in range(len(laneres[i])):
                x.append(laneres[i][j][0])
                y.append(laneres[i][j][1])
            x = np.array(x)
            y = np.array(y)
    
            plt.plot(x, y)

        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator()) 
        plt.gca().yaxis.set_major_locator(NullLocator())
        savepath = os.path.join('output', key+'.png')
        #plt.savefig(savepath, bbox_inches="tight", pad_inches=0.0, dpi=300)
        vout.write(img)

        #cv2.imwrite(os.path.join('output', key+'.png'), img)
    vout.release()


