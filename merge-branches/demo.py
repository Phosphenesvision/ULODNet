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

def area_intersection(cc, y):
    # area intersection algorithm
    wholearea = cv2.contourArea(cc)
    interarea = 0
    res = []
    cc = cc.tolist()
    #print(y)
    for i in range(len(y)):
        d = y[i].tolist()
        
        polygon3 = intersect(d, cc)
        if polygon3:
            p = change_poly_to_array(polygon3)
            res.append(p)
            
            interarea += cv2.contourArea(p)
            #print(i, p.tolist(), interarea)

    #print(wholearea, interarea)
    return res, interarea/wholearea

def comparethreshold(iou):
    if iou < 0.1:
        return (0, 200, 0)
    elif iou < 0.3:
        return (200, 200, 0)
    else:
        return (200, 0, 0)

if '__main__' == __name__:

    middlea = np.array([[820, 10], [780, 50], [800, 50], [800, 150], [840, 150], [840,50], [860, 50]])
    lefta = np.array([[560, 50], [580, 90], [580, 70], [640, 70], [640, 150], [680,150], [680, 30], [580, 30], [580, 10]])
    righta = np.array([[1080, 50], [1060, 90], [1060, 70], [1000, 70], [1000, 150], [960,150], [960, 30], [1060, 30], [1060, 10]])


    txtpath = '/media/phosphenesvision/新加卷/shujuji/yolov3/list/test_split/test1_crowd.txt'
    txtpath = 'bobo.txt'
    with open(txtpath, 'r') as file:
        imgfolder = file.readlines()
    imgfolder = [line.strip() for line in imgfolder]
    #imgfolder = imgfolder[121:122]
    #imgfolder = imgfolder[:1]
    root = '/media/phosphenesvision/新加卷/shujuji/yolov3'
    #imgfolder = ['samples/normal.jpg']
    #root = './'
    yoloinfo = ODBranch(imgfolder, data_root=root)
    laneinfo = LDBranch(imgfolder, data_root=root)
    assert len(yoloinfo) == len(laneinfo)



    for key, yolores in yoloinfo.items():
        
        lanei = laneinfo[key]
        laneres = []
        for x in lanei:
            if x:
                laneres.append(x)


        imgpath = key
        print(imgpath)
        # load raw image
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
        #print(yoloarea)
        # process lane detection branch result
        ress = []
        for k in laneres:
            if k:
                ress.append(k)
        laneres = ress
        #print(laneres)
        if not laneres:
            continue
        linelen = len(laneres)
        if linelen < 2:
            plt.figure()
            fig, ax = plt.subplots()
            ax.imshow(img)
            cv2.fillPoly(img, [lefta], (255, 255, 255))
            cv2.fillConvexPoly(img, middlea, (255, 255, 255))
            cv2.fillPoly(img, [righta], (255, 255, 255))
            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator()) 
            plt.gca().yaxis.set_major_locator(NullLocator())
            savepath = os.path.join('output', key[-9:])
            plt.savefig(savepath, bbox_inches="tight", pad_inches=0.0, dpi=300)
            plt.cla()
            plt.close("all")
            continue
        else:
            #if linelen == 4:
            #    laneres = laneres[1:]
            #    linelen -= 1
            fuxinbo = []
            
            zengjinghua = -1
            chenhaosen = []
            for i in range(linelen-1):
                c = []
                if laneres[i+1][0][1] < height:
                    #c += [[laneres[i+1][0][0], height]]
                    #print(laneres[i+1][0])
                    pass
                if laneres[i][0][1] < height:
                    #c = [[laneres[i][0][0], height]]
                    #print(laneres[i][0])
                    pass
                c += laneres[i]
                c += laneres[i+1][::-1]
                
                #pan duan shi di ji tiao chedao
                if laneres[i][0][0] <= width/2 and laneres[i+1][0][0] >= width/2:
                    zengjinghua = i

                rees, iou = area_intersection(np.array(c), yoloarea)
                chenhaosen.append(iou)
                #jingboran += str(iou)
                #jingboran += ' '
                fuxinbo += rees
        
        
        jingboran = ''
        if zengjinghua == 0:

            cv2.fillPoly(img, [lefta], (255, 255, 255))

            color = comparethreshold(chenhaosen[0])
            cv2.fillConvexPoly(img, middlea, color)

            if linelen > 2:
                color = comparethreshold(chenhaosen[1])
                cv2.fillPoly(img, [righta], color)
            else:
                cv2.fillPoly(img, [righta], (255, 255, 255))

            


        elif zengjinghua == 1:
            
            color = comparethreshold(chenhaosen[0])
            cv2.fillPoly(img, [lefta], color)

            color = comparethreshold(chenhaosen[1])
            cv2.fillConvexPoly(img, middlea, color)

            if linelen > 3:
                color = comparethreshold(chenhaosen[2])
                cv2.fillPoly(img, [righta], color)
            else:
                cv2.fillPoly(img, [righta], (255, 255, 255))
                

        elif zengjinghua == 2:
            
            color = comparethreshold(chenhaosen[1])
            cv2.fillPoly(img, [lefta], color)

            color = comparethreshold(chenhaosen[2])
            cv2.fillConvexPoly(img, middlea, color)

            cv2.fillPoly(img, [righta], (255, 255, 255))


        else:
            print('wrong lane')

                
        c = []
        if laneres[0][0][1] < height:
            c = [[laneres[0][0][0], height]]
        c += laneres[0]
        c += laneres[linelen-1][::-1]
        if laneres[linelen-1][0][1] < height:
            c += [[laneres[linelen-1][0][0], height]]
        lanearea = np.array(c)

        

        
        c = lanearea
        dimg = np.zeros((height, width, 3), dtype=int)
        cimg = dimg.astype(np.uint8).copy()
        cv2.fillPoly(cimg, [c], (0, 255, 0))
        #cv2.fillPoly(img, [c], (0, 150, 0))
        img = cv2.addWeighted(img, 1, cimg, 0.15, 0)
        #print(cimg.shape, type(cimg))
        #print(img.shape, type(img))
        

        rees, iou = area_intersection(c, yoloarea)
        
        #rees = fuxinbo
        for p in rees:
            cv2.fillPoly(img, [p], (0, 0, 0))
        #print(iou)

        plt.figure(figsize=(width,height),dpi=300)
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

        plt.text(50,
                    50,
                    s=jingboran,
                    color="white",
                    verticalalignment="top",
                    bbox={"color": 'r', "pad": 0},)
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator()) 
        plt.gca().yaxis.set_major_locator(NullLocator())
        savepath = os.path.join('output1', key[-9:])
        plt.savefig(savepath, bbox_inches="tight", pad_inches=0.0, dpi=300)
        
        
        plt.cla()
        plt.close("all")

        
    