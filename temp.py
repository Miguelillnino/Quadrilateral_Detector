# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
import time
import numpy as np
import cv2

print(cv2.__version__)
import sys

PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

print("Version of software 2.0.1")

"Detector Funcional para quadrilateros en celulares"
"Median Blur"
"ADAPTIVE_THRESH_MEAN_C"
"dilate"

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    #print(type(d1))
    #print(type(d2))
    #print(np.dot(d1, d2))
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )


def find_quads(img):
    #ret, bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
    #print(ret)
    squares = []
    blurred = cv2.GaussianBlur(img,(11,11),0)
        #bin = cv2.Canny(blurred, cv2.THRESH_OTSU, 255)# apertureSize = 5; cv2.THRESH_OTSU, revuelve 2 objetos ret, 

    cv2.imshow("blur", blurred)
    bin = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        # bin = cv2.dilate(bin, kernel, iterations=1)
    bin = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, kernel)

        #cv2.imshow("Dilate", bin)
        #cv2.imshow("open", bin)
    
    cv2.imshow("result", bin)
    
    contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True) #0.02
        print("cnt Before condition",cnt)
        if len(cnt) == 4 and cv2.contourArea(cnt) > 7000: #and cv2.isContourConvex(cnt):
            print("if have 4 points and it is greater than 7000 pixels",cnt)
            print("contourArea",cv2.contourArea(cnt))        
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
            print("max_cos general",max_cos)
            if max_cos < 0.3:  # 0.3
                print("max_cos appoved",cnt)
                squares.append(cnt)
            
    return squares




cap_video = cv2.VideoCapture(0)
cap_video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('U','Y','V','Y'))
cap_video.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap_video.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    
Width  = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
Height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Width=", Width)
print("Height=",Height)

if (cap_video.isOpened() == False):
    print("Error reading video file")

cap_duration = 200

start_time = time.time()
while(True):
    ret, frame = cap_video.read()
    if ret == True:
        # Use de image to convert in gray scale
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #result.write(frame)
        # Use de image to obtain canny edge
        quads = find_quads(img)
        cv2.drawContours(img, quads, -1, (255, 0, 0), 3 )
        cv2.imshow('quads', img)
        #cv2.imwrite("quad"+ str(time.time())+'.png', img)
        #cv2.imshow("OpenCVCam", frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap_video.release()
cv2.destroyAllWindows()