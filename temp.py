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

def findIntersection(params1, params2):
    x, y = -1, -1
    det = params1[0] * params2[1] - params2[0] * params1[1]
    if abs(det) < 0.5:  # lines are approximately parallel
        return (-1, -1)
    else:
        x = (params2[1] * -params1[2] - params1[1] * -params2[2]) / det
        y = (params1[0] * -params2[2] - params2[0] * -params1[2]) / det
    return (x, y)

def calcParams(p1, p2):
    if p2[1] - p1[1] == 0:
        a = 0.0
        b = -1.0
    elif p2[0] - p1[0] == 0:
        a = -1.0
        b = 0.0
    else:
        a = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = -1.0

    c = (-a * p1[0]) - b * p1[1]
    return np.array([a, b, c], dtype=np.float32)

def find_quads(img, frame):
    #ret, bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
    #print(ret)
    squares = []
    blurred = cv2.GaussianBlur(img,(11,11),0)
    #bin = cv2.Canny(blurred, cv2.THRESH_OTSU, 255)# apertureSize = 5; cv2.THRESH_OTSU, revuelve 2 objetos ret, 

    bin = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        # bin = cv2.dilate(bin, kernel, iterations=1)
    bin = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, kernel)

        #cv2.imshow("Dilate", bin)
        #cv2.imshow("open", bin)
    
    cv2.imshow("result", bin)
    
    convexHull_mask = np.zeros_like(bin)
    contours, _ = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #contours = sorted(range(len(contours)), key=lambda x: len(contours[x]), reverse=True)
    #contours = sorted(range(len(contours)), key=lambda x: len(contours[x]), reverse=True)
    #print(contours)
    hull = cv2.convexHull(contours[0], False)
    cv2.drawContours(convexHull_mask, [hull], 0, (255), thickness=cv2.FILLED)
    lines = cv2.HoughLinesP(convexHull_mask, 1, np.pi / 200, 50, minLineLength=50, maxLineGap=10)
    corners = []
    print(lines)
    if lines is not None and len(lines)>3 and len(lines) == 4:
        params = []
        for line in lines:
            p1 = (line[0], line[1])
            p2 = (line[2], line[3])
            params.append(calcParams(p1, p2))

        
        for i in range(len(params)):
            for j in range(i, len(params)):  # j starts at i so we don't have duplicated points
                intersec = findIntersection(params[i], params[j])            
                if (0 < intersec[0] < bin.cols) and (0 < intersec[1] < bin.rows):
                    print("corner:", intersec)
                    corners.append(intersec)
        
        for corner in corners:
            cv2.circle(frame, (int(corner[0]), int(corner[1])), 3, (0, 0, 255), -1)

        if len(corners) == 4:  # we have the 4 final corners
            return corners
        
    return corners




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
        quads = find_quads(img,frame)
        
        warpedCard = np.zeros((640, 480, 3), dtype=np.uint8)

        if len(quads) == 4:
            homography, _ = cv2.findHomography(np.array(quads), np.array([(480, 0), (480, 640), (0, 0), (0, 640)]))
            warpedCard = cv2.warpPerspective(frame, homography, (480, 640))
        
        #cv2.drawContours(img, quads, -1, (255, 0, 0), 3 )
        cv2.imshow('quads', warpedCard)
        #cv2.imwrite("quad"+ str(time.time())+'.png', img)
        #cv2.imshow("OpenCVCam", frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap_video.release()
cv2.destroyAllWindows()