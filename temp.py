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


face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    #print(type(d1))
    #print(type(d2))
    #print(np.dot(d1, d2))
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def backgound_cnt(img):
    _, thresholded = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate the area of the largest contour
    background_area = cv2.contourArea(largest_contour)
    return background_area

def find_quads(img):
    #ret, bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
    #print(ret)
    # Threshold the image to segment the background
    #print(backgound_cnt(img))
    height, width = img.shape
    squares = []
    blurred = cv2.GaussianBlur(img,(11,11),0)
        #bin = cv2.Canny(blurred, cv2.THRESH_OTSU, 255)# apertureSize = 5; cv2.THRESH_OTSU, revuelve 2 objetos ret, 

    #cv2.imshow("blur", blurred)
    bin = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        # bin = cv2.dilate(bin, kernel, iterations=1)
    bin = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, kernel)

        #cv2.imshow("Dilate", bin)
        #cv2.imshow("open", bin)
    
    #cv2.imshow("result", bin)
    
    contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True) #0.02
        #print("cnt Before condition",cnt)
        if len(cnt) == 4 and cv2.contourArea(cnt) > 7000 and cv2.contourArea(cnt)<((height*width)*.995): #and cv2.isContourConvex(cnt):
            #print("if have 4 points and it is greater than 7000 pixels",cnt)
            #print("contourArea",cv2.contourArea(cnt))        
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
            #print("max_cos general",max_cos)
            if max_cos < 0.3:  # 0.3
                #print("max_cos appoved",cnt)
                squares.append(cnt)
            
    return squares



def detect_bounding_box(img):
    faces = face_classifier.detectMultiScale(img, 1.1, 5, minSize=(40, 40))
    cropped_faces = []
    
    for (x, y, w, h) in faces:
        cropped_faces.append(img[y:y+h, x:x+w])
        #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return cropped_faces


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
        
        faces = detect_bounding_box(img)
        ##Agregar código para saber si face se encuentra dentro del cuadrilatero
        ##Después de revisar el rostro hacer un análisis de los píxeles al rededor
        #por análissi de textura, local binary pattern, frecuency domain
        quads = find_quads(img)
        #print("quads",quads)
        #result = []
        threshold_similarity = 0.8  
        
        
        ##Normalizar la profundad de Face y Quad
        if(len(faces) != 0) and (len(quads) != 0):
            for face in faces:
                for quad in quads:
                    if len(face.shape) > 2:
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    if len(quad.shape) > 2:
                        quad = cv2.cvtColor(quad, cv2.COLOR_BGR2GRAY)
                        
                    result = cv2.matchTemplate(quad, face, cv2.TM_CCOEFF_NORMED)
                    if result is not None:
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                        if max_val > threshold_similarity:
                            print("The smaller image is inside the larger image.")
        
        
        
        
        #result = cv2.matchTemplate(larger_gray, smaller_gray, cv2.TM_CCOEFF_NORMED)
        #quads = find_quads(img)
        cv2.drawContours(img, quads, -1, (255, 0, 0), 3 )
        cv2.imshow('quads', img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap_video.release()
cv2.destroyAllWindows()