#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 13:40:09 2019

@author: alexdrake
"""

import numpy as np
import cv2

# create the cascade
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

img = cv2.imread('images/blur.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, 
                                     minSize=(30, 30))

print("Found {0} faces".format(len(faces)))

# continue only if faces are detected!
if len(faces) != 0:
    for f in faces:
        #x, y, w, h = [i for i in f]
        x, y, w, h = f
        
        cv2.rectangle(img, (x, y), (x+w,y+h), (255,255,0), 2)
        face_image = img[y:y+h, x:x+w]
        
        face_image = cv2.GaussianBlur(face_image, (23, 23), 30)
        img[y:y+face_image.shape[0], x:x+face_image.shape[1]] = face_image
    
#cv2.imshow("Faces detected", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(".results/blur_detected.png", img)
