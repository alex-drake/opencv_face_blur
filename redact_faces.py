#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 13:40:09 2019

@author: alexdrake
"""

import cv2
import os

# create the cascade
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier('cascades/haarcascade_profileface_default.xml')

included_extensions = ['jpg','jpeg', 'bmp', 'png'] # allowed extensions

# get list of files in the images directory
file_names = [fn for fn in os.listdir('images/')
              if any(fn.endswith(ext) for ext in included_extensions)]

# apply gaussian blur to all faces in all image files
for file in file_names:
    #img = cv2.imread('images/blur.png') # for testing
    img = cv2.imread('images/'+file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, 
                                         minSize=(30, 30))
    profiles = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                minSize=(30,30))
    print("Found {0} faces and {1} profiles".format(len(faces),len(profiles)))
    
    # first deal with frontal faces
    for (x, y, w, h) in faces:
        face_image = img[y:y+h, x:x+w] # location of face in image
        face_image = cv2.GaussianBlur(face_image, (11, 13), 30) # apply gaussian blur to area
        img[y:y+face_image.shape[0], x:x+face_image.shape[1]] = face_image # add blur to original image
  
    # now deal with faces in profile
    for (x, y, w, h) in profiles:
        face_image = img[y:y+h, x:x+w]  # location of face in image
        face_image = cv2.GaussianBlur(face_image, (11,13), 30) # apply gaussian blur to area
        img[y:y+face_image.shape[0], x:x+face_image.shape[1]] = face_image # add blur to original image

    # save image
    cv2.imwrite('results/' + file.split('.')[0] + '_detected.png', img) 
