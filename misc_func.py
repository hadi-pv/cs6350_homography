#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 14:02:50 2022

@author: hadi
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    kernel = np.ones((7,7), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    area_thresh = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > area_thresh:
            area_thresh = area
            big_contour = c
            
    
    peri = cv2.arcLength(big_contour, True)
    corners = cv2.approxPolyDP(big_contour, 0.04 * peri, True)
    

    return np.array(corners)

def extra_points(corners):
    lb=corners[0]
    rb=corners[1]
    rt=corners[2]
    lt=corners[3]
    
    points=[]
    points.append((lb+rb)/2)
    points.append((rt+rb)/2)
    points.append((rt+lt)/2)
    points.append((lt+lb)/2)
    
    '''points.append((points[0]+points[1])/2)
    points.append((points[2]+points[1])/2)
    points.append((points[2]+points[3])/2)
    points.append((points[0]+points[3])/2)'''
    
    points=[lb,rb,rt,lt]+points
    
    return np.array(points)

def mse(imageA, imageB):
    imageA = cv2.resize(imageA, (imageB.shape[0],imageB.shape[1]))
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    	
    return err

def rotate_and_correct(im1,im2):
    min_val=mse(im1,im2)
    res=im2
    for each in [cv2.ROTATE_90_CLOCKWISE,cv2.ROTATE_180,cv2.ROTATE_90_COUNTERCLOCKWISE]:
        image = cv2.rotate(im2, each)
        if mse(im1,image)<min_val:
            res=image
            min_val=mse(im1,image)
    return res
    
        
    