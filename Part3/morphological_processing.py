# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 21:06:57 2021

@author: aktas
"""
import cv2
import numpy as np
from scipy import ndimage 

def Erosion_Opencv(img, kernel):
        cv2.imshow("Input Image", img)
        cv2.waitKey(0) 
        erosion = cv2.erode(img,kernel,iterations = 1)
        cv2.imshow("Erosion Output Image", erosion)
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 


img = cv2.imread('er1.png')
img2 = cv2.imread('b2.png', 0)

structe_element = np.ones((7,7),np.uint8)

""" Erosion OpenCV """

Erosion_Opencv(img,structe_element)
Erosion_Opencv(img2,structe_element)

""" Erosion with 2D arrays """

im2 = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 0, 0, 1, 1, 0],
                [0, 1, 1, 0, 0, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]])

structure_element = np.array([[0, 1, 0],[1, 1, 1], [0, 1, 0]])

erode = ndimage.binary_erosion(im2, structure_element).astype(im2.dtype)
print(erode)


def Dilation_Opencv(img, kernel):
        cv2.imshow("Input Image", img)
        cv2.waitKey(0) 
        dilate = cv2.dilate(img,kernel,iterations = 1)
        cv2.imshow("Dilation Output Image", dilate)
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 



""" Dilation Opencv"""


Dilation_Opencv(img,structe_element)
Dilation_Opencv(img2,structe_element)

""" Dilation with 2D arrays """

im2 = np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1],[1, 1, 1, 1]])
structure_element = np.array([[1,1],[1,1]])
dilation = ndimage.binary_dilation(im2, structure_element).astype(im2.dtype)
print(dilation)

""" Opening & Closing Opencv """

img = cv2.imread('noisy.jpeg')
structure_element = np.array([[0, 1, 0],[1, 1, 1], [0, 1, 0]],  np.uint8)
structure_element2 = np.ones((5,5),np.uint8)

print(structure_element)
print(structure_element2)

opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, structure_element)
cv2.imshow("Opening Output Image", opening)
cv2.waitKey(0) 


closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, structure_element)
cv2.imshow("Opening and Closing Output Image", closing)
cv2.waitKey(0) 
cv2.destroyAllWindows()

""" Morphological Edge Detection """

img = cv2.imread('morpho_edge.png')
img = cv2.imread('lena.png')
img = cv2.imread('peppers.png')

# External Boundary Extraction

dilate = cv2.dilate(img,structure_element2,iterations = 1)
ebe = np.subtract(dilate, img)
cv2.imshow("External Boundary Extraction Output Image", ebe)
cv2.waitKey(0) 

# Internal Boundary Extraction

erode = cv2.erode(img,structure_element2,iterations = 1)
ibe = np.subtract(img,erode)
cv2.imshow("Internal Boundary Extraction Output Image", ibe)
cv2.waitKey(0) 


# Morphological Gradient

erode = cv2.erode(img,structure_element2,iterations = 1)
dilate = cv2.dilate(img,structure_element2,iterations = 1)
mg = np.subtract(dilate,erode)
cv2.imshow("Morphological Gradient Output Image", mg)
cv2.waitKey(0) 
cv2.destroyAllWindows()