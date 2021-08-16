# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 15:13:46 2021

@author: aktas
"""
import copy
from nonlinear_filtering import padding, Print_Mode
import cv2
import numpy as np
from enum import Enum
import numpy as np
import scipy.stats as st



class OPERATION_TYPE(Enum):
    Convolution = 1
    Correlation = 2
    
    
def rotate_180(kernel):
    n = len(kernel)
    kernel.reverse()
    for x in range(n):
        for y in range(n-1, x-1, -1):
            kernel[x][y], kernel[y][x] = kernel[y][x], kernel[x][y]
    
    kernel.reverse()
    for x in range(n):
        for y in range(n-1, x-1, -1):
            kernel[x][y], kernel[y][x] = kernel[y][x], kernel[x][y]        
    return(kernel)        

def linear_filtering(image, kernel, padding_size=None, mode=OPERATION_TYPE.Correlation, print_mode = Print_Mode.ON):
    h = len(image)
    w = len(image[0])  
    
    filter_size = (len(kernel), len(kernel[0]))
    
    print("filter size:", filter_size)
        
    if mode == OPERATION_TYPE.Convolution:  #rotate the kernel 180 degree
          kernel = rotate_180(kernel)
        #  print(kernel)
    elif mode ==  OPERATION_TYPE.Correlation:
          pass
    """ Apply padding if a padding size is given as a tuple (padding_width, padding_height) """
    
    if padding_size != None:
       padding(image, padding_size)
       
    
    pad_w = padding_size[0]
    pad_h = padding_size[1] 
    
    h_padded = h + pad_h*2
    w_padded = w + pad_w*2
    
 #   print(h_padded, w_padded)
    
    if print_mode == Print_Mode.ON:
        print("input image with size " + str((h,w)) + "and with padding size" + str(padding_size) + "\n")
        
        for i in range(w):
            for j in range(h):
                print(image[i][j], end=' ')
            print("\n")    
 
    output_image = [[0 for _ in range(w_padded-filter_size[0]+1)] for _ in range(h_padded-filter_size[1]+1)]

    h2 = len(output_image)
    w2 = len(output_image[0])
       

    k = 0
    l = 0
    
    """ apply chosen filter to the input image and create output image"""
    sum_ = 0
    
   
    for i in range(w_padded-filter_size[0]+1):
        for j in range(h_padded-filter_size[1]+1):
            for a in range(filter_size[0]):
                for b in range(filter_size[1]):
                    sum_ += image[i+a][j+b] * kernel[a][b]            
            output_image[k][l] = sum_
            sum_ = 0
            l += 1
            if l >= w2:
                l = 0
                k += 1
    """print the output image """     
    
    if print_mode == Print_Mode.ON:
        print("output image with size " + str((h2,w2)) + " after " + str(mode).split('.')[1] +  " filtering \n")
        
        for i in range(w2):
            for j in range(h2):
                print(output_image[i][j], end=' ')
            print("\n")                

    return output_image       


def Test_Correlation_Convolution_2D():
    im1 = [[0,0,2,5,50,50,100,150,150], 
           [0,10,12,25,15,10,5,200,204],
           [2,1,5,10,100,4,150,178,101],
           [12,10,15,20,101,2,1,5,12],
           [10,3,5,13,72,88,95,4,7],
           [15,0,3,90,102,90,100,101,10],
           [4,4,5,52,18,10,10,17,50],
           [50,50,15,45,100,15,22,1,8],
           [10,4,2,0,0,0,50,5,103]]
    
    
    kernel = [[1,0,1], [2,0,2], [3,0,3]]
   
    input_image = copy.deepcopy(im1)   
    linear_filtering(input_image,kernel,(1,1),mode=OPERATION_TYPE.Convolution)
    
    input_image = copy.deepcopy(im1)   
    linear_filtering(input_image,kernel,(1,1),mode=OPERATION_TYPE.Correlation)
    
    
def gkern(kernlen=11, nsig=1):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return (kern2d/kern2d.sum()).tolist()
    
def Smoothing():
    gaussian1 = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
    gaussian1 = [[element / 16 for element in sub_gaussian] for sub_gaussian in gaussian1]
 
    im2 = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE) 
    cv2.imshow("input_image", im2)
    cv2.waitKey(0) # Press a key, not x (cross) button on coming window
   
    
    input_image = im2.tolist()
    output_image = np.array(linear_filtering(input_image,gaussian1, (1,1), mode=OPERATION_TYPE.Convolution, print_mode = Print_Mode.OFF), dtype=np.uint8)
    cv2.imshow("3X3 Gaussian Blur Output Image", output_image)
    cv2.waitKey(0) # Press a key, not x (cross) button on coming window
    
    
    gaussian2 = [[1, 4, 7, 4 ,1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4 ,1]]
    gaussian2 = [[element / 273 for element in sub_gaussian] for sub_gaussian in gaussian2]
    
    input_image = im2.tolist()
    output_image = np.array(linear_filtering(input_image,gaussian2, (2,2), mode=OPERATION_TYPE.Convolution, print_mode = Print_Mode.OFF), dtype=np.uint8)
    cv2.imshow("5X5 Gaussian Blur Output Image", output_image)
    cv2.waitKey(0) # Press a key, not x (cross) button on coming window
    
    gaussian3 = gkern(11,1)
    input_image = im2.tolist()
    output_image = np.array(linear_filtering(input_image,gaussian3, (5,5), mode=OPERATION_TYPE.Convolution, print_mode = Print_Mode.OFF), dtype=np.uint8)
    cv2.imshow("11x11 Gaussian Blur Output Image", output_image)
    cv2.waitKey(0) # Press a key, not x (cross) button on coming window
        
    cv2.destroyAllWindows() 
    
def Opencv_Smoothing():
    img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE) 
    img = cv2.GaussianBlur(img,(11,11),1)    
    cv2.imshow("11X11 Gaussian Blur Output Image", img)
    cv2.waitKey(0) # Press a key, not x (cross) button on coming window
    cv2.destroyAllWindows() 
    
def UnsharpMask_Sharpening(blur_kernel_size, k):
    """Blur Step"""
    gaussian_kernel = gkern(blur_kernel_size,1)
    
    padding_size = int((blur_kernel_size - 1) / 2)
      
    im2 = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE) 
    
    input_image = im2.tolist()
    tmp = copy.deepcopy(input_image)   
   
    blurred_image = np.array(linear_filtering(tmp,gaussian_kernel, (padding_size,padding_size), mode=OPERATION_TYPE.Convolution, print_mode = Print_Mode.OFF), dtype=np.uint8)
    
    mask = np.subtract(im2, blurred_image)
    
    weighted_mask = [[element * k for element in sub_mask] for sub_mask in mask]
    
    output_image = np.add(input_image,weighted_mask)
        
    output_image = np.array(output_image, dtype=np.uint8)
    cv2.imshow("Unsharp Masking", output_image)
    cv2.waitKey(0) 
        
    cv2.destroyAllWindows()     
    
    
def Opencv_UnsharpMask_Sharpening(blur_kernel_size,k):
    img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE) 
    img_blurred = cv2.GaussianBlur(img,(blur_kernel_size,blur_kernel_size),1)
    mask = cv2.addWeighted(img, 1, img_blurred, -1, 0)  #input_image - blurred_image
    output_image = cv2.addWeighted(img, 1, mask, k, 0)  # input_image + k * mask
    
    cv2.imshow("Opencv Unsharp Masking", output_image)
    cv2.waitKey(0) 
        
    cv2.destroyAllWindows()   

def HighPass_Sharpening():   
    kernel1 = [[-1, -1, -1],[-1,  8, -1],[-1, -1, -1]]
    kernel2 = [[-1, -1, -1, -1, -1], [-1,  1,  2,  1, -1],[-1,  2,  4,  2, -1],[-1,  1,  2,  1, -1], [-1, -1, -1, -1, -1]]
   
    im2 = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE) 
    cv2.imshow("input_image", im2)
    cv2.waitKey(0) # Press a key, not x (cross) button on coming window
   
    
    input_image = im2.tolist()
    output_image = np.array(linear_filtering(input_image,kernel1, (1,1), mode=OPERATION_TYPE.Convolution, print_mode = Print_Mode.OFF), dtype=np.uint8)
    cv2.imshow("High Pass 3x3 Output Image", output_image)
    cv2.waitKey(0) # Press a key, not x (cross) button on coming window
    
    input_image = im2.tolist()
    output_image = np.array(linear_filtering(input_image,kernel1, (1,1), mode=OPERATION_TYPE.Convolution, print_mode = Print_Mode.OFF), dtype=np.uint8)
    cv2.imshow("High Pass 5x5  Output Image", output_image)
    cv2.waitKey(0) # Press a key, not x (cross) button on coming window
    
    cv2.destroyAllWindows()   
   
def Mean_Filter(kernel_size):
    kernel = [[1/kernel_size**2 for _ in range(kernel_size)] for _ in range(kernel_size)]
    
    im2 = cv2.imread("noisy.jpeg", cv2.IMREAD_GRAYSCALE) 
    cv2.imshow("input_image", im2)
    cv2.waitKey(0) # Press a key, not x (cross) button on coming window
    input_image = im2.tolist()
    padding_size = int((kernel_size - 1) /2) 
    output_image = np.array(linear_filtering(input_image,kernel, (padding_size,padding_size), mode=OPERATION_TYPE.Convolution, print_mode = Print_Mode.OFF), dtype=np.uint8)
    cv2.imshow("Mean Filter  Output Image", output_image)
    cv2.waitKey(0) # Press a key, not x (cross) button on coming window
    
    cv2.destroyAllWindows() 
    

def Edge_Detection(kernel):
    kernel_size = len(kernel)
    padding_size = int((kernel_size - 1) / 2)
   
    im2 = cv2.imread("cameraman.tif", cv2.IMREAD_GRAYSCALE) 
    input_image = im2.tolist()
    
    # gaussian2 = [[1, 4, 7, 4 ,1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4 ,1]]
    # gaussian2 = [[element / 273 for element in sub_gaussian] for sub_gaussian in gaussian2]
    # blurred = linear_filtering(input_image,gaussian2, (2,2), mode=OPERATION_TYPE.Convolution, print_mode = Print_Mode.OFF)
    
    
    output_image = np.array(linear_filtering(input_image,kernel, (padding_size,padding_size), mode=OPERATION_TYPE.Convolution, print_mode = Print_Mode.OFF), dtype=np.uint8)
    cv2.imshow("Edge_Detection Output Image", output_image)
    cv2.waitKey(0) # Press a key, not x (cross) button on coming window
    cv2.destroyAllWindows()
    
    
def FirstDerivative_Opencv():    
    img = cv2.imread("cameraman.tif", cv2.IMREAD_GRAYSCALE) 
    img_gaussian = cv2.GaussianBlur(img,(3,3),0)
    
    """Sobel"""
    img_sobelx = cv2.Sobel(img_gaussian,cv2.CV_8U,1,0,ksize=5)
    img_sobely = cv2.Sobel(img_gaussian,cv2.CV_8U,0,1,ksize=5)
    img_sobel = img_sobelx + img_sobely
    
    cv2.imshow("Sobel 5x5 Opencv Output", img_sobel)
    cv2.waitKey(0) # Press a key, not x (cross) button on coming window
      
    """Prewitt"""
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
    img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
        
    img_prewitt = img_prewittx + img_prewitty
    
    cv2.imshow("Prewitt 5x5 Opencv Output", img_prewitt)
    cv2.waitKey(0) # Press a key, not x (cross) button on coming window
    cv2.destroyAllWindows()
    
def SecondDerivative_Opencv(): 
    img = cv2.imread("cameraman.tif", cv2.IMREAD_GRAYSCALE) 
    img_gaussian = cv2.GaussianBlur(img,(3,3),0)
    laplacian = cv2.Laplacian(img_gaussian,cv2.CV_8U,ksize=5) 
    cv2.imshow("Laplacian 5x5 Opencv Output", laplacian)
    cv2.waitKey(0) # Press a key, not x (cross) button on coming window
    cv2.destroyAllWindows()
   
#Smoothing()   
#Opencv_Smoothing()
#UnsharpMask_Sharpening(5,2)
#Opencv_UnsharpMask_Sharpening(5,2)
#HighPass_Sharpening()
# Mean_Filter(3)
# Mean_Filter(9)
# Mean_Filter(25)  

"""Prewitt Kernels"""

# prewitt3x = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
# prewitt3y = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]

# prewitt5x = [[-2, -2, -2, -2, -2], [-1, -1, -1, -1, -1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1],  [2, 2, 2, 2, 2]]
# prewitt5y = [[-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2]]

# First_Derivative(prewitt3x)
# First_Derivative(prewitt3y)

# First_Derivative(prewitt5x)
# First_Derivative(prewitt5y)


"""Sobel Kernels"""

# sobel3x = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
# sobel3y = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

# sobel5x = [[-2, -2, -4, -2, -2], [-1, -1, -2, -1, -1], [0, 0, 0, 0, 0], [1, 1, 2, 1, 1],  [2, 2, 4, 2, 2]]
# sobel5y = [[-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2], [-4, -2, 0, 2, 4], [-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2]]

# Edge_Detection(sobel3x)
# Edge_Detection(sobel3y)

# Edge_Detection(sobel5x)
# Edge_Detection(sobel5y)

#FirstDerivative_Opencv()

"""Laplacian Kernels"""

# laplacien1 = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
# laplacien2 = [[1, 1, 1], [1, -8, 1], [1, 1, 1]]

# Edge_Detection(laplacien1)
# Edge_Detection(laplacien2)

#SecondDerivative_Opencv()