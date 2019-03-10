#!/usr/bin/env python
# coding: utf-8

# importing PIL for only read and write image no other function used

from PIL import Image 
import numpy as np      # numpy for array manupulation
import os  # for creating output directory

# ## Utility functions

try:
	os.makedirs('output') ## create the output directory
except:
	pass	



def clamp(p):  # function clamp pixel between 0 and 255
    if p >= 255:
        return 255
    elif p<0:
        return 0
    else:
        return p
    

def convolve_channel(kernel,image):  # this function do convolution on the single channel image using given kernel
    h,w = image.shape
    pad_val = kernel.shape[0]//2
    padded_image = np.zeros((image.shape[0] + pad_val*2 , image.shape[1] + pad_val*2))
    padded_image[pad_val:-pad_val,pad_val:-pad_val] = image
    image_out = np.zeros_like(image)
    
    # loop across every pixel and finding new value for it
    for x in range(pad_val,w):
        for y in range(pad_val,h):
            roi_x1 = x - pad_val   # ROI for filter and image (X start)
            roi_x2 = x + pad_val   # ROI for filter and image (X end) 
            roi_y1 = y - pad_val   # ROI for filter and image (Y start)
            roi_y2 = y + pad_val   # ROI for filter and image (Y end)
            
            v = np.sum(np.sum(padded_image[roi_y1:roi_y2+1,roi_x1:roi_x2+1] * kernel))
            image_out[y,x] = v
    
    return image_out[:,:] 




def fspecial_gauss(size, sigma):    # function for creating gaussian filter
    
    # gaussian kernel according to gassian formula based on distance from origin

    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


# ## Adjust brightness

print("<<<==== Computing brightness ====>>>")

def brightness(img_in,factor): # this function set brightness according to factor
    h,w = img_in.shape[:2]
    img_out = np.zeros_like(img_in)

    for i in range(h):
        for j in range(w):

            r,g,b = img_in[i,j]
            r = clamp(factor * r)
            g = clamp(factor * g)
            b = clamp(factor * b)
            img_out[i,j] = (r,g,b)
    
    return img_out



img_in = np.asarray(Image.open('input/princeton_small.jpg')) # open image and covert to numpy array


brightness_1 = brightness(img_in , 0.0)  # evluating brightness on 0.0
brightness_2 = brightness(img_in , 0.5)  # evluating brightness on 0.5
brightness_3 = brightness(img_in , 2.0)  # evluating brightness on 2.0


# converting numpy array to PIL image
brightness_1 = Image.fromarray(np.uint8(brightness_1))
brightness_2 = Image.fromarray(np.uint8(brightness_2))
brightness_3 = Image.fromarray(np.uint8(brightness_3))

# saving files
brightness_1.save('output/princeton_small_brightness_0.0.jpg')
brightness_2.save('output/princeton_small_brightness_0.5.jpg')
brightness_3.save('output/princeton_small_brightness_2.0.jpg')


# ## Adjusting contrast




print("<<<==== Computing contrast (large file take upto 2 min) ====>>>")

def change_contrast(img_in , factor):
    
    h,w = img_in.shape[:2]
    img_out = np.zeros_like(img_in)
    avg_luminance = 0

    for i in range(h):
        for j in range(w):
            r,g,b = img_in[i,j]
            avg_luminance += (0.3*r + 0.59*g + 0.11*b)

    for i in range(h):
        for j in range(w):
            r,g,b = img_in[i,j]
            r = (1 - factor)*avg_luminance + factor*r
            g = (1 - factor)*avg_luminance + factor*g
            b = (1 - factor)*avg_luminance + factor*b
            img_out[i,j] = (r,g,b)

    return img_out







img_in = np.asarray(Image.open('input/c.jpg'))

contrast_1 = change_contrast(img_in,-0.5)
contrast_2 = change_contrast(img_in, 0.0)
contrast_3 = change_contrast(img_in, 0.5)
contrast_4 = change_contrast(img_in, 2.0)
contrast_1 = Image.fromarray(np.uint8(contrast_1))
contrast_2 = Image.fromarray(np.uint8(contrast_2))
contrast_3 = Image.fromarray(np.uint8(contrast_3))
contrast_4 = Image.fromarray(np.uint8(contrast_4))
contrast_1.save('output/c_contrast_-0.5.jpg')
contrast_2.save('output/c_contrast_0.0.jpg') 
contrast_3.save('output/c_contrast_0.5.jpg')
contrast_4.save('output/c_contrast_2.0.jpg')


# ## Gaussian blur






print("<<<==== Computing gaussian blur ====>>>")

def gaussian_blur(img_org,sigma):
    img = img_org**(2.2)
    size = sigma*2 +1
    if sigma < 1:
        size = 3
    
    
    
    gaussian_kernel = fspecial_gauss(size ,sigma)
    kernel = gaussian_kernel
    pad_val = kernel.shape[0]//2
    
    r = convolve_channel(kernel , img[:,:,0])
    g = convolve_channel(kernel , img[:,:,1])
    b = convolve_channel(kernel , img[:,:,2])
    
    img_final = np.dstack((r,g,b))
    img_final = img_final**(1.0/2.2)    
    
    return img_final[pad_val:,pad_val:]



img_org = np.asarray(Image.open('input/princeton_small.jpg'))/255.
blurred_1 = gaussian_blur(img_org , 0.125)
blurred_2 = gaussian_blur(img_org , 2)
blurred_3 = gaussian_blur(img_org , 8)



blurred_1 = Image.fromarray(np.uint8(blurred_1*255.))
blurred_2 = Image.fromarray(np.uint8(blurred_2*255.))
blurred_3 = Image.fromarray(np.uint8(blurred_3*255.))
blurred_1.save('output/blur_0.125.jpg')
blurred_2.save('output/blur_2.jpg')
blurred_3.save('output/blur_8.jpg')


# ## Sharpen






print("<<<==== Computing sharpen ====>>>")

def sharpen(img_in):
    img_blurred = gaussian_blur(img_in , 2)
    h,w = img_blurred.shape[:2]
    img = img_in[:h,:w]*255
    img_blurred = img_blurred*255
    img_sharpen = np.zeros_like(img)

    factor = 2
    for i in range(h):
        for j in range(w):
            r,g,b = img[i,j]
            r_blur,g_blur,b_blur = img_blurred[i,j]
            r = clamp((1-factor)*r_blur + factor*r)
            g = clamp((1-factor)*g_blur + factor*g)
            b = clamp((1-factor)*b_blur + factor*b)
            img_sharpen[i,j] = (r,g,b)  
            
    return img_sharpen


img_org = np.asarray(Image.open('input/princeton_small.jpg'))/255.
sharp = sharpen(img_org)
sharp_im = Image.fromarray(np.uint8(sharp))
sharp_im.save('output/sharpen.jpg')


# ## Edge Detection





print("<<<==== Computing edges ====>>>")

def edges(img_org):
    
    Gx = np.array([[-1,0,1],[-2,0,2],[-1,0,+1]])
    Gy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    
    img_gray = 0.30*img_org[:,:,0] + 0.59*img_org[:,:,1] + 0.11*img_org[:,:,2]
    
    h_edge = convolve_channel(Gx,img_gray)
    v_edge = convolve_channel(Gy,img_gray)
    clamp_vec = np.vectorize(clamp)
    h_edge = clamp_vec(h_edge)
    v_edge = clamp_vec(v_edge)
    
    edge = np.sqrt(h_edge**2 + v_edge**2)
    
    return edge



img_org = np.asarray(Image.open('input/princeton_small.jpg'))
edge = edges(img_org)
sharp_im = Image.fromarray(np.uint8(edge))
sharp_im.save('output/edgedetect.jpg')




print("<<<==== All Done open writeup.html file to see results ====>>>")
