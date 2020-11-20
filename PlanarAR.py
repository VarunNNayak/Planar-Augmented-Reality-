# Undistorting the checkerboard and computing a 3*3 homography to set an image ovelay on the checkerboard 
import numpy as np
import cv2
import signal
import sys
import glob

cap = cv2.VideoCapture(0)

#now providing image so that it will be overlayed on the checkerboard
#reading the image from the defined subject
images = glob.glob('IMG_20191102_124031.jpg') 
#the provided image is selected
currentImg=0  
#now the defined image is read using 'imread' function
replaceImg=cv2.imread(images[currentImg])
#obtaining the dimensions of the image (height,width & channels)
height,width,ch = replaceImg.shape
#these dimensions decide the image points on the checkerboard
pt1 = np.float32([[0,0],[width,0],[width,height],[0,height]])    

maskThreshold=10

while(True):
    #capturing a frame (frame by frame)
    ret, img = cap.read()
    # converts the image from one color space to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #It return the corners points of the chessboard
    # considering height-9 and width-6 as the number of corners points on the checkerboard
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    if ret == True:
        #finding checkerboard corners and adding object points, image points (after refining them)
        #pt2 is used for calculating the perspective transform matrix of the checkerboard
        pt2 = np.float32([corners[0,0],corners[8,0],corners[len(corners)-1,0],corners[len(corners)-9,0]])
        #perspective transform matrix is calculated from four pairs of points 
        M = cv2.getPerspectiveTransform(pt1,pt2)
        #obtaining the dimensions of the image (height,width & channels)
        height,width,ch = img.shape
        #applies a perspective transformation to the image.
        dst = cv2.warpPerspective(replaceImg,M,(width,height))
        #mask function is used for adding the two images
        #maskThreshold is used to substract the black background from different image
        ret, mask = cv2.threshold(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY),maskThreshold, 1, cv2.THRESH_BINARY_INV)     
        #erode and dilate  commands are used to denoise which are present in the image
        mask = cv2.erode(mask,(3,3))
        mask = cv2.dilate(mask,(3,3))         
        #both the images are added using the mask function so that the image will be overlayed on the checkerboard
        for c in range(0,3):
            img[:, :, c] = dst[:,:,c]*(1-mask[:,:]) + img[:,:,c]*mask[:,:]
     #displaying the output image
    cv2.imshow('img',img)  
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# release everything 
cap.release()
cv2.destroyAllWindows()