#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 23:35:46 2018

@author: neck
"""

###############################################################################
###############################################################################
######################                                   ######################
######################              Imports              ######################
######################                                   ######################
###############################################################################
###############################################################################
import cv2
import numpy as np
import imutils
import StringIO
from skimage.util import img_as_ubyte
from skimage.measure import compare_ssim


###############################################################################
###############################################################################
######################                                   ######################
######################              Functions            ######################
######################                                   ######################
###############################################################################
###############################################################################

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt , cropping , PtList

	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
    if event == cv2.EVENT_LBUTTONDOWN:
		cropping = True

    if event == cv2.EVENT_LBUTTONUP:
        refPt.append((x,y))
        cropping = False
        cv2.circle(image,refPt[PtList],3,(0,255,0),-1)
        cv2.imshow("image", image)
        PtList += 1

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
    return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
    dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")

	# compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# return the warped image
    return warped

def getImgPrev(cap,dx,dy,grid_color):

    print('press space to capture prev image!')
    while True:
        ret,prev = cap.read()
        prev2 = prev.copy()
        prev2[:,::dy,:] = grid_color
        prev2[::dx,:,:] = grid_color
        cv2.imshow('frame',prev2)
        k = cv2.waitKey(1)
        if k==32  :
            break

    cv2.destroyWindow('frame')

    #img = prev_image.copy()
    #clone = img.copy()
    cv2.imwrite('/home/neck/Documents/project/Output/Andrikelo/prevImg.png',prev.copy())

    return  prev.copy(), prev2.copy()

def getImgAfter(cap,dx,dy,grid_color):

    print('press space to capture after image!')
    while True:
      ret,after = cap.read()
      after2 = after.copy()
      after2[:,::dy,:] = grid_color
      after2[::dx,:,:] = grid_color
      cv2.imshow('frame',after2)
      k = cv2.waitKey(1)
      if k==32 :
          break

    cv2.destroyWindow('frame')
    cap.release()

    cv2.imwrite('/home/neck/Documents/project/Output/Andrikelo/afterImg.png',after.copy())

    return after.copy(),after2.copy()

def tfImgPrev(prev,prevWgrid):
    global image,refPt, cropping , PtList
    refPt = []
    cropping = False
    PtList = 0

    image = prevWgrid.copy()
    clone = prevWgrid.copy()

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)

    # keep looping until the 'q' key is pressed
    while True:
    	# display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            prevWgrid = clone.copy()
        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break

    roiPrev = four_point_transform(prev,np.asarray(refPt))
    previousPts = refPt
    h,w,_= roiPrev.shape
    return roiPrev,previousPts,(h,w)

def tfImgAfter(after,afterWgrid,previousPts,(h,w)):
    global image,refPt, cropping , PtList

    refPt = []
    cropping = False
    PtList = 0

    for i in range(0,len(previousPts)):
        cv2.circle(afterWgrid,previousPts[i],3,(0,0,255),-1)

    image = afterWgrid.copy()

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)

    clone = afterWgrid.copy()

    # keep looping until the 'q' key is pressed
    while True:
    	# display the image and wait for a keypress
    	cv2.imshow("image", image)
    	key = cv2.waitKey(1) & 0xFF

    	# if the 'r' key is pressed, reset the cropping region
    	if key == ord("r"):
    		image = clone.copy()

    	# if the 'c' key is pressed, break from the loop
    	elif key == ord("c"):
    		break

    # if there are two reference points, then crop the region of interest
    # from teh image and display it
    roiAfter = four_point_transform(after,np.asarray(refPt))
    roiAfter = cv2.resize(roiAfter,(w,h))

    return roiAfter

def ImageDifference(cropped_img_prev,cropped_img_after):
    h,w,_ = cropped_img_prev.shape
    cropped_img_after = cv2.resize(cropped_img_after,(w,h))

    prev = cv2.cvtColor(cropped_img_prev, cv2.COLOR_BGR2GRAY)
    after = cv2.cvtColor(cropped_img_after, cv2.COLOR_BGR2GRAY)

    prev = img_as_ubyte(prev)
    after = img_as_ubyte(after)

    cv2.imwrite('/home/neck/Documents/project/Output/Andrikelo/prevImgFinal.png',prev.copy())
    cv2.imwrite('/home/neck/Documents/project/Output/Andrikelo/afterImgFinal.png',after.copy())

    blur_prev = cv2.blur(prev,(5,5))
    blur_after = cv2.blur(after,(5,5))

    (score, diff) = compare_ssim(blur_prev, blur_after, full=True)
    diff = (diff * 255).astype("uint8")#255
    print("SSIM: {}".format(score))

    cv2.namedWindow('Mask')
    threshLow = 30

    while(True):
#        diff_copy = diff_copy_copy.copy()
        cv2.createTrackbar('Threshold','Mask',threshLow,255,callback)

        threshLow = cv2.getTrackbarPos('Threshold','Mask')

        thresh = cv2.threshold(diff, threshLow, 255,	cv2.THRESH_BINARY_INV)[1]# | cv2.THRESH_OTSU
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        cv2.imshow('Mask',thresh)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('c') or k == 113:
            break

    #thresh = cv2.threshold(diff, 30, 255,	cv2.THRESH_BINARY_INV)[1]
    return thresh,diff,prev,after

def callback(x):
    pass

def FindPrevContours(cropped_img_prev):
    blur = cv2.GaussianBlur(cropped_img_prev,(5,5),0)
    prev_copy_copy = cropped_img_prev.copy()

    cv2.namedWindow('Mask')
    threshLow = 100

    while(True):
        prev_copy = prev_copy_copy.copy()
        cv2.createTrackbar('Threshold','Mask',threshLow,255,callback)
        threshLow = cv2.getTrackbarPos('Threshold','Mask')

        thresh_prev = cv2.threshold(cv2.bitwise_not(blur),threshLow,255,cv2.THRESH_BINARY)[1]
        thresh_prev = cv2.erode(thresh_prev, None, iterations=2)
        thresh_prev = cv2.dilate(thresh_prev, None, iterations=2)

        prev_cnts = cv2.findContours(cv2.bitwise_not(thresh_prev), cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
        prev_cnts = prev_cnts[0] if imutils.is_cv2() else prev_cnts[1]

        if(len(prev_cnts)>0):
            max_prev_cnts = max(prev_cnts, key=cv2.contourArea)
            cv2.drawContours(prev_copy,[max_prev_cnts],-1,(0,255,0),-1)
        cv2.imshow('Mask',prev_copy)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('c') or k == 113:
            break

    return max_prev_cnts


def FindContours(thresh,cropped_img_prev,cropped_img_after,diff):
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # loop over the contours
    for c in cnts:
    	# compute the bounding box of the contour and then draw the
    	# bounding box on both input images to represent where the two
    	# images differ
    	(x, y, w, h) = cv2.boundingRect(c)
    	cv2.rectangle(cropped_img_prev, (x, y), (x + w, y + h), (0, 0, 255), 2)
    	cv2.rectangle(cropped_img_after, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # show the output images
    cv2.imshow("Original", cropped_img_prev)
    cv2.imshow("Modified", cropped_img_after)
    cv2.imshow("Diff", diff)
    cv2.imshow("Thresh", thresh)
    #cv2.imshow('blur',blur)
    cv2.waitKey(0)
    return cnts,cropped_img_prev,cropped_img_after,diff,thresh

def FindCenters(cnts,cropped_img_prev,cropped_img_after,output):
    a = 1
    centers = []
    #FIND COORDINATES OF CONTOURS
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX,cY))

            cv2.putText(cropped_img_prev,"center " + str(a) + "@ (" + str(cX) + "," + str(cY)+")",
                        (cX-20,cY-20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,150),1)
            cv2.imshow('target with contours',cropped_img_prev)

            string = "shot " + str(a) + " @ (" + str(cX) + "," + str(cY) + ")\n"
            print(string)
            print('\n')
            output.write(string)
            a=a+1
    cv2.imwrite('/home/neck/Documents/project/Output/Andrikelo/TargetWithVirtualShots.png',cropped_img_prev.copy())
    return output,centers

def Scoring(output,centers,prev_cnts):

    score = 0
    for i in range(0,len(centers)):
        print('i = ' + str(i+1))
        if(cv2.pointPolygonTest(prev_cnts,(centers[i][0],centers[i][1]),0)>0):
            scorei = 1
            score = score + scorei
            string = "Shot " + str(i+1) + "\n" + "Target class  = on \n"
            print(string)
            output.write(string)
            string = 'score of bullet ' + str(i+1) + ' = ' + str(scorei) + '\n'
            print(string)
            output.write(string)
        else:
            scorei = 0
            score = score + scorei

    string = 'final score = ' + str(score)
    print(string)
    output.write(string)

    OutputContents = output.getvalue()
    output.close()

    f = open("/home/neck/Documents/project/Output/Andrikelo/output.txt",'w')
    f.write(OutputContents)
    f.close()


###############################################################################
###############################################################################
######################                                   ######################
######################              Main Loop            ######################
######################                                   ######################
###############################################################################
###############################################################################

refPt = []
cropping = False
PtList = 0
dx = 50
dy = 50
grid_color = [255,255,0]

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,1024)

prev,prev2 = getImgPrev(cap,dx,dy,grid_color)

after,after2 = getImgAfter(cap,dx,dy,grid_color)

cap.release()

roiPrev,previousPts,(h,w) = tfImgPrev(prev,prev2)

roiAfter = tfImgAfter(after,after2,previousPts,(h,w))

thresh,diff,prev,after = ImageDifference(roiPrev,roiAfter)

prev_cnts = FindPrevContours(prev)

cnts,prev,after,diff,thresh = FindContours(thresh,prev,after,diff)

output = StringIO.StringIO()

output,centers = FindCenters(cnts,prev,after,output)

Scoring(output,centers,prev_cnts)

cv2.waitKey(0)
cv2.destroyAllWindows()
