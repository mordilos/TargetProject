#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 10:54:29 2017

@author: neck
"""

import cv2
import numpy as np
import imutils
import StringIO

cap = cv2.VideoCapture(0)
print('press space to capture prev image!')
while True:
  ret,prev = cap.read()
  cv2.imshow('frame',prev)
  k = cv2.waitKey(1)
  if k==32  :
      break
cv2.destroyWindow('frame')

symbool = cv2.imread('Images/image111.jpg',0)
w, h = symbool.shape[::-1]


img = prev
cv2.imwrite('Output/Andrikelo/prevImg.png',img.copy())
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
found = None
for scale in np.linspace(0.01, 1.0, 20)[::-1]:
    resized = cv2.resize(img_gray, None, fx=scale, fy=scale)
    r = img_gray.shape[1] / float(resized.shape[1])
    if resized.shape[0] < h or resized.shape[1] < w:
        break
    result = cv2.matchTemplate(resized, symbool, cv2.TM_CCOEFF_NORMED)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
    if found is None or maxVal > found[0]:
        found = (maxVal, maxLoc, r)

(maxVal, maxLoc, r) = found
(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
(endX, endY) = (int((maxLoc[0] + w) * r), int((maxLoc[1] + h) * r))

threshold = 0.97
if maxVal >= threshold:
    cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 255), 1)
cv2.imshow('1', img)

newPrevImg = img[startY:endY,startX:endX]
cv2.imshow("newPrevImg",newPrevImg)
cv2.imwrite('Output/Andrikelo/newPrevImg.png',newPrevImg.copy())

print('press space to capture after image!')
while True:
  ret,after = cap.read()
  cv2.imshow('frame',after)
  k = cv2.waitKey(1)
  if k==32 :
      break

cv2.destroyWindow('frame')
img = after
cv2.imwrite('Output/Andrikelo/afterImg.png',img)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
found = None
for scale in np.linspace(0.01, 1.0, 20)[::-1]:
    resized = cv2.resize(img_gray, None, fx=scale, fy=scale)
    r = img_gray.shape[1] / float(resized.shape[1])
    if resized.shape[0] < h or resized.shape[1] < w:
        break
    result = cv2.matchTemplate(resized, symbool, cv2.TM_CCOEFF_NORMED)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
    if found is None or maxVal > found[0]:
        found = (maxVal, maxLoc, r)

(maxVal, maxLoc, r) = found
(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
(endX, endY) = (int((maxLoc[0] + w) * r), int((maxLoc[1] + h) * r))

threshold = 0.97
if maxVal >= threshold:
    cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 255), 1)
cv2.imshow('3', img)

cap.release()

newAfterImg = after[startY:endY,startX:endX]
cv2.imshow("newAfterImg",newAfterImg)
cv2.imwrite('Output/Andrikelo/newAfterImg.png',newAfterImg)

h,w,_ = newPrevImg.shape
newAfterImg = cv2.resize(newAfterImg,(w,h))
cv2.imshow("TELIKO TELIKO newAfterImg",newAfterImg)
cv2.imwrite('Output/Andrikelo/FinalAfterImg.png',newAfterImg)

grayA = cv2.cvtColor(newPrevImg.copy(), cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(newAfterImg.copy(), cv2.COLOR_BGR2GRAY)

sz = newAfterImg.shape

warp_mode = cv2.MOTION_TRANSLATION

if warp_mode == cv2.MOTION_HOMOGRAPHY:
    warp_matrix = np.eye(3,3,dtype=np.float32)
else:
    warp_matrix = np.eye(2,3,dtype = np.float32)
    
number_of_iterations = 5000

termination_eps = 1e-10

criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT , number_of_iterations,termination_eps)

(cc,warp_matrix) = cv2.findTransformECC(grayA,grayB,warp_matrix,warp_mode,criteria)

if warp_mode == cv2.MOTION_HOMOGRAPHY:
    newImgAligned = cv2.warpPerspective(grayB,warp_matrix,(sz[1],sz[0]),flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
else:
    newImgAligned = cv2.warpAffine(grayB,warp_matrix,(sz[1],sz[0]),flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                                   

cv2.imshow("grayA",grayA)
cv2.imshow("grayB",grayB)                    
cv2.imshow("newgrayB",newImgAligned)
cv2.waitKey(0)






blurA = cv2.blur(grayA,(1,1))
cv2.imshow("blurA",blurA)
blurB = cv2.blur(newImgAligned,(1,1))
##################################################################################################################################

absdiff = cv2.absdiff(blurA,blurB)
kernel = np.ones((2,2),np.uint8)
opening = cv2.morphologyEx(absdiff,cv2.MORPH_OPEN,kernel,iterations = 2)

cv2.imshow('opening',opening)
cv2.waitKey(0)

after_thresh = cv2.inRange(opening,0,90)
cv2.imshow('after_thresh',after_thresh)
cv2.waitKey(0)

_,thresh = cv2.threshold(opening,50,255,cv2.THRESH_BINARY)
cv2.imshow('threshold',thresh)
cv2.waitKey(0)

sure_ = cv2.dilate(after_thresh,kernel,iterations=1)
cv2.imshow('sure_',sure_)
cv2.waitKey(0)
cv2.imwrite('Output/Andrikelo/shots.png',sure_)



prev_cp = newPrevImg.copy()

cnts = cv2.findContours(sure_.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#thresh.copy()
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

cnts = sorted(cnts,key=cv2.contourArea)
del cnts[-1]


#FIND THE TARGET CONTOURS
target_contours = cv2.findContours(blurA,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)#~~~~~~~~~
target_contours = target_contours[0] if imutils.is_cv2() else target_contours[1]
for c1 in target_contours:
    M2 = cv2.moments(c1)
    cX2 = int(M2["m10"] / M2["m00"])
    cY2 = int(M2["m01"] / M2["m00"])
    Center = np.array([cX2,cY2])
    
    
temp = prev_cp.copy()    
cv2.drawContours(temp,target_contours,-1,(0,255,255),3)
cv2.drawContours(temp,cnts,-1,(0,255,0),3)
cv2.imshow("temp",temp)
cv2.waitKey(0)

output = StringIO.StringIO()
a = 1
dist = []
centers = []
#FIND COORDINATES AND DISTANCE FROM CENTER
for c in cnts:
    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        shot_ = np.array([cX,cY])
        centers.insert(a,shot_)

        cv2.drawContours(prev_cp,[c],-1,(0,255,0),2)
        cv2.circle(prev_cp,(cX,cY),7,(255,255,255),-1)
        cv2.putText(prev_cp,"center " + str(a) + "@ (" + str(cX) + "," + str(cY)+")",(cX-20,cY-20) ,
                    cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)
        cv2.imshow('target with contours',prev_cp)
        string = "shot " + str(a) + " @ (" + str(cX) + "," + str(cY) + ")\n"
        print(string)
        output.write(string)

        dist.insert(a,np.linalg.norm(Center-shot_))
        
        string = "shot " + str(a) + " distance from center : " + str(dist[a-1]) + "\n"
        print(string)
        output.write(string)
        a=a+1
        print('\n')

cv2.circle(prev_cp,(Center[0],Center[1]),(cX2+cY2)/2,(255,255,0))


rings = [0]
numOfRings = 3
for i in range(1,numOfRings):
    rings.insert(i,int(i*(h/2)/numOfRings))
    cv2.circle (prev_cp,(Center[0],Center[1]),int(i*(h/2)/numOfRings),(255,0,255))
rings.insert(i+1,h/2)
    
    
cv2.imshow("target with contours",prev_cp)
cv2.waitKey(0)
cv2.imwrite('Output/Andrikelo/TargetWithVirtualShots.png',prev_cp)

TargetClasses =[]
for i in range(0,numOfRings):
    TargetClasses.insert(i,[rings[i],rings[i+1]])
string = "TargetClasses : " + str(TargetClasses) + "\n"
print(string)
output.write(string)
print('\n')

score = 0
for i in range(0,a-1):
    print('i = ' + str(i+1))
    for j in range (0,numOfRings-1):
        if ( int(dist[i]) > TargetClasses[j][0]  and int(dist[i]) <= TargetClasses[j][1] and cv2.pointPolygonTest(target_contours[0],(centers[i][0],centers[i][1]),0) >= 0):
            scorei = 100/(j+1)
            score = score + 100/(j+1)
            string = "Shot " + str(i+1) + "\n" + "Target class " + str(j+1) + " = " + str (TargetClasses[j]) + "\n"
            print(string)
            output.write(string)

            string = 'dist ' + str(i+1) + ' = ' + str(int(dist[i])) + "\n"
            print(string)
            output.write(string)

            string = 'score of bullet ' + str(i+1) + ' = ' + str(scorei) + '\n'
            print(string)
            output.write(string)

string = 'final score = ' + str(score)
print(string)
output.write(string)

OutputContents = output.getvalue()
output.close()
f = open("Output/Andrikelo/output.txt",'w')
f.write(OutputContents)
f.close()

cv2.waitKey(0)
cv2.destroyAllWindows()






cv2.destroyAllWindows()