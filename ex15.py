import cv2
import numpy as np

def get_contour_areas(contours):
    all_areas=[]
    for cnt in contours:
        areas=cv2.contourArea(cnt)
        all_areas.append(areas)
        return all_areas
    
#load the image
img=cv2.imread("pix.png")
original_img=img

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#find canny edge
edged=cv2.Canny(gray,50,200)
cv2.imshow("canny edged",edged)
cv2.waitKey(0)

_,contours,hierarchy=cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

#let's print the area of the contours
print("contour area before sorting")
print(get_contour_areas(contours))

sorted_contours=sorted(contours,key=cv2.contourArea,reverse=True)

print("contour area after sorting")
print(get_contour_areas(sorted_contours))

for c in sorted_contours:
    cv2.drawContours(original_img,[c],-1,(0,255,0),3)
    cv2.waitKey(0)
    cv2.imshow("contours by area",original_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
