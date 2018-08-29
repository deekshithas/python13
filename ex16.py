import cv2
import numpy as np

device =cv2.VideoCapture(0)
while True:
    ret,frame=device.read()
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    lower_range_blue=np.array([30,150,50])
    upper_range_blue=np.array([255,255,180])
    mask=cv2.inRange(hsv,lower_range_blue,upper_range_blue)

    lower_range_red=np.array([150,70,30])
    upper_range_red=np.array([180,255,150])
    mask1=cv2.inRange(hsv,lower_range_red,upper_range_red)

    cv2.imshow("show",frame)

    result1=cv2.bitwise_and(frame,frame,mask=mask1)
    result2=cv2.bitwise_and(frame,frame,mask=mask1)

    cv2.imshow("show1",result1)
    cv2.imshow("show1",result1)

    if cv2.waitKey(1)==13:
        break

    device.release()
    cv2.destroyAllWindows()

