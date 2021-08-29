import numpy as np
import cv2 as cv
import time
imgL = cv.imread('tsukuba_L.png',0)
imgR = cv.imread('tsukuba_R.png',0)
#imgL = cv.imread("stereoscopic_left.jpg",0) #left
#imgR = cv.imread("stereoscopic_right.jpg",0) #right
start_time = time.time()

#stereo = cv.StereoBM_create(numDisparities=16,blockSize=5)
stereo = cv.StereoSGBM_create(minDisparity=0,numDisparities=16,blockSize=1,disp12MaxDiff=40)
disparity = stereo.compute(imgL,imgR)
disparity = np.uint8(disparity)


end_time = time.time()
final_timer = (end_time-start_time) 
print("End time"+str(final_timer))
cv.imwrite("salida.jpg",disparity)

cv.imshow("Disparity map",disparity)


cv.waitKey()
cv.destroyAllWindows()

