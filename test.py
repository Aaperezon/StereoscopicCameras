from cv2 import cuda
import cv2
import time
imgL = cv2.imread('tsukuba_L.png',0)
imgR = cv2.imread('tsukuba_R.png',0)



src1 = cv2.cuda_GpuMat()
src2 = cv2.cuda_GpuMat()

src1.upload(imgL)
src2.upload(imgR)

start_time = time.time()
stereo = cv2.cuda_StereoSGM(minDisparity=0,numDisparities=16,blockSize=1,disp12MaxDiff=40)
disparity = stereo.compute(src1,src2)
#disparity = np.uint8(disparity)

final_timer = (time.time()-start_time)

cv2.imshow("result", disparity)
cv2.waitKey(0)