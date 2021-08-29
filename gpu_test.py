import cv2

cpu_image = cv2.imread("image.png", 0)

source_gpu = result_gpu = cv2.cuda_GpuMat()

source_gpu.upload(cpu_image)

gpu_thres = cv2.cuda()



      
      