import cv2
import numpy as np
import sys
import time
from threading import Thread
MAX_INT = sys.maxsize

function = {"ssd": lambda base: pow(base,exp=2), "sad": lambda num: abs(num)}

def CalculateDisparityMap(position,left_image, right_image, image_depth_sad, image_depth_ssd,patch_size, method = " "):
    width_image = len(left_image[0])
    height_image = len(left_image)
    for y in range(0,height_image-patch_size[0]):
        range_last_coincidence_x = 1
        for x in range(0,width_image-patch_size[1]+1):
            coincidence_sad = (MAX_INT,0,0) #Change for method
            coincidence_ssd = (MAX_INT,0,0) #Change for method
            if x != 0:
                for x_patch in range(x, x-range_last_coincidence_x , -1):
                    if x_patch >-1:
                        differences = np.subtract(left_image[y:y+patch_size[0], x:x+patch_size[1]].tolist(), right_image[y:y+patch_size[0], x_patch:x_patch+patch_size[1]].tolist())
                        sad = np.sum(np.absolute(differences))  #Change for method
                        ssd = np.sum(np.power(differences,2)) #Change for method
                        if sad <= coincidence_sad[0]:            #Change for method
                            coincidence_sad = (sad,x_patch,y)   #Change for method
                        if ssd <= coincidence_ssd[0]:            #Change for method
                            coincidence_ssd = (ssd,x_patch,y)   #Change for method
                    #print(f"checking:{x_patch}")
                #print(f"currentX:{x} coincidenceIn:{coincidence_sad[1]}")
                range_last_coincidence_x = (x-coincidence_sad[1]+3) if x > 15 else x
            image_depth_sad[((position-1)*height_image)+y-1,x] = int((x-coincidence_sad[1])*12.75) #Change for method
            image_depth_ssd[((position-1)*height_image)+y-1,x] = int((x-coincidence_ssd[1])*12.75) #Change for method
    return (image_depth_sad, image_depth_ssd)  #Change for signle return
def GetDisparityMap(left_image, right_image, patch_size, method = " ",number_of_threads=4):
    t = [None] * number_of_threads
    width_image = int(len(left_image[0]))
    height_image = int(len(left_image)/number_of_threads)
    image_depth_sad = np.zeros((height_image*number_of_threads,width_image,1),np.uint8)    #Change for method
    image_depth_ssd = np.zeros((height_image*number_of_threads,width_image,1),np.uint8)    #Change for method
    for i in range(1,number_of_threads+1):
        t[i-1] = Thread(target=CalculateDisparityMap, args=(i,left_image[height_image*(i-1):height_image*(i),:],right_image[height_image*(i-1):height_image*(i),:],image_depth_sad,image_depth_ssd,patch_size,method,))
        t[i-1].start()
     

    for thread in t:
        #print(f"waiting for t{id}")
        thread.join()
      
    
    return image_depth_sad, image_depth_ssd

def main(): 
    left_image = cv2.imread("tsukuba_L.png",0) #left
    right_image = cv2.imread("tsukuba_R.png",0) #right
    
    #left_image = cv2.imread("stereoscopic_left.jpg",0) #left
    #right_image = cv2.imread("stereoscopic_right.jpg",0) #right

    start_time = time.time()
    
    #sad_result,ssd_result = GetDisparityMap(left_image, right_image,(3,3),number_of_threads= 5)

    width_image = int(len(left_image[0]))
    height_image = int(len(left_image))

    image_depth_sad = np.zeros((height_image,width_image,1),np.uint8)    #Change for method
    image_depth_ssd = np.zeros((height_image,width_image,1),np.uint8)    #Change for method
    image_depth_sad,image_depth_ssd=CalculateDisparityMap(1,left_image, right_image,image_depth_sad,image_depth_ssd,(3,3),method="")


    
    end_time = time.time()
    final_timer = (end_time-start_time) 
    print(f"Final timer {final_timer}")
    cv2.imshow("Depth SAD  "+str(final_timer),image_depth_sad)
    cv2.imshow("Depth SSD  "+str(final_timer),image_depth_ssd)



if __name__== "__main__":
    main()
    cv2.waitKey()
    cv2.destroyAllWindows()


