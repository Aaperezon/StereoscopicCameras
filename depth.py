import cv2
import numpy as np
import sys

MAX_INT = sys.maxsize


function = {"ssd": lambda base: pow(base,exp=2), "sad": lambda num: abs(num)}

def GetDisparityMap(left_image, right_image, patch_size, method = " "):
    width_image = len(left_image[0])
    height_image = len(left_image)

    image_depth_sad = np.zeros((height_image,width_image,1),np.uint8)    #Change for method
    image_depth_ssd = np.zeros((height_image,width_image,1),np.uint8)    #Change for method
    limit_comparation_range_x = 20

    max_temp = 0 #Remove in a future
    for y in range(0,height_image-patch_size[0]+1):
        last_coincidence_x = 0
        for x in range(0,width_image-patch_size[1]+1):
            coincidence_sad = (MAX_INT,0,0) #Change for method
            coincidence_ssd = (MAX_INT,0,0) #Change for method
            if x != 0:
                for x_patch in range(x, x-limit_comparation_range_x if x > limit_comparation_range_x else 0, -1):
                    differences = np.subtract(left_image[y:y+patch_size[0], x:x+patch_size[1]].tolist(), right_image[y:y+patch_size[0], x_patch:x_patch+patch_size[1]].tolist())
                    #result_method = function[method](differences).sum()
                    sad = abs(differences).sum()   #Change for method
                    ssd = pow(differences,2).sum() #Change for method
                    if sad < coincidence_sad[0]:            #Change for method
                        coincidence_sad = (sad,x_patch,y)   #Change for method
                    if ssd < coincidence_ssd[0]:            #Change for method
                        coincidence_ssd = (ssd,x_patch,y)   #Change for method

            image_depth_sad[y,x] = int((x-coincidence_sad[1])*12.75) #Change for method
            image_depth_ssd[y,x] = int((x-coincidence_ssd[1])*12.75) #Change for method

            if max_temp < image_depth_sad[y,x]:  #Remove in a future
                max_temp = image_depth_sad[y,x]  #Remove in a future
    print(max_temp)#Remove in a future
    return image_depth_sad, image_depth_ssd   #Change for signle return


def main(): 
    left_image = cv2.imread("tsukuba_L.png",0) #left
    right_image = cv2.imread("tsukuba_R.png",0) #right

    
    sad_result,ssd_result = GetDisparityMap(left_image, right_image,(3,3))

    cv2.imshow("Depth SAD",sad_result)
    cv2.imshow("Depth SSD",ssd_result)



if __name__== "__main__":
    main()
    cv2.waitKey()
    cv2.destroyAllWindows()


