import cv2
import numpy as np
import sys

MAX_INT = sys.maxsize
coincidence_sad = (MAX_INT,0,0)
coincidence_ssd = (MAX_INT,0,0)



function = {"ssd": lambda base: pow(base,exp=2), "sad": lambda num: abs(num)}

def GetDisparityMap(left_image, right_image, patch, method):
    width_image = len(left_image[0])
    height_image = len(left_image)

    image_depth_method = np.zeros((height_image,width_image,1),np.uint8)

    max_temp = 0 #Remove in future
    for y in range(0,height_image-patch[0]+1):
        last_coincidence_x = 0
        for x in range(0,width_image-patch[1]+1):
            coincidence_method = (MAX_INT,0,0)
            coincidence_ssd = (MAX_INT,0,0)
            if x != 0:
                for x_patch in range(x, x-last_coincidence_x if x > last_coincidence_x else 0, -1):
                    differences = np.subtract(left_image[y:y+patch[0], x:x+patch[1]].tolist(), right_image[y:y+patch[0], x_patch:x_patch+patch[1]].tolist())
                    result_method = function[method](differences).sum()
                    sad = abs(differences).sum()
                    ssd = pow(differences,2).sum()

                    if sad < coincidence_sad[0]:
                        coincidence_sad = (sad,x_patch,y)
                    if ssd < coincidence_ssd[0]:
                        coincidence_ssd = (ssd,x_patch,y)

            image_depth_sad[y,x] = int((x-coincidence_sad[1])*12.75)
            image_depth_ssd[y,x] = int((x-coincidence_ssd[1])*12.75)

            if max_temp < image_depth_sad[y,x]:  #Remove in future
                max_temp = image_depth_sad[y,x]  #Remove in future
    print(max_temp)



def main(): 
    global coincidence_sad,coincidence_ssd
    left_image = cv2.imread("tsukuba_L.png",0) #left
    right_image = cv2.imread("tsukuba_R.png",0) #right

    


    cv2.imshow("Depth SAD",image_depth_sad)
    cv2.imshow("Depth SSD",image_depth_ssd)



if __name__== "__main__":
    main()
    cv2.waitKey()
    cv2.destroyAllWindows()


