import cv2
import numpy as np
import sys

MAX_INT = sys.maxsize

def main(): 
    left_image = cv2.imread("tsukuba_L.png",0) #left
    right_image = cv2.imread("tsukuba_R.png",0) #right

    width_image = len(left_image[0])
    height_image = len(left_image)

    width_patch = 8
    height_patch = 8

    limit_comparation_range_x = 20

    


    image_depth_sad = np.zeros((height_image,width_image,1),np.uint8)
    image_depth_ssd = np.zeros((height_image,width_image,1),np.uint8)

    max_temp = 0
    for y in range(0,height_image-height_patch+1):
        for x in range(0,width_image-width_patch+1):
            coincidence_sad = (MAX_INT,0,0)
            coincidence_ssd = (MAX_INT,0,0)
            if x != 0:
                for x_patch in range(x, x-limit_comparation_range_x if x > limit_comparation_range_x else 0, -1):
                    differences = np.subtract(left_image[y:y+height_patch, x:x+width_patch].tolist(), right_image[y:y+height_patch, x_patch:x_patch+width_patch].tolist())
                    sad = abs(differences).sum()
                    ssd = pow(differences,2).sum()
                    if sad < coincidence_sad[0]:
                        coincidence_sad = (sad,x_patch,y)
                    if ssd < coincidence_ssd[0]:
                        coincidence_ssd = (ssd,x_patch,y)

            image_depth_sad[y,x] = int((x-coincidence_sad[1])*12.75)
            image_depth_ssd[y,x] = int((x-coincidence_ssd[1])*12.75)

            if max_temp < image_depth_sad[y,x]:
                max_temp = image_depth_sad[y,x]
    print(max_temp)
        




    cv2.imshow("Depth SAD",image_depth_sad)
    cv2.imshow("Depth SSD",image_depth_ssd)



if __name__== "__main__":
    main()
    cv2.waitKey()
    cv2.destroyAllWindows()


