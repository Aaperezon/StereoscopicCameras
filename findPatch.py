import cv2
import numpy as np
import sys

MAX_INT = sys.maxsize
coincidence_sad = (MAX_INT,0,0)
coincidence_ssd = (MAX_INT,0,0)
def main(): 
    global coincidence_sad, coincidence_ssd
    image = cv2.imread("image.png") #original
    patch = cv2.imread("patch.png") #patch
    #print(f"image size {np.shape(image)}")
    #print(f"patch size {np.shape(patch)}")
    width_image = len(image[0])
    height_image = len(image)
    width_patch = len(patch[0])
    height_patch = len(patch)
    #print(f"patch x,y : {width_patch},{height_patch}")

    for layer in range(0,1):
        for y in range(0,height_image-height_patch+1):
            for x in range(0,width_image-width_patch+1):
                differences = np.subtract(image[y:y+height_patch, x:x+width_patch,layer].tolist(), patch[:,:,layer].tolist())
                sad = abs(differences).sum()
                ssd = pow(differences,2).sum()

                if sad < coincidence_sad[0]:
                    coincidence_sad = (sad,x,y)
                if ssd < coincidence_ssd[0]:
                    coincidence_ssd = (ssd,x,y)
    image_result_sad = image.copy()
    image_result_ssd = image.copy()
    
    cv2.rectangle(image_result_sad,(coincidence_sad[1],coincidence_sad[2]),(coincidence_sad[1]+width_patch,coincidence_sad[2]+height_patch), (0,0,255), -1)
    cv2.rectangle(image_result_ssd,(coincidence_ssd[1],coincidence_ssd[2]),(coincidence_ssd[1]+width_patch,coincidence_ssd[2]+height_patch), (0,0,255), -1)
    
    cv2.imshow("Result SAD", image_result_sad)
    cv2.imshow("Result SSD", image_result_ssd)


if __name__== "__main__":
    main()
    cv2.waitKey()
    cv2.destroyAllWindows()


