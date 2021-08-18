import cv2
import numpy as np


def main(): 
    image1 = cv2.imread("image.png") #original
    image2 = cv2.imread("image1.png") #modified
    #image1 = cv2.resize(image1, (320, 240))
    #image2 = cv2.resize(image2, (320, 240))
    image_result = image1.copy()

    global_sum_absolute = 0
    global_sum_pow2 = 0
    feature_size = 10

    width = len(image1[0])
    height = len(image1)
    for layer in range(0,3):
        for y in range(0,height-feature_size+1):
            for x in range(0,width-feature_size+1):
                differences = np.subtract(image1[y:y+feature_size,x:x+feature_size,layer].tolist(), image2[y:y+feature_size,x:x+feature_size,layer].tolist())
                sad = abs(differences).sum()
                ssd = pow(differences,2).sum()
                global_sum_absolute += sad
                global_sum_pow2 += ssd
                if (sad != 0.0):
                    #cv2.circle(image_result, (y+(feature_size/2),x+(feature_size/2)),15, (0,0,255),1)
                    cv2.rectangle(image_result,(x,y),(x+feature_size,y+feature_size),(0,0,255), 1,cv2.LINE_AA)
    print(f"Absolute method: {global_sum_absolute}")
    print(f"Power 2 method: {global_sum_pow2}")
    cv2.imshow("Result", image_result)


if __name__== "__main__":
    main()
    cv2.waitKey()
    cv2.destroyAllWindows()


