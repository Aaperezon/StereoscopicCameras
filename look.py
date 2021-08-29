from inspect import getmembers, isfunction
from cv2 import cuda
import cv2

for member in getmembers(cv2.threshold, getmembers):
    print(member)
