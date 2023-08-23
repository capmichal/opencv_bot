import cv2 as cv
import numpy as np
import os
from time import time
from video_capture import WindowCapture
from vision import Vision
from hsvfilter import HsvFilter

wincap = WindowCapture("METIN2")
vision_kot = Vision("sources/kot-hsv1.png")

vision_kot.init_control_gui()

hsv_filter = HsvFilter(7, 0, 198, 119, 255, 255, 0, 0, 12, 0)

loop_time = time()
while True:
        
    screenshot = wincap.get_screenshot() # using this to perform quick screenshots, cutting out middleman

    processed_image = vision_kot.apply_hsv_filter(screenshot, hsv_filter)

    rectangles = vision_kot.find(processed_image, threshold=0.47, comparison_method=cv.TM_CCOEFF_NORMED)

    output_img = vision_kot.draw_rectangles(screenshot, rectangles)

    cv.imshow("Processed", processed_image)
    cv.imshow("Matches", output_img)

    print("FPS {}".format(1 / (time() - loop_time))) 

    loop_time = time()


    if cv.waitKey(1) == ord("q"):
        cv.destroyAllWindows()
        break


print("Done.")



# CURRENT TODOS
# thresholding --> oddzielanie obrazu ktory szukamy, od tła, wykorzystywanie konrastow
# add HSV (hue, saturation, value) for better image recognition than using basic match_template