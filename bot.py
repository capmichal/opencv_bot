import cv2 as cv  # version 4.7.0
import numpy as np

def alchemik():
    gra_img = cv.imread("sources/cala_gra.png", cv.IMREAD_UNCHANGED)
    alchemik_img = cv.imread("sources/alchemik.png", cv.IMREAD_UNCHANGED) # downsized two times IMREAD_REDUCED_COLOR_2

    result = cv.matchTemplate(gra_img, alchemik_img, cv.TM_CCOEFF_NORMED) # returns matrix of confidence score, experiment with this form of matching --> TM_CCORR_NORMED seems to be getting the best results

    _, max_val, _, max_loc = cv.minMaxLoc(result) 
    print("Best match top left position: ", max_loc)
    print("Best match confidence: ", max_val)

    threshold = 0.7

    if max_val >= threshold:
        print("Found alchemik")

        alchemik_w = alchemik_img.shape[1]
        alchemik_h = alchemik_img.shape[0]

        top_left = max_loc
        bottom_right = (top_left[0] + alchemik_w, top_left[1] + alchemik_h)

        cv.rectangle(gra_img, top_left, bottom_right, color=(0,255,0), thickness=2, lineType=cv.LINE_4)
        cv.circle(gra_img, top_left, 20, color=(0,255,0))

        cv.imwrite("outcome/result.jpg", gra_img)


    else:
        print("Alchemik not found")

    return 0


def kot(img, item): 

    gra_img = cv.imread(img, cv.IMREAD_UNCHANGED)
    kot_img = cv.imread(item, cv.IMREAD_UNCHANGED)

    result = cv.matchTemplate(gra_img, kot_img, cv.TM_CCOEFF_NORMED)

    _, max_val, _, max_loc = cv.minMaxLoc(result) 

    threshold = 0.95 

    locations = np.where(result > threshold)
    locations = list(zip(*locations[::-1])) 
    print(len(locations))

     

    kot_w = kot_img.shape[0]
    kot_h = kot_img.shape[1]

    click_points = []

    if locations:
        for location in locations:
            top_left = location
            bottom_right = (top_left[0] + kot_w, top_left[1] + kot_h)

            cv.rectangle(gra_img, top_left, bottom_right, color=(0,255,0), thickness=2, lineType=cv.LINE_4)
            
            center_x = top_left[0] + int(kot_w / 2)
            center_y = top_left[1] + int(kot_h / 2)

            cv.drawMarker(gra_img, (center_x, center_y), (0,0,255), cv.MARKER_CROSS)

            click_points.append((center_x, center_y))
            
            cv.imwrite("outcome/result_kot.jpg", gra_img)
    else:
        print("nie ma kotow")


    return click_points


kot("sources/cala_gra4.png", "sources/kot3.png")



