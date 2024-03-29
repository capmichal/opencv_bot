import cv2 as cv
import numpy as np
import os
from time import time
import win32gui, win32ui, win32con


class WindowCapture():

    # properties
    w = 0 
    h = 0 
    hwnd = None
    cropped_x = 0 # cropping borders of game window
    cropped_y = 0 #
    offset_x = 0
    offset_y = 0

    def __init__(self, window_name=None):
        
        # if the window is not found just get entire desktop
        if window_name == None:
            self.hwnd = win32gui.GetDesktopWindow() 
        else: 
            self.hwnd = win32gui.FindWindow(None, window_name)


        # getting exact measurements of window with our game
        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = window_rect[2] - window_rect[0]
        self.h = window_rect[3] - window_rect[1] 

        #account for window borders and titlebar
        border_pixels = 8
        titlebar_pixels = 30
        self.w = self.w - (border_pixels * 2)
        self.h = self.h - titlebar_pixels - border_pixels
        self.cropped_x = border_pixels
        self.cropped_y = titlebar_pixels

        self.offset_x = window_rect[0] + self.cropped_x
        self.offset_y = window_rect[1] + self.cropped_y


    def get_screenshot(self):

        
        

        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0,0), (self.w, self.h), dcObj, (self.cropped_x, self.cropped_y), win32con.SRCCOPY)
        #dataBitMap.SaveBitmapFile(cDC, "debug.bmp")

        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype="uint8")
        img.shape = (self.h, self.w, 4)

        # Free Resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        #img = img[...,:3]
        img = np.ascontiguousarray(img)

        return img
    
    def get_screen_position(self, pos): # DOES NOT WORK IF WE WORK A GAME WINDOW WHILE AFTER RUNNING SCRIPT
        return (pos[0] + self.offset_x, pos[1] + self.offset_y)





