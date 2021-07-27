# import the necessary packages
import cv2
from imutils import contours
import numpy as np
import random as rng

import imutils
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border


class sudoku:

    def __init__(self, image):
        self.originalImage = image
        self.originalGray = None


    def find_puzzle(self, debug=False):

        # Convert the image to grayscale and blur it slightly
        self.originalGray = cv2.cvtColor(self.originalImage, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(self.originalGray, (7, 7), 3)
        if debug: cv2.imshow('Blurred image', blurred)

        # Apply adaptive thresholding and then invert the threshold map
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        thresh = cv2.bitwise_not(thresh)       # to remove if u used cv2.THRESH_BINARY_INV
        if debug: cv2.imshow("Threshed image", thresh)

        # Find external contours in the thresholded image and sort them by size in descending order
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)  # or cnts = cnts[0] if len(cnts) == 2 else cnts[1] (depend of version of OpenCV)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        # Initialize a contour that corresponds to the puzzle outline
        self.puzzleCnt = None

        # approximate each contour, and check the first one that has four points
        # then we can assume we have found the outline of the puzzle
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                self.puzzleCnt = approx
                break

        # if the puzzle contour is empty then our script could not find
        # the outline of the Sudoku puzzle so raise an error
        if self.puzzleCnt is None:
            raise Exception(("Could not find Sudoku puzzle outline. "
                "Try debugging your thresholding and contour steps."))

        if debug:
            output = self.originalImage.copy()
            cv2.drawContours(output, [self.puzzleCnt], -1, (0, 255, 0), 2)
            cv2.imshow("Puzzle Outline", output)
            
        # Apply a four point perspective transform to both the original image and grayscale image 
        # to obtain a top-down bird's eye view of the puzzle
        self.warpedPuzzleRGB  = four_point_transform(self.originalImage, self.puzzleCnt.reshape(4, 2))
        self.warpedPuzzleGray = four_point_transform(self.originalGray , self.puzzleCnt.reshape(4, 2))

        if debug:
            # show the output warped image (again, for debugging purposes)
            cv2.imshow("Puzzle Transform", self.puzzleRGB)
            cv2.waitKey(0)
