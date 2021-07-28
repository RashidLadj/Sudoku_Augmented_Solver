# import the necessary packages
import cv2
from imutils import contours
import numpy as np
import random as rng

import imutils
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border

from keras.models import load_model
from keras.preprocessing.image import img_to_array

from sudoku import Sudoku

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

    def __extract_digit(self, cell, debug=False):
        # apply automatic thresholding to the cell and then clear any
        # connected borders that touch the border of the cell
        thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = clear_border(thresh)
        # check to see if we are visualizing the cell thresholding step
        if debug:
            cv2.imshow("Cell Thresh", thresh)
            cv2.waitKey(0)

        # find contours in the thresholded cell
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # if no contours were found than this is an empty cell
        if len(cnts) == 0:
            return None
        # otherwise, find the largest contour in the cell and create a
        # mask for the contour
        c = max(cnts, key=cv2.contourArea)
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        # compute the percentage of masked pixels relative to the total
        # area of the image
        (h, w) = thresh.shape
        percentFilled = cv2.countNonZero(mask) / float(w * h)
        # if less than 3% of the mask is filled then we are looking at
        # noise and can safely ignore the contour
        if percentFilled < 0.03:
            return None
        # apply the mask to the thresholded cell
        digit = cv2.bitwise_and(thresh, thresh, mask=mask)
        # check to see if we should visualize the masking step
        if debug:
            cv2.imshow("Digit", digit)
            cv2.waitKey(0)
        # return the digit to the calling function
        return digit       

    def create_SudokuMatrix(self, modelPath):
        """
            This method takes as input the warped image representing our sudoku in a perspective view, 
            and returns us the matrix in this sudoku after having extracted the digit of each cell using 
            the method __extract_digit () 
        """

        model = load_model(modelPath)
        inputLayer = model.layers[0]

        # Initialize our 9x9 Sudoku board
        self.board = np.zeros((9, 9), dtype="int")
        # A Sudoku puzzle is a 9x9 grid (81 individual cells), so we can infer the location of each cell 
        # by dividing the warped image into a 9x9 grid
        stepX = self.warpedPuzzleGray.shape[1] // 9
        stepY = self.warpedPuzzleGray.shape[0] // 9
        # Initialize a list to store the (x, y)-coordinates of each cell location
        self.cellLocs = []
        # Loop over the grid locations
        for y in range(0, 9):
            # Initialize the current list of cell locations
            row = []
            for x in range(0, 9):
                # Compute the starting and ending (x, y)-coordinates of the current cell
                startX = x * stepX
                startY = y * stepY
                endX = (x + 1) * stepX
                endY = (y + 1) * stepY
                # Add the (x, y)-coordinates to our cell locations list
                row.append((startX, startY, endX, endY))

                # Crop the cell from the warped transform image and then extract the digit from the cell
                cell = self.warpedPuzzleGray[startY:endY, startX:endX]
                digit = self.__extract_digit(cell, debug=False)
               
                # Verify that the digit is not empty
                if digit is not None:
                    # cv2.imshow("test", digit)
                    # cv2.waitKey()

                    # Resize the cell to 28x28 pixels and then prepare the cell for classification
                    roi = cv2.resize(digit, (28, 28))
                    roi = roi.astype("float") / 255.0
                    roi = img_to_array(roi)           # Add dimention of gray channel (28, 28) --> (28, 28, 1) 
                    roi = np.expand_dims(roi, axis=0) # Add a dimension to prepare the data set to predict (in our case only one data)  (28, 28, 1) --> (1, 28, 28, 1) 

                    # Assert that our data set has the right shape than what the first layer of the model expects  
                    np.testing.assert_allclose(inputLayer.input_shape[1:], roi.shape[1:], atol=1e-4)

                    # Classify the digit and update the Sudoku board with the prediction Let's check:
                    pred = model.predict(roi)
                    pred = pred.argmax(axis=1)
                    pred = pred[0]
                    self.board[y, x] = pred
                # add the row to our cell locations
            self.cellLocs.append(row)

    def show_Sudoku(self):
        # cCnstruct a Sudoku puzzle from the board
        print("[INFO] construct Sudoku board:")
        puzzle = Sudoku(3, 3, board=self.board.tolist())
        puzzle.show()