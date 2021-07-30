# Import the necessary packages
import numpy as np
import cv2

import imutils
from imutils.perspective import four_point_transform
from imutils.perspective import order_points
from skimage.segmentation import clear_border

from keras.models import load_model
from keras.preprocessing.image import img_to_array

from sudoku import Sudoku

class sudoku:

    def __init__(self, imagePath):
        self.originalImage = cv2.imread(imagePath)
        self.originalImage =imutils.resize(self.originalImage, width=600)
        self.originalGray  = None
        self.puzzleCnt        = None # Initialize a contour that corresponds to the puzzle outline
        self.warpedPuzzleRGB  = None
        self.warpedPuzzleGray = None
        self.board = np.zeros((9, 9), dtype="int") # Initialize our 9x9 Sudoku board
        self.cellLocs         = None
        self.boardSolution    = None

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
            # Show the output warped image (again, for debugging purposes)
            cv2.imshow("Puzzle Transform", self.warpedPuzzleRGB)
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

                # Add the row to our cell locations
            self.cellLocs.append(row)

    def solveAndShow_Sudoku(self):
        # construct a Sudoku puzzle from the board
        print("[INFO] construct Sudoku board:")
        puzzle = Sudoku(3, 3, board=self.board.tolist())
        puzzle.show()

        # Solve a Sudoku puzzle using Sudoku library
        print("[INFO] solving Sudoku puzzle...")
        self.boardSolution = puzzle.solve()
        self.boardSolution.show_full()

    def generateWarpedSolution(self, debug = False):
        """
            Generation of the solution on the warped puzzle image with black background and red forground (without display all of puzzle - only solution).
        """
        # Create new image 
        self.warpedResultRGB  = np.zeros((self.warpedPuzzleGray.shape[0], self.warpedPuzzleGray.shape[1], 3), np.uint8)
        self.warpedResultMask = np.zeros((self.warpedPuzzleGray.shape[0], self.warpedPuzzleGray.shape[1], 1), np.uint8)
        # Loop over the cell locations and board
        for (cellRow, boardRowSol, boardRowInitial) in zip(self.cellLocs, self.boardSolution.board, self.board):
            # Loop over individual cell in the row
            for (cellBox, digitSol, digitInitial) in zip(cellRow, boardRowSol, boardRowInitial):
                if digitInitial == 0:
                    # Unpack the cell coordinates
                    startX, startY, endX, endY = cellBox
                    # Compute the coordinates of where the digit will be drawn on the output puzzle image
                    textX = int((endX - startX) * 0.33)
                    textY = int((endY - startY) * -0.2)
                    textX += startX
                    textY += endY
                    # Draw the result digit on the Sudoku puzzle image
                    cv2.putText(self.warpedResultRGB , str(digitSol), (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.putText(self.warpedResultMask, str(digitSol), (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255), 2)
        if debug:
            # Show the output image
            cv2.imshow("mySudoku Warped solution", self.warpedResultRGB)
            cv2.waitKey(0)

    def generateSolution(self):
        """
            Generation of the solution on the original puzzle image with filling only empty cells 
        """
        height = self.warpedPuzzleGray.shape[0]
        width  = self.warpedPuzzleGray.shape[1]

        # Define the 4 points representing the corners of our Sudoku puzzle
        srcPoints = np.array([
                    [0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1]], dtype="float32")
        srcPoints = order_points(srcPoints.reshape(4, 2))

        # Define the equivalent 4 points representing the corners of our Sudoku puzzle in the original image
        self.puzzleCnt = order_points(self.puzzleCnt.reshape(4, 2))


        # Compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(srcPoints, self.puzzleCnt)

        height = self.originalImage.shape[0]
        width  = self.originalImage.shape[1]

        result = cv2.warpPerspective(self.warpedResultRGB,  M, (width, height))
        mask   = cv2.warpPerspective(self.warpedResultMask,  M, (width, height))

        # Combine foreground + background
        forground = cv2.bitwise_or(result, result, mask = mask)

        mask = cv2.bitwise_not(mask)
        background = cv2.bitwise_or(self.originalImage, self.originalImage, mask=mask)

        # Show Solution in original Image
        self.resultImage = cv2.bitwise_or(forground, background)
        cv2.imshow("mySudoku", self.resultImage)
        cv2.waitKey()