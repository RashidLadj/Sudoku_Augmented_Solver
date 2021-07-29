import argparse
import cv2
import imutils
from Puzzle.puzzleDetection import sudoku

# Construct the argument parser and parse the arguments
args = argparse.ArgumentParser()

args.add_argument("-m", "--model", required=True, help="path to trained digit classifier")
args.add_argument("-i", "--image", required=True, help="path to input Sudoku puzzle image")

args = vars(args.parse_args())



# Load the input image from disk and resize it
print("[INFO] processing image...")
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)

mySudoku = sudoku(image)
# Find the puzzle in the image and then
mySudoku.find_puzzle()
cv2.imshow("MySudokuPuzzle", mySudoku.warpedPuzzleRGB)
cv2.waitKey()

# Find the puzzle in the image and then
mySudoku.create_SudokuMatrix(args["model"])
mySudoku.solveAndShow_Sudoku()

