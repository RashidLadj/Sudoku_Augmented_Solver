import argparse
from Puzzle.puzzleDetection import sudoku

# Construct the argument parser and parse the arguments
args = argparse.ArgumentParser()

args.add_argument("-m", "--model", required=True, help="path to trained digit classifier")
args.add_argument("-i", "--image", required=True, help="path to input Sudoku puzzle image")

args = vars(args.parse_args())



# Load the input image from disk and resize it
print("[INFO] processing image...")
mySudoku = sudoku(args["image"])
mySudoku.find_puzzle()
mySudoku.create_SudokuMatrix(args["model"])
mySudoku.solveAndShow_Sudoku()
mySudoku.generateWarpedSolution()
mySudoku.generateSolution()