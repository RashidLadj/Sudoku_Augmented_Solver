# Sudoku_Augmented_Solver

Sudoku is a popular number puzzle that requires you to fill blanks in a 9X9 grid with digits so that each column, each row, and each of the nine 3 × 3 subgrids contains all of the digits from 1 to 9. There have been various approaches to solving that, including computational ones. in our case, for a first version, we are not interested in solving Sudoku by ourselves, we will use pySudoku to solve the puzzle.
Our objective is indeed the detection of the puzzle on the image (Video) as well as each digit (box with value) by using a CNN classifier leading it on the MNIST dataset of Keras and the display of the solution on the 'image (video).


# File description
- Sudokunet.py : File containing the CNN architecture of the model used to predict the value of each digit.
- train_digit_classifier.py : File containing all the steps carried out to train our CNN SudokuNet model with the MNIST dataset.

# Usage
``` python src\modelsCNN\train_digit_classifier.py -m modelOutputPath.h5 ```  
``` python src\solve_sudoku_puzzle.py -i imagePth.png -m modelPath.h5```  