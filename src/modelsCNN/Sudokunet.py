# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


class SudokuNet:

    @staticmethod
    def build(width, height, nb_channel, nb_classes):
        """build the SudokuNet CNN architecture implemented with TensorFlow and Keras in order to recognize the digits belonging to the Sudoku grid.
        
        Parameters:
        - height    : Input image height
        - width     : Input images width
        - nb_channel: Input images depth ( 1 for gray images, and 3 for RGB images )
        - nb_classes: Number of output classes

        Returns Classifier architecture."""

        assert (nb_channel == 1 or nb_channel == 3), "depth of images must be 1 for gray images, or 3 for RGB images"
        inputShape = (height, width, nb_channel)

        # initialize the model
        classifier = Sequential()

        # first set of CONV => RELU => POOL layers
        classifier.add(Conv2D(32, (5, 5), padding="same", activation = "relu", input_shape=inputShape))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))

        # second set of CONV => RELU => POOL layers
        classifier.add(Conv2D(32, (3, 3), padding="same", activation = "relu"))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))

        # first set of FC => RELU layers
        classifier.add(Flatten())
        classifier.add(Dense(64, activation = "relu"))
        classifier.add(Dropout(0.5))

        # second set of FC => RELU layers
        classifier.add(Dense(64, activation = "relu"))
        classifier.add(Dropout(0.5))

        # softmax classifier 
        classifier.add(Dense(nb_classes, activation = "softmax"))
        
        classifier.summary()

        # return the constructed network architecture
        return classifier
