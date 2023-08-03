# Hello World ML
## Requirements
* `tensorflow`
* `matplotlib`

## Steps
1. Load the MNIST dataset 
    ```python
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    ```
   * The MNIST data set which consists of 60,000+ grayscale images of handwritten digits for training and 10,000 images for testing.
   * `load_data()` returns a tuple that contains 2 NumPy arrays: `(x_train, y_train), (x_test, y_test)`
     * `x_train` and `x_test` contain the input data, which are the images in the MNIST dataset. They are 28x28 pixel grayscale images of handwritten digits (0 to 9).
     * `y_train` and `y_test` contain the corresponding labels for the images in `x_train` and `x_test`. The labels are the actual digit values (0 to 9) representing what each image depicts.
   * **OPTIONAL**: The following snippet can be used to display the images and labels
       ```python
        # Display one of the training images
        index_to_display = 2  # Change this value to display a different image
        plt.imshow(x_train[index_to_display], cmap='Blues_r') # imshow() can be used to show the image and cmap is used to set the colormap
        plt.title('Label: ' + str(y_train[index_to_display]))
        plt.show()
        ```
2. 

This data set which consists of 60,000+ grayscale images of handwritten digits for training
 and 10,000 images for testing.
 load_data() returns a tuple that contains 2 NumPy arrays: (x_train, y_train), (x_test, y_test)