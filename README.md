# OCR-Extracting-Text-from-Images
OCR using a simple network developed from scratch on NIST36 dataset vs with CNN on PyTorch on EMNIST dataset

**1)From scratch with NIST36 dataset**   

Training Models

Since our input images are 32 × 32 images, unrolled into one 1024 dimensional vector, that gets multiplied by W(1), each row of W(1) can be seen as a weight image. Reshaping each row into a 32×32 image can give us an idea of what types of images each unit in the hidden layer has a high response to.

The training data in nist36 train.mat contains samples for each of the 26 upper-case letters of the alphabet and the 10 digits. This is the set you should use for training your network. The crossvalidation set in nist36 valid.mat contains samples from each class, and should be used in the training loop to see how the network is performing on data that it is not training
on. This will help to spot over fitting. Finally, the test data in nist36 test.mat contains testing data, and should be used for the final evaluation on your best model to see how well  it will generalize to new unseen data.

We train a network from scratch using a single hidden layer with 64 hidden units, and train for at least 30 epochs. We modify the script to plot generate two plots:
one showing the accuracy on both the training and validation set over the epochs, and the other showing the cross-entropy loss averaged over the data. The x-axis should represent the epoch number, while the y-axis represents the accuracy or loss. We see an accuracy on the validation set of 75%.

Visualizing the confusion matrix for your best model.     

![5](/results/5.png)

We can observe that the top misclassified classes are:     
O confused with 0, D  
8 confused with B  
I confused with 1     



Now that we have a network that can recognize handwritten letters with reasonable accuracy, we can now use it to parse text in an image. Given an image with some text on it, our goal is to have a function that returns the actual text in the image. 
However, since your neural network expects a a binary image with a single character, we will need to process the input image to extract each character. 

Steps:

1. Process the image (blur, threshold, opening morphology, etc. (perhaps in that order))
to classify all pixels as being part of a character or background.  
2. Find connected groups of character pixels (see skimage.measure.label). Place a bounding box around each connected component.   
3. Group the letters based on which line of the text they are a part of, and sort each group so that the letters are in the order they appear on the page.   
4. Take each bounding box one at a time and resize it to 32 × 32, classify it with LeNet5 network, and show the characters in order (inserting spaces when it makes sense).    

![1](/results/1.png)
![2](/results/2.png)
![3](/results/3.png)
![4](/results/4.png)


**1)From EMNIST dataset with PyTorch using a convolutional net (LeNet5) **   

![5](/results/6.png)

We can see that this is much better detection as compared to previous implementation.
Just some minor confusions are there for 0 and O, S and 5, k and I   
Addition of constitutional layers helps in extracting features. This architecture learns much more data.

The graphs for training are shown below:
![6](/results/7.png)

