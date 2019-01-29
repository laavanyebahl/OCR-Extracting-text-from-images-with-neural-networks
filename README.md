# OCR-Extracting-Text-from-Images
OCR using a network developed from scratch on NIST36 vs EMNIST on PyTorch

**1)From scratch with NIST36 dataset**   

Training Models

Since our input images are 32 × 32 images, unrolled into one 1024 dimensional vector, that gets multiplied by W(1), each row of W(1) can be seen as a weight image. Reshaping each row into a 32×32 image can give us an idea of what types of images each unit in the hidden layer has a high response to.

The training data in nist36 train.mat contains samples for each of the 26 upper-case letters of the alphabet and the 10 digits. This is the set you should use for training your network. The crossvalidation set in nist36 valid.mat contains samples from each class, and should be used in the training loop to see how the network is performing on data that it is not training
on. This will help to spot over fitting. Finally, the test data in nist36 test.mat contains testing data, and should be used for the final evaluation on your best model to see how well  it will generalize to new unseen data.

We train a network from scratch using a single hidden layer with 64 hidden units, and train for at least 30 epochs. We modify the script to plot generate two plots:
one showing the accuracy on both the training and validation set over the epochs, and the other showing the cross-entropy loss averaged over the data. The x-axis should represent the epoch number, while the y-axis represents the accuracy or loss. We see an accuracy on the validation set of 75%.

Visualizing the confusion matrix for your best model.   

We can observe that the top misclassified classes are:   
O confused with 0, D
8 confused with B
I confused with 1

**1)From EMNIST dataset with PyTorch**   

