'''Creating the SVM file, from the IPython notebook
I am going to be copy-pasting a lot of the code'''


import random
import numpy as np
from data_utils import load_CIFAR10 #change path here.      ---------------------------------------------- changed
import matplotlib.pyplot as plt

'''Not sure what following 3 lines do'''

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

'''Need to change file paths everywhere, since I have changed the location'''

cifar10_dir = 'datasets/cifar-10-batches-py'   #change file path here. ------------------------------------------ changed
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)      # change file paths inside data_utils also ------------------------------------- nothing to change.

# As a sanity check, we print out the size of the training and test data.
print( 'Training data shape: {0}'.format( X_train.shape))
print( 'Training labels shape:{0} '.format(y_train.shape))
print('Test data shape: {0}'.format( X_test.shape))
print( 'Test labels shape: {0}'.format( y_test.shape) )


# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()


''' This is where different X and y are created !  '''

# Split the data into train, val, and test sets. In addition we will
# create a small development set as a subset of the training data;
# we can use this for development so our code runs faster.
num_training = 19000                  ## ===================================== An important deviation from the assignment to keep in mind.
num_validation = 1000
num_test = 1000
num_dev = 500

print("checkpoint 1")


# Our validation set will be num_validation points from the original
# training set.
mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask]

print("checkpoint2")

# Our training set will be the first num_train points from the original
# training set.
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

print("checkpoint3")

# We will also make a development set, which is a small subset of
# the training set.
mask = np.random.choice(num_training, num_dev, replace=False)
X_dev = X_train[mask]
y_dev = y_train[mask]


print("checkpoint4")

# We use the first num_test points of the original test set as our
# test set.
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

print( 'Train data shape:{0} '.format( X_train.shape) )
print( 'Train labels shape: {0}'.format(y_train.shape) )
print( 'Validation data shape: {0}'.format(X_val.shape) )
print( 'Validation labels shape: {0}'.format(y_val.shape) )
print( 'Test data shape: {0}'.format(X_test.shape) )
print( 'Test labels shape: {0}'.format(y_test.shape) )


'''  This is where X and y are given the shapes that we'll be using while writing code   '''
# Preprocessing: reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

# As a sanity check, print out the shapes of the data
print( 'Training data shape: {0}'.format( X_train.shape ) )
print( 'Validation data shape: {0}'.format(X_val.shape) )
print( 'Test data shape: {0}'.format( X_test.shape ) )
print( 'dev data shape: {0}'.format( X_dev.shape ) )





''' Next , we calculate mean image , and visualise it'''

# Preprocessing: subtract the mean image
# first: compute the image mean based on the training data
mean_image = np.mean(X_train, axis=0)
print(mean_image[:10]) # print a few of the elements
plt.figure(figsize=(4,4))
plt.imshow(mean_image.reshape((32,32,3)).astype('uint8')) # visualize the mean image
plt.show()



''' Then, we are going to subtract mean image, from all images , but we won't be calculating standard deviation '''

# second: subtract the mean image from train and test data .... Why did we do this anyway?
X_train -= mean_image
X_val -= mean_image
X_test -= mean_image
X_dev -= mean_image

'''  Here, we'll add the bias terms  '''

# third: append the bias dimension of ones (i.e. bias trick) so that our SVM
# only has to worry about optimizing a single weight matrix W.
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

print(X_train.shape, X_val.shape, X_test.shape, X_dev.shape)


'''This is where the actual algorithm implementation begins'''



import time

'''  I am going to skip the naie svm loss and gradient calculation, and directly use vectorized coding.  '''
from linear_svm import svm_loss_vectorized   #Change path here !  ------------------------------------------- changed

W = np.random.randn(3073, 10) * 0.0001 
tic = time.time()
loss_vectorized, _ = svm_loss_vectorized(W, X_dev, y_dev, 0.00001)
toc = time.time()
print( 'Vectorized loss: %e computed in %fs' %(loss_vectorized, toc - tic)   )

# The losses should match but your vectorized implementation should be much faster.
#print 'difference: %f' % (loss_naive - loss_vectorized)



''' Now, I am going to write code for training the model. It's a simple one.'''

from linear_classifier import LinearSVM           #Change path  ----------------------------------------- changed
svm = LinearSVM()
tic = time.time()
loss_hist = svm.train(X_train, y_train, learning_rate=1e-7, reg=5e4,
                      num_iters=1500, verbose=True)
toc = time.time()
print( 'That took %fs' % (toc - tic) )

# A useful debugging strategy is to plot the loss as a function of
# iteration number:
plt.plot(loss_hist)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()


'''  Then, we make predictions on our training and validation X, and print the accuracies in each case  '''

# Write the LinearSVM.predict function and evaluate the performance on both the
# training and validation set
y_train_pred = svm.predict(X_train)
print( 'training accuracy: %f' % (np.mean(y_train == y_train_pred), ) )
y_val_pred = svm.predict(X_val)
print( 'validation accuracy: %f' % (np.mean(y_val == y_val_pred), ))




''' Now, the following code is to figure out the best combo of hyperparameters. '''


learning_rates = [   float( "1e-{0}".format(a) ) for a in range(6)   ]   #This wasn't in original code .. Hell Yeah !   ************

regularization_strengths =[(j+0.1*i)*1e4 for j in range(1,5) for i in range(0,10)]  # A total of 50 regs.



# results is dictionary mapping tuples of the form
# (learning_rate, regularization_strength) to tuples of the form
# (training_accuracy, validation_accuracy). The accuracy is simply the fraction
# of data points that are correctly classified.
results = {}
best_val = -1   # The highest validation accuracy that we have seen so far.
best_svm = None # The LinearSVM object that achieved the highest validation rate.



for reg in regularization_rates:
    for lr in learning_rates:

        svm = LinearSVM()
        loss_hist = svm.train( X_train, y_train, lr, reg, num_iters=1500   )

        y_train_pred = svm.predict(X_train)
        train_accuracy = np.mean(y_train == y_train_pred)

        y_val_pred = svm.predict(X_val)
        val_accuracy = np.mean(y_val == y_val_pred)

        if val_accuracy > best_val:
            best_val = val_accuracy
            best_svm = svm           

        results[(lr,reg)] = train_accuracy, val_accuracy

        


# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (lr, reg, train_accuracy, val_accuracy))
    
print( 'best validation accuracy achieved during cross-validation: %f' % best_val )















