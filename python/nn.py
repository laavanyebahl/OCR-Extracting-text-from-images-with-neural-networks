import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

# Q 2.1
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):

    low = - np.sqrt( 6/( in_size + out_size ) )
    high =  np.sqrt( 6/( in_size + out_size ) )

    W = np.random.uniform(low, high, size=( in_size, out_size) )
    b = np.zeros(out_size)
    
    params['W' + name] = W
    params['b' + name] = b

# Q 2.2.1
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = 1 / (1 + np.exp(-x))
    return res

# Q 2.2.2
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    # your code here
    pre_act = X.dot(W) + b
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

# Q 2.2.2 
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = []
    for row in range(x.shape[0]):
        exps = np.exp(x[row, :] - np.max(x[row, :]))
        res.append(exps / np.sum(exps))
    res = np.asarray(res)
    return res

# Q 2.2.3
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None

    loss = 0
    for i in range(probs.shape[0]):
        loss += -np.dot(y[i,:], np.log(probs[i,:]))

    y = np.argmax(y, axis=1)
    pred = np.argmax(probs, axis=1)
    acc = np.mean(pred == y)
    return loss, acc

# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # your code here
    # do the derivative through activation first
    # then compute the derivative W,b, and X

    batch_size =  X.shape[0]

    ## TAKE SUM OF ALL THE TRAINING EXAMPLES AS ALL THE EXAMPLES ARE CONSIDERED IN COMPUTING THE LOSS
    grad_X  = (delta* activation_deriv(post_act)).dot(W.T)
    grad_W  = X.T.dot(delta* activation_deriv(post_act)) / batch_size
    grad_b  = (delta* activation_deriv(post_act)).sum(axis=0) /batch_size

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

# Q 2.4
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []

    random_idexes = np.arange(x.shape[0])
    np.random.shuffle(random_idexes)
    x = x[random_idexes, :]
    y = y[random_idexes, :]

    num_batches = x.shape[0]//batch_size
    for i in range(num_batches):
        batches.append( ( x[i*batch_size: (i+1)*batch_size,:], y[i*batch_size: (i+1)*batch_size,:] ) )

    return batches
