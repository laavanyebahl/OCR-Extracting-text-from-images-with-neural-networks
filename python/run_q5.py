import numpy as np
import scipy.io
from nn import *
from collections import Counter

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

num_samples = train_x.shape[0]

max_iters = 100
# pick a batch size, learning rate
batch_size = 25
learning_rate =  4e-3
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# initialize layers here
initialize_weights(train_x.shape[1], hidden_size, params, 'layer1')
initialize_weights(hidden_size, hidden_size, params, 'hidden')
initialize_weights(hidden_size, hidden_size, params, 'hidden2')
initialize_weights(hidden_size, train_x.shape[1], params, 'output')

params['m_Wlayer1'] = np.zeros((train_x.shape[1], hidden_size))
params['m_Whidden']= np.zeros((hidden_size,hidden_size))
params['m_Whidden2']= np.zeros((hidden_size, hidden_size))
params['m_Woutput']= np.zeros((hidden_size, train_x.shape[1]))

train_loss_list = []

# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        # forward
        h1 = forward(xb, params, 'layer1', relu)
        h2 = forward(h1, params, 'hidden', relu)
        h3 = forward(h2, params, 'hidden2', relu)
        out = forward(h3, params, 'output', sigmoid)

        # loss
        loss = np.sum((out- xb)**2)
        total_loss += loss

        # backward
        delta1 = 2*(out - xb)
        delta2 = backwards(delta1, params, 'output', sigmoid_deriv)
        delta3 = backwards(delta2, params, 'hidden2', relu_deriv)
        delta4 = backwards(delta3, params, 'hidden', relu_deriv)
        backwards(delta4, params, 'layer1', relu_deriv)

        # apply momentum gradient
        params['m_Wlayer1'] = 0.9*params['m_Wlayer1'] -learning_rate*params['grad_Wlayer1']
        params['m_Whidden'] = 0.9*params['m_Whidden'] -learning_rate*params['grad_Whidden']
        params['m_Whidden2'] = 0.9*params['m_Whidden2'] -learning_rate*params['grad_Whidden2']
        params['m_Woutput'] = 0.9*params['m_Woutput'] -learning_rate*params['grad_Woutput']

        params['Wlayer1'] += params['m_Wlayer1']
        params['Whidden'] += params['m_Whidden']
        params['Whidden2'] += params['m_Whidden2']
        params['Woutput'] += params['m_Woutput']

        params['blayer1'] +=  -learning_rate*params['grad_blayer1']
        params['bhidden'] += -learning_rate*params['grad_bhidden']
        params['bhidden2'] += -learning_rate*params['grad_bhidden2']
        params['boutput'] += -learning_rate*params['grad_boutput']

    avg_loss = total_loss / num_samples
    train_loss_list.append(avg_loss)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9


# visualize some results
# Q5.3.1
import matplotlib.pyplot as plt
h1 = forward(valid_x,params,'layer1',relu)
h2 = forward(h1,params,'hidden',relu)
h3 = forward(h2,params,'hidden2',relu)
out = forward(h3,params,'output',sigmoid)

step = int(valid_x.shape[0]/36)
i=1
index = step*1
plt.subplot(5,4,1)
plt.imshow(valid_x[index].reshape(32,32).T)
plt.subplot(5,4,2)
plt.imshow(out[index].reshape(32, 32).T)
plt.subplot(5,4,3)
plt.imshow(valid_x[index+1].reshape(32,32).T)
plt.subplot(5,4,4)
plt.imshow(out[index+1].reshape(32, 32).T)
i+=4
index = step*(i)-50
plt.subplot(5,4,5)
plt.imshow(valid_x[index].reshape(32,32).T)
plt.subplot(5,4,6)
plt.imshow(out[index].reshape(32, 32).T)
plt.subplot(5,4,7)
plt.imshow(valid_x[index+1].reshape(32,32).T)
plt.subplot(5,4,8)
plt.imshow(out[index+1].reshape(32, 32).T)
i+=3
index = step*(i)-50
plt.subplot(5,4,9)
plt.imshow(valid_x[index].reshape(32,32).T)
plt.subplot(5,4,10)
plt.imshow(out[index].reshape(32, 32).T)
plt.subplot(5,4,11)
plt.imshow(valid_x[index+1].reshape(32,32).T)
plt.subplot(5,4,12)
plt.imshow(out[index+1].reshape(32, 32).T)
i+=2
index = step*(i)-50
plt.subplot(5,4,13)
plt.imshow(valid_x[index].reshape(32,32).T)
plt.subplot(5,4,14)
plt.imshow(out[index].reshape(32, 32).T)
plt.subplot(5,4,15)
plt.imshow(valid_x[index+1].reshape(32,32).T)
plt.subplot(5,4,16)
plt.imshow(out[index+1].reshape(32, 32).T)
i+=3
index = step*(i)-50
plt.subplot(5,4,17)
plt.imshow(valid_x[index].reshape(32,32).T)
plt.subplot(5,4,18)
plt.imshow(out[index].reshape(32, 32).T)
plt.subplot(5,4,19)
plt.imshow(valid_x[index+1].reshape(32,32).T)
plt.subplot(5,4,20)
plt.imshow(out[index+1].reshape(32, 32).T)

plt.show()

figure = plt.figure()
plt.plot(np.arange(max_iters)+1, train_loss_list)
plt.title('Training loss over the epochs')
plt.show()

from skimage.measure import compare_psnr as psnr
# evaluate PSNR
# Q5.3.2

h1 = forward(valid_x,params,'layer1',relu)
h2 = forward(h1,params,'hidden',relu)
h3 = forward(h2,params,'hidden2',relu)
out_valid = forward(h3,params,'output',sigmoid)

total = []
for pred,gt in zip(out_valid, valid_x):
    total.append(psnr(gt,pred))
print(np.array(total).mean())