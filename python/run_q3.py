import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

num_samples = train_x.shape[0]

max_iters = 100
# pick a batch size, learning rate
batch_size = None
learning_rate = None
batch_size = 20
hidden_size = 64

# BEST CASE
learning_rate = 5e-2

# CASE 2:
# learning_rate = 5e-3

# CASE 3:
# learning_rate = 5e-1


batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

batches_valid = get_random_batches(valid_x,valid_y,batch_size)
batch_num_valid = len(batches_valid)

params = {}

# initialize layers here
initialize_weights(train_x.shape[1], hidden_size, params, 'layer1')
initialize_weights(hidden_size, train_y.shape[1], params, 'output')

W_initial = params['Wlayer1'].copy()

train_acc_list = []
train_loss_list = []
valid_acc_list = []

# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    avg_acc = 0
    for xb,yb in batches:

        # training loop can be exactly the same as q2!
        # forward
        h1 = forward(xb, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)

        # loss
        # be sure to add loss and accuracy to epoch totals
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        total_acc += acc

        # backward
        delta1 = probs.copy()
        delta1 = delta1 - yb
        delta2 = backwards(delta1, params, 'output', linear_deriv)
        backwards(delta2, params, 'layer1', sigmoid_deriv)

        # apply gradient
        params['Wlayer1'] += -learning_rate * params['grad_Wlayer1']
        params['Woutput'] += -learning_rate * params['grad_Woutput']
        params['blayer1'] += -learning_rate * params['grad_blayer1']
        params['boutput'] += -learning_rate * params['grad_boutput']

    avg_acc = total_acc / batch_num
    avg_loss = total_loss / num_samples

    train_loss_list.append(avg_loss)
    train_acc_list.append(avg_acc)

    if itr % 2 == 0:
        print("itr: {:02d} \t train loss: {:.2f} \t train acc : {:.2f}".format(itr,total_loss,avg_acc))

# run on validation set and report accuracy! should be above 75%
    valid_acc = None
    total_loss_valid = 0
    total_acc_valid = 0
    for xb, yb in batches_valid:
        # training loop can be exactly the same as q2!
        # forward
        h1 = forward(xb, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)

        # loss
        # be sure to add loss and accuracy to epoch totals
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss_valid += loss
        total_acc_valid += acc

    valid_acc = total_acc_valid / batch_num_valid
    valid_acc_list.append(valid_acc)

    if itr % 2 == 0:
        print("itr: {:02d} \t valid loss: {:.2f} \t valid acc : {:.2f}".format(itr,total_loss_valid,valid_acc))

print('Validation accuracy: ',valid_acc)


if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)


figure = plt.figure()
plt.plot(np.arange(max_iters)+1, train_acc_list)
plt.plot(np.arange(max_iters)+1, valid_acc_list)
plt.title('Training accuracy vs Validation accuracy over the epochs')
plt.legend(['Training accuracy','Validation accuracy'])
plt.show()

figure = plt.figure()
plt.plot(np.arange(max_iters)+1, train_loss_list)
plt.title('Training loss over the epochs')
plt.show()


# Q3.1.2
import matplotlib.pyplot as plt
import pickle

params = pickle.load(open('q3_weights.pickle', 'rb'))

batches_test = get_random_batches(test_x,test_y,batch_size)
batch_num_test = len(batches_test)

actual_label_list = np.zeros( (1,1) )
predicted_label_list = np.zeros( (1,1) )

test_acc = 0
total_acc_test = 0
for xb, yb in batches_test:
        # training loop can be exactly the same as q2!
        # forward
        h1 = forward(xb, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)

        predicted_label_list = np.vstack( ( predicted_label_list, np.expand_dims(np.argmax(probs, axis=1), 1) ) )
        actual_label_list = np.vstack( (actual_label_list, np.expand_dims(np.argmax(yb, axis=1), 1) ) )

        # loss
        # be sure to add loss and accuracy to epoch totals
        loss, acc = compute_loss_and_acc(yb, probs)
        total_acc_test += acc

        test_acc = total_acc_test / batch_num_test

print('test_acc: ', test_acc)

predicted_label_list = predicted_label_list[1:, :].astype(int)
actual_label_list = actual_label_list[1:, :].astype(int)

# if True: # view the data
#     for crop in xb:
#         import matplotlib.pyplot as plt
#         plt.imshow(crop.reshape(32,32))
#         plt.show()


# Q3.1.3
from mpl_toolkits.axes_grid1 import ImageGrid

fig = plt.figure()
grid = ImageGrid(fig, 111, nrows_ncols=(8, 8), axes_pad=0.1 )
for i in range(64):
    grid[i].imshow( W_initial[:,i].reshape(32,32))      # The AxesGrid object work as a list of axes.

W_final = params['Wlayer1']
fig = plt.figure()
grid = ImageGrid(fig, 111, nrows_ncols=(8, 8), axes_pad=0.1 )
for i in range(64):
    grid[i].imshow( W_final[:,i].reshape(32,32))      # The AxesGrid object work as a list of axes.

plt.show()


# Q3.1.4

import matplotlib.pyplot as plt

confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

y_pred = predicted_label_list
y = actual_label_list

for i in range(y_pred.shape[0]):
    confusion_matrix[ y[i,0],  y_pred[i,0] ] += 1

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()