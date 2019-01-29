import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

dim = 32
# do PCA

U, E, Vh = np.linalg.svd(train_x)
proj_Mat = Vh[:32, :]

print('proj_Mat shape: ', proj_Mat.shape)

reconstruction_Mat = proj_Mat.T.dot(proj_Mat)

# rebuild a low-rank version
lrank = np.linalg.matrix_rank(proj_Mat)
print(lrank)

# rebuild it
recon = train_x.dot(reconstruction_Mat)

# build valid dataset
recon_valid = valid_x.dot(reconstruction_Mat)

# for i in range(5):
#     plt.subplot(2,1,1)
#     plt.imshow(valid_x[i].reshape(32,32).T)
#     plt.subplot(2,1,2)
#     plt.imshow(recon_valid[i].reshape(32,32).T)
#     plt.show()

step = int(valid_x.shape[0]/36)
i=1
index = step*1
plt.subplot(5,4,1)
plt.imshow(valid_x[index].reshape(32,32).T)
plt.subplot(5,4,2)
plt.imshow(recon_valid[index].reshape(32, 32).T)
plt.subplot(5,4,3)
plt.imshow(valid_x[index+1].reshape(32,32).T)
plt.subplot(5,4,4)
plt.imshow(recon_valid[index+1].reshape(32, 32).T)
i+=4
index = step*(i)-50
plt.subplot(5,4,5)
plt.imshow(valid_x[index].reshape(32,32).T)
plt.subplot(5,4,6)
plt.imshow(recon_valid[index].reshape(32, 32).T)
plt.subplot(5,4,7)
plt.imshow(valid_x[index+1].reshape(32,32).T)
plt.subplot(5,4,8)
plt.imshow(recon_valid[index+1].reshape(32, 32).T)
i+=3
index = step*(i)-50
plt.subplot(5,4,9)
plt.imshow(valid_x[index].reshape(32,32).T)
plt.subplot(5,4,10)
plt.imshow(recon_valid[index].reshape(32, 32).T)
plt.subplot(5,4,11)
plt.imshow(valid_x[index+1].reshape(32,32).T)
plt.subplot(5,4,12)
plt.imshow(recon_valid[index+1].reshape(32, 32).T)
i+=2
index = step*(i)-50
plt.subplot(5,4,13)
plt.imshow(valid_x[index].reshape(32,32).T)
plt.subplot(5,4,14)
plt.imshow(recon_valid[index].reshape(32, 32).T)
plt.subplot(5,4,15)
plt.imshow(valid_x[index+1].reshape(32,32).T)
plt.subplot(5,4,16)
plt.imshow(recon_valid[index+1].reshape(32, 32).T)
i+=3
index = step*(i)-50
plt.subplot(5,4,17)
plt.imshow(valid_x[index].reshape(32,32).T)
plt.subplot(5,4,18)
plt.imshow(recon_valid[index].reshape(32, 32).T)
plt.subplot(5,4,19)
plt.imshow(valid_x[index+1].reshape(32,32).T)
plt.subplot(5,4,20)
plt.imshow(recon_valid[index+1].reshape(32, 32).T)

plt.show()


total = []
for pred,gt in zip(recon_valid,valid_x):
    total.append(psnr(gt,pred))
print(np.array(total).mean())