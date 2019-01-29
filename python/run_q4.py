import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw, cmap='gray')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.

    ## SORT ACCORDING TO CENTROID HEIGHT
    bboxes.sort(key = lambda x: (x[0] + x[2])//2 )

    total_list = []
    sublist = []
    sublist.append(list(bboxes[0]))

    for i in range(1, len(bboxes)):
        box_prev = bboxes[i-1]
        box_curr = bboxes[i]
        centroid_prev = ( (box_prev[0] + box_prev[2])//2, (box_prev[1] + box_prev[3])//2 )
        centroid_curr = ( (box_curr[0] + box_curr[2])//2, (box_curr[1] + box_curr[3])//2 )
        height_prev = box_prev[2] - box_prev[0]
        height_curr = box_curr[2] - box_curr[0]

        ## TO AVOID VERY SMALL BOUNDING BOXES FOR BROKEN LINES, ETC.
        if height_curr> (height_prev*0.4):
            ## CHECK IF DIFFERENCE IN HEIGHT OF CURRENT AND PREV IS MORE THAN HALF OF BOUNDING BOX
            ## IF YES, NEW LINE
            if(centroid_curr[0]-centroid_prev[0] > (height_prev/2)):
                total_list.append(sublist)
                sublist = []
                sublist.append(list(box_curr))
            else:
                sublist.append(list(box_curr))

    total_list.append(sublist)


    ## SORT ACCORDING TO CENTROID WIDTH
    for row in total_list:
        row.sort(key = lambda x: (x[1] + x[3])//2 )


    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset

    print('For image: ', img)
    for row in range(len(total_list)):
        print(' ')
        # print('row: ', row, ', num_col: ', len(total_list[row]))
        for col in range(len(total_list[row])):
            box_curr = total_list[row][col]
            crop_image = bw[ box_curr[0]: box_curr[2], box_curr[1]: box_curr[3] ]
            # crop_image = skimage.morphology.erosion(crop_image)
            max_dim = max(crop_image.shape[0], crop_image.shape[1])
            pad_margin = int(max_dim*0.15)
            ones = np.ones((max_dim+pad_margin, max_dim+pad_margin))
            width_diff = ones.shape[1] - crop_image.shape[1]
            height_diff = ones.shape[0] - crop_image.shape[0]
            ones[ height_diff//2: height_diff//2 + crop_image.shape[0] ,  width_diff//2: width_diff//2 + crop_image.shape[1] ] = crop_image
            final_crop = ones.T
            final_crop = skimage.transform.resize(final_crop, (32,32))
            # plt.imshow(final_crop)
            # plt.show()

            final_crop_flat = final_crop.reshape(1, -1)
            h1 = forward(final_crop_flat, params, 'layer1')
            probs = forward(h1, params, 'output', softmax)
            predicted_label_index = np.argmax(probs, axis=1)[0]
            print( letters[ predicted_label_index ]  , end ="")

    print('\n')


