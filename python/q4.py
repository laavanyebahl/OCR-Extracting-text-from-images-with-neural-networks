import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    ## REMOVING NOISE TAKES TOO MUCH TIME
    # sigma_est = skimage.restoration.estimate_sigma(image, multichannel=True, average_sigmas=True)
    # print('noise before: ', sigma_est)

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    gray_img = skimage.color.rgb2gray(image)

    # apply threshold
    thresh = skimage.filters.threshold_otsu(gray_img)

    binary_img = gray_img > thresh
    bw = binary_img.astype(float)

    bw = skimage.morphology.opening(bw, skimage.morphology.square(7))

    # label image regions
    label_image = skimage.measure.label(bw, neighbors=8, background=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for region in skimage.measure.regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 100:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='blue', linewidth=0.5)
            ax.add_patch(rect)
            bboxes.append(region.bbox)

    resolution  = bw.shape[0]*bw.shape[1]
    print(bw.shape[0]*bw.shape[1])

    if (resolution>3500000):
        bw = apply_erosion(bw,2)

    if (resolution>4000000):
        bw = apply_erosion(bw,1)

    if (resolution>10000000):
        bw = apply_erosion(bw,7)

    return bboxes, bw


def apply_erosion(bw, n):
    for i in range(n):
        bw = skimage.morphology.erosion(bw)
    return bw