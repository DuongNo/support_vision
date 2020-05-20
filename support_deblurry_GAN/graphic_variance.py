import numpy as np
from scipy.ndimage import variance, sobel
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import laplace
from skimage.transform import resize


import os
import sys
import matplotlib.pyplot as plt

def laplace_folder(input_folder, sub_folder_b=False):
    sub_folders = []
    if sub_folder_b:
        sub_folders = os.listdir(input_folder)
    else:
        sub_folders.append(input_folder)
    variances = []
    maximumes = []

    variance_sobel = []
    maximumes_sobel = []

    for sub_folder in sub_folders:
        if sub_folder_b:
            folder = os.path.join(input_folder,sub_folder)
        else:
            folder = sub_folder
        if not os.path.isdir(folder):
            continue
        list_file = os.listdir(folder)

        for file in list_file:
            if file.endswith(('.png','.jpg','JPEG')):
                input_file = os.path.join(folder,file)

                #preprocessing
                image = io.imread(input_file)
                img = resize(image,(112,112))
                img = rgb2gray(img)

                #edge detection use laplace
                edge_laplace = laplace(img, ksize=3)

                #edge detection with Sobel filter
                edge_soble = sobel(img)

                variances.append(variance(edge_laplace))
                maximumes.append(np.amax(edge_laplace))

                # test_blurry = '/home/duong/Documents/researching/GAN/common/image_enhance/image_cmt/test_blurry'
                # blurry_folder = os.path.join(test_blurry,file)

                # test_good = '/home/duong/Documents/researching/GAN/common/image_enhance/image_cmt/test_good'
                # good_folder = os.path.join(test_good,file)

                # if variance(edge_laplace) < 0.0015 and np.amax(edge_laplace) < 0.3:
                #     io.imsave(blurry_folder,image)
                # else:
                #     io.imsave(good_folder,image)


                #variance_sobel.append(variance(edge_soble))
                #maximumes_sobel.append(np.amax(edge_soble))
    #return variances, maximumes, variance_sobel, maximumes_sobel
    return variances, maximumes

#input_folder = '/home/duong/Documents/researching/GAN/common/image_enhance/image_cmt/test_blurry'
input_folder = '/home/duong/Documents/researching/GAN/data_SRGan/data_blurry_image'
#variances, maximumes, variance_sobel, maximumes_sobel = laplace_folder(input_folder)
variances, maximumes = laplace_folder(input_folder,sub_folder_b=True)

line1 = plt.plot(variances, maximumes,'ro')
#line2 = plt.plot(variance_sobel, maximumes_sobel,'b.')
plt.plot()
plt.xlabel('Variance (Laplace)')
plt.ylabel('Maximum (Laplace)')
plt.setp(line1,linewidth=4)
#plt.setp(line2,linewidth=2)
#plt.legend(('laplace', 'sobel'),loc='upper_left')
plt.grid(True)
plt.title('Variance of image')
plt.show()

