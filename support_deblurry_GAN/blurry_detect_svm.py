import numpy as np

from scipy.ndimage import variance
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import laplace
from skimage.transform import resize
from sklearn import preprocessing, svm
from sklearn.externals import joblib
from sklearn import metrics

import os
import sys
import matplotlib.pyplot as plt

model = joblib.load('laplaces_model_v1.pkl')

input_image = '/home/duong/Documents/researching/GAN/common/image_enhance/image_cmt/test_blurry/blurry_3/40.JPEG'
img = io.imread(input_image)
#img = resize(img,(100,100))
img = rgb2gray(img)

edge_laplace = laplace(img, ksize=3)

variances = variance(edge_laplace)
maximum = np.amax(edge_laplace)

print('variances: ',variances)
print('maximum: ',maximum)

result = model.predict(np.array([(variances,maximum)]))
print('class: ',result)