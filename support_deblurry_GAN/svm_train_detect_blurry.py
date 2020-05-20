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

# Load image
not_blurry_folder = f'/home/duong/Documents/researching/GAN/common/image_enhance/image_cmt/test_good'
blurry_folder = '/home/duong/Documents/researching/GAN/common/image_enhance/image_cmt/test_blurry'

def laplace_image(input_folder):
    sub_folders = os.listdir(input_folder)
    variances = []
    maximumes = []

    for folder in sub_folders:
        sub_folder = os.path.join(input_folder,folder)
        if not os.path.isdir(sub_folder):
            continue
        list_file = os.listdir(sub_folder)

        for file in list_file:
            if file.endswith(('.png','.jpg','JPEG')):
                input_file = os.path.join(sub_folder,file)

                #preprocessing
                img = io.imread(input_file)
                img = resize(img,(400,600))
                img = rgb2gray(img)

                #Edge Detection use Laplace
                edge_laplace = laplace(img, ksize=3)

                #print(f"Variance: {variance(edge_laplace)}")
                variances.append(variance(edge_laplace))

                #print(f'Maximum: {np.amax(edge_laplace)}')
                maximumes.append(np.amax(edge_laplace))
    return variances, maximumes

variances, maximumes = laplace_image(not_blurry_folder)
variances1, maximumes1 = laplace_image(blurry_folder)

print('length of sharp:',len(variances))
print('length of blurry:',len(variances1))

load_model = False
use_svm = True

sharp_laplaces = list(zip(variances,maximumes))
blurry_laplaces = list(zip(variances1, maximumes1))

y = np.concatenate((np.ones((51,)), np.zeros((51,))), axis=0)
print("y = ",y)
laplaces = np.concatenate((np.array(sharp_laplaces), np.array(blurry_laplaces)), axis=0)

#laplaces = preprocessing.scale(laplaces)


clf = svm.SVC(kernel='linear')
clf.fit(laplaces, y)

# print("Accuracy :",metrics.accuracy_score(laplaces[:50],y[:50]))

print(f'Weights: {clf.coef_[0]}')
print(f'Intercept: {clf.intercept_}')

r1 = clf.predict([[0.00040431, 0.1602369]])  # result: 0 (blurred)
r2 = clf.predict([[0.01, 0.6]])  # result: 1 (sharp)
print('r1 = ',r1)
print('r2 = ',r2)

#save model
joblib.dump(clf, 'laplaces_model.pkl')

clf = joblib.load('laplaces_model.pkl')
resutl = clf.predict(np.array([blurry_laplaces[33]]))
print(resutl)

resutl = clf.predict(np.array([sharp_laplaces[30]]))
print(resutl)

x0 = [0,0.02]
y0 = [0.5,0]

plt.plot(variances, maximumes,'ro')
plt.plot(variances1, maximumes1,'b.')
#plt.plot(x0,y0)
plt.plot()
#plt.axis([0.0, 0.1, 0.0, 3.0])
plt.xlabel('Variance (Laplace)')
plt.ylabel('Maximum (Laplace)')
plt.grid(True)
plt.show()

