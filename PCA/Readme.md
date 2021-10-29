# <font face="Times">Homework #0</font>

###### tags: `Deep Learning for Computer Vision`

## <font face="Times">Problem 1: Principal Component Analysis (100%)</font>

<font face="Times">Principal component analysis (PCA) is a technique of dimensionality reduction, which linearly maps data onto a lower-dimensional space, so that the variance of the projected data in the associated dimensions would be maximized. In this problem, you will perform PCA on a dataset of face images.
The folder p1_data contains face images of 40 different subjects (classes) and 10 grayscale images for each subject, all of size (56, 46) pixels. Note that i_j.png is the j-th image of the i-th person, which is denoted as $Person_iImage_j$ for simplicity.

First, split the dataset into two subsets (i.e., training and testing sets). The first subset contains the first 9 images of each subject, while the second subset contains the remaining images. Thus, a total of $9 \times 40 = 360$ images are in the training set, and $1 \times 40 = 40$ images in the testing set.

In this problem, you will compute the eigenfaces of the training set, and project face images from both the training and testing sets onto the same feature space with reduced dimension.</font>

### <font face="Times">1. (20%) Perform PCA on the training set. Plot the mean face and the first four eigenfaces.</font>

``` python
from PIL import Image
import numpy as np

# Split training set and testing set
def get_image_data(loc):
    return list(Image.open(loc).getdata())
    
path = '/Users/lee/Downloads/p1_data/'

train_set = []
train_label = []
test_set = []
test_label = []

for i in range(1, 41):
    for j in range(1, 10):
        train_set.append(get_image_data(path + '%s_%s.png'%(i, j)))
        train_label.append(i)
    
    test_set.append(get_image_data(path + '%s_10.png'%i))
    test_label.append(i)

# Switch list to array
train_set = np.array(train_set).astype(np.uint8).T #2576x360
train_label = np.array(train_label).T
test_set = np.array(test_set).astype(np.uint8).T    
test_label = np.array(test_label).T

print('Training set size: ', len(train_set))
print('testing set size: ', len(test_set))
```
<img src="https://i.imgur.com/8ZNDe15.png" width="200"/>

``` python
# Mean face
mean_face = np.mean(train_set, axis=1).astype(np.uint8)
print('Shape of mean face: ',mean_face.shape)
plt.imshow(mean_face.reshape(56,46),cmap=plt.cm.bone)
```
<img src="https://i.imgur.com/SqfMK0b.png" width="320"/>


``` python
# Eigenface
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.metrics import mean_squared_error

pca = decomposition.PCA(n_components=360, whiten=True, svd_solver='full')
pca.fit(train_set.T)

fig = plt.figure(figsize=(16, 6))
for i in range(4):
    ax = fig.add_subplot(1, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(pca.components_[i].reshape(56,46),cmap=plt.cm.bone)
    ax.set_title('Eigenface '+ str(i+1))
```

<img src="https://i.imgur.com/YX9eptY.png" width="700"/>



---

### <font face="Times">2. (20%) If the last digit of your student ID number is odd, take person2image1. If the last digit of your student ID number is even, take person8image1. Project it onto the PCA eigenspace you obtained above. Reconstruct this image using the first n = 3, 50, 170, 240, 345 eigenfaces. Plot the five reconstructed images.</font>

### <font face="Times">3. (20%) For each of the five images you obtained in 2., compute the mean squared error (MSE) between the reconstructed image and the original image. Record the corresponding MSE values in your report.</font>

``` python
# person8image1

imagedata = np.array(get_image_data(path + '%s_%s.png'%(8, 1)))
plt.title('Person8_Image1')
plt.imshow(imagedata.reshape(56,46),cmap=plt.cm.bone)
```
<img src="https://i.imgur.com/XOiJacx.png" width="200"/>

``` python
# Eigenface Reconstruction
# n = 3, 50, 170, 240, 345

n = [3, 50, 170, 240, 345]

weights = []
for i in range(len(pca.components_)):
    weights.append(np.dot(pca.components_[i].reshape(1,-1), (imagedata.reshape(-1,1) - pca.mean_))[0])
    
def reconstruction(n, weight):
    reimage = 0
    for i in range(n):
        reimage += weight[i]*pca.components_[i]
    return reimage

fig = plt.figure(figsize=(16, 6))
for i in range(len(n)):
    ax = fig.add_subplot(1, len(n), i + 1, xticks=[], yticks=[])
    ax.imshow(reconstruction(n[i], weights).reshape(56,46),cmap=plt.cm.bone)
    norm_image = cv2.normalize(reconstruction(n[i], weights), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8).reshape(-1,1)
    mse = mean_squared_error(imagedata, norm_image)
    ax.set_title('n='+str(n[i])+'   MSE='+'%.2f'%(mse))
```
<img src="https://i.imgur.com/jjoherL.png" width="700"/>

* $n=3$, $MSE=8298.68$
* $n=50$, $MSE=1197.48$
* $n=170$, $MSE=1087.68$
* $n=240$, $MSE=1045.82$
* $n=345$, $MSE=980.33$


---

### <font face="Times">4. (20%) Now, apply the k-nearest neighbors algorithm to classify the testing set images. First,you will need to determine the best k and n values by 3-fold cross-validation. For simplicity, the choices for such hyperparameters are k = {1, 3, 5} and n = {3, 50, 170}. Show the cross-validation results and explain your choice for (k, n).</font>

``` python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

dim = [3, 50, 170]
k_set = [1, 3, 5]

def pca_preprocessing(n, data): # d^2 x N
    p = decomposition.PCA(n_components=n, whiten=True, svd_solver='full').fit(data.T)
    return p

score = {}

for d in dim:
    pca = pca_preprocessing(d, train_set)
    for k in k_set:
        knn = KNeighborsClassifier(n_neighbors=k)
        auc = cross_val_score(knn, pca.transform(train_set.T), train_label, cv=3, scoring='accuracy')
        score['n='+str(d)+','+'k='+str(k)] = np.mean(auc)
        print('n='+str(d)+', '+'k='+str(k)+', '+'auc='+str(np.mean(auc)))
```
$n=3$, $k=1$, $accuracy=0.672$
$n=3$, $k=3$, $accuracy=0.603$
$n=3$, $k=5$, $accuracy=0.531$
==$n=50$, $k=1$, $accuracy=0.889$==
$n=50$, $k=3$, $accuracy=0.842$
$n=50$, $k=5$, $accuracy=0.800$
$n=170$, $k=1$, $accuracy=0.608$
$n=170$, $k=3$, $accuracy=0.352$
$n=170$, $k=5$, $accuracy=0.261$

<font face="Times" style="font-size: 22px;">I select the hyperparameters set of highest accuracy on training set when $n=50$, $k=1$ </font>


---
### <font face="Times">5. (20%) Use your hyperparameter choice in 4. and report the recognition rate of the testing set.</font>

``` python
# predict on testing set
knn = KNeighborsClassifier(n_neighbors=1)
pca = pca_preprocessing(50, train_set)
knn.fit(pca.transform(train_set.T), train_label)
knn.score(pca.transform(test_set.T), test_label.reshape(-1,1))
```
``` python
accuracy=0.875
```
---

## <font face="Times">Reference</font>

> https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_eigenfaces.html
> https://en.wikipedia.org/wiki/Eigenface
> https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
> https://scikit-learn.org/stable/modules/cross_validation.html
> https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html











