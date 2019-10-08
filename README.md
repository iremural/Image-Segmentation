# Image Segmentation
An accurate segmentation of 10 images from Breast Cancer Histopathological Database is tried to be achieved by using the color and texture informations of images. It is hard to find fixed parameters for the segmentation of all images. In order to overcome this problem, the following technique is applied:

Firstly, SLIC0 algorithm is used to find the superpixels that are homogenous and preserve the object boundaries. 
Secondly, Gabor texture information of each superpixel is calculated. This is accomplished by using a filter bank of 4 scales and 4 orientations, and it is applied on the grayscale version of the images.
After computing Gabor features of each superpixel, lab color space is used to find color representations of each superpixel.
Then, in order to group the superpixels K-means clustering algorithm is used.

Further detailed description and results can be find from : [Project Report]()

