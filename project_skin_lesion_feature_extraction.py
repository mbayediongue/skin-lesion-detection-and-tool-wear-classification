# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 13:56:04 2021

@author: mbaye
"""

import os
import glob
from skimage import util
from skimage.measure import regionprops
import numpy as np
import h5py
from scipy import ndimage
import skimage.io
import pandas as pd
import scipy.ndimage
import skimage.measure # some geometrical descriptors
import matplotlib.pyplot as plt

def get_shape_features(image_region):
    """

    Parameters
    ----------
    image : ubyte array
        Black and white image with the region that is going to be described

    Returns
    -------
    Vector with the descriptor of the input image

    """
    shape_features = np.empty(shape=8)

     # Look at the documentation of regionprops for descriptors of shape features
    props = regionprops(image_region)
    # ============================================================
    # Perimeter: Perimeter of object which approximates the contour as a 
    # line through the centers of border pixels using a 4-connectivity
    peri=props[0].perimeter
    
    # Convex Area: Number of pixels of convex hull image, which is the 
    # smallest convex polygon that encloses the region
    area=props[0].convex_area
    
    # feature 1: a ration between perimeter and area
    shape_features[0]=4*np.pi*area/peri**2

    # ============================================================

    # Eccentricity: Eccentricity of the ellipse that has the same second-
    # moments as the region
    shape_features[1]=props[0].eccentricity
    
    # R: Ratio between the major and minor axis of the ellipse that has 
    # the same second central moments as the region.
    minor_len=props[0].minor_axis_length # length of the minor axis of the ellipse
    major_len=props[0].major_axis_length # # length of the major axis of the ellipse
    
    R=major_len/minor_len
    
    shape_features[2]=R
    
    # ============================================================

    # Solidity: Ratio of pixels in the region to pixels of the convex 
    # hull image.
    # ====================== YOUR CODE HERE ======================
    shape_features[3]=props[0].solidity
    # ============================================================
    
    
    # On ajoute quelques "features" déja utilisées dans le TP 'Shape digrams'
    radius = inscribedRadius(image_region);
    d,D = feret_diameter(image_region);
    crofton = crofton_perimeter(image_region);
    
    elongation=d/D
    thinness=2*radius / D;
    roundness=4*np.sum(image_region>100)/(np.pi * D**2);
    z=crofton / (np.pi * D);
    
    shape_features[4]=elongation
    shape_features[5]=thinness
    shape_features[6]=roundness
    shape_features[7]=z
    
    return (shape_features)

def crofton_perimeter(I):
    """ Computation of crofton perimeter
    """
    inter = [];
    h = np.array([[1, -1]]);
    for i in range(4):
        II = np.copy(I);
        I2 = ndimage.rotate(II, 45*i, mode='nearest');
        I3 = scipy.ndimage.convolve(I2, h);
        
        inter.append(np.sum(I3>100));
        
        
    crofton = np.pi/4. * (inter[0]+inter[2] + (inter[1]+inter[3])/np.sqrt(2));
    return crofton

def feret_diameter(I):
    """ 
    Computation of the Feret diameter
    minimum: d (meso-diameter)
    maximum: D (exo-diameter)
    
    Input: I binary image
    """
    d = np.max(I.shape);
    D = 0;
    
    for a in np.arange(0, 180, 30):
        I2 = ndimage.rotate(I, a, mode='nearest');
        F = np.max(I2, axis=0);
        measure = np.sum(F>100);
        
        if (measure<d):
            d = measure;
        if (measure>D):
            D = measure;
    return d,D;

def inscribedRadius(I):
    """
    computes the radius of the inscribed circle
    """
    dm = scipy.ndimage.morphology.distance_transform_cdt(I>100);
    radius = np.max(dm);
    return radius;


def histogramProperties(Image):
    I = Image # cv2.cvtColor(Image, cv2.COLOR_RGB2YCrCb) # espace YCrCb
    #n,m,_=Image.shape

    nb_features=7
    
    features = np.empty(shape=2*nb_features)
    
    for i in range(2): # Car on a une image colorée dans l'espace 2YCrCb
         # h: histogramme des probabilités (h[k] est la proportion de pixels ayant la valeur k avec 0<=k<=256)
         h, edges = np.histogram(I[:,:,i], density=True, bins=256)
      
         vect=np.array([i for i in range(256)])
         # La moyenne
         moy=sum(h*vect)
         features[0+nb_features*i]= moy
             
         # L'écart type
         sd=np.sqrt( sum( ( vect-moy)**2*h ) )
         features[1+nb_features*i]=sd
        
         # La skew
         features[2+nb_features*i]=sum( ((vect-moy)**3*h)/sd**3)
        
         # Energie
         features[3+nb_features*i]=sum( h**2)
        
         # Entropie
         features[4+nb_features*i]=-sum( h*np.log2(h+1e-10) )
         
         # Q2 : quantile à 25%
         features[5+nb_features*i]=np.quantile(h,0.25)
         
         # Q3: quantile à 75%
         features[6+nb_features*i]=np.quantile(h,0.25)
        
    return features


# Local binary pattern
def LBP(I):
    """
    Local Binary Pattern of image I
    construct descriptor for each pixel, then evaluate histogram
    I: grayscale image (size nxm)
    """
    B = np.zeros(np.shape(I))

    code = np.array([[1, 2, 4], [8, 0, 16], [32, 64, 128]])

    # loop over all pixels except border pixels
    for i in np.arange(1, I.shape[0]-2):
        for j in np.arange(1, I.shape[1]-2):
            w = I[i-1:i+2, j-1:j+2]
            w = w >= I[i, j]
            w = w * code
            B[i, j] = np.sum(w)

    h, edges = np.histogram(B[1:-1, 1:-1], density=True, bins=256)
    plt.plot(h)
    nb_features=5
    
    features = np.empty(shape=1*nb_features)
    vect=np.array([i for i in range(256)])
    # La moyenne
    moy=sum(h*vect)
    features[0]= moy
    
    # L'écart type
    sd=np.sqrt( sum( ( vect-moy)**2*h ) )
    features[1]=sd
       
    # La skew (asymétrie)
    features[2]=sum( ((vect-moy)**3*h)/sd**3)
       
    # Energie
    features[3]=sum( h**2)
       
    # Entropie
    features[4]=-sum( h*np.log2(h+1e-10) )
    
    return features


def check_if_directory_exists(name_folder):
    """
    check_if_directory_exists(name_folder)
    INPUT:
        name_folder: name of the directory to be checked
    OUTPUT:
        a message indicating that the directory does not exist and if it is created
        
    @author: Eduardo Fidalgo (EFF)
    """

    if not os.path.exists(name_folder):
        print(name_folder + " directory does not exist, created")
        os.makedirs(name_folder)
    else:
        print(name_folder + " directory exists, no action performed")


# ------------------------
# ----- MAIN PROGRAM -----
# ------------------------


original=[] # original images
segmented =[] # segmented images
superpixels=[]
img_names=[]
liste_file=glob.glob("./PROJECT_Data/*")

for i, fileName in enumerate( liste_file):
    X=skimage.io.imread(fileName )
    #print(fileName[-15:-4])
    if fileName[-15:-4]=='superpixels':
        superpixels.append(X)
    elif fileName[-16:-4]=='segmentation':
        segmented.append(X)
    else:  # original images
        original.append(X)
        img_names.append(fileName[-16:-4])


# Holding true labels (1: melanomas and 0: not confirmed as melanomas)
Y = np.array([])
data_=pd.read_csv("ISIC-2017_Data_GroundTruth_Classification.csv", sep=",")
data=np.array(data_)
nb_img=len(original) # number of images
j=0
for i in range(nb_img):

    found=False
    while not found:
        if data[j,0]==img_names[i]:
            lab_num=int( data[j,1])
            Y = np.append(Y, lab_num)
            found=True
        j+=1

# Extract geometrical and morphological features using segmented images
# variables to hold features and labels
nb_geo_features=10  # number of GEOMETRICAL and MORPHOLOGICAL features
nb_other_features=0
nb_features=nb_geo_features+nb_other_features
Xgeo = np.empty((0, nb_geo_features))  # variables to hold geometrical features

for i in range( len(segmented)):
   img = util.img_as_ubyte( segmented[i])
   features = get_shape_features(img)
   Xgeo = np.append(Xgeo, np.array([np.transpose(features)]), axis=0)
   #print(i)


X=Xgeo


# Extract Local binary patterns and intensity/texture decriptors from
# original and superpixels
# On utilise pour ces descripteurs les LBP (local binary pattern) et des 'histogrammes d'intensité'
nb_features=14  # number of features extracted from original images and superpixels
features=[0 for i in range(nb_features)]
nn=200
hh=[]
for i in range(nn):
    features[0:14]= histogramProperties(original[i])
    #features[14:19]=LBP( original[i][:,:,0])
    hh.append( features[:])
    #print(i)

"""
hh_bar=[]
ind=[ 5,6,12,13]
for i in range(nn):
    x=[ hh[i][j] for j in ind]
    hh_bar.append(x)
"""
# Regroupement de tous des descripteurs morphologiques, d'intensité et de texture
X=np.concatenate((hh,Xgeo), axis=1)

# Sauvegarde des descripteurs et des labels pour ne pas à les recaluler à nouveau (car chronophage)
dir_output = "Output_intensi"
features_path = dir_output + "/features.h5"
labels_path = dir_output + "/labels_high-low.h5"
print("/n")
print("/n")
print("[INFO] Saving descriptors in folder " + dir_output)

# Save the features and labels into a hdf5 file in the directory dir_output
# If the directory does not exist, create it
check_if_directory_exists(dir_output)

# Save features and labels
try:
    h5f_data = h5py.File(features_path, 'w')
except:
    a = 1

h5f_data.create_dataset("dataset_skin_lesion", data=X)

h5f_label = h5py.File(labels_path, 'w')
h5f_label.create_dataset("dataset_skin_lesion", data=Y)

h5f_data.close()
h5f_label.close()
