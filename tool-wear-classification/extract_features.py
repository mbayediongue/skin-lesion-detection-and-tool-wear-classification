# -*- coding: utf-8 -*-
"""
@author: Victor Gonzalez Castro (victor.gonzalez@unileon.es)

"""

import os
import glob
from sklearn.preprocessing import LabelEncoder
from skimage.io import imread
from skimage import util
from skimage.measure import label, regionprops
import numpy as np
import h5py


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

    shape_features = np.empty(shape=10)

     # Look at the documentation of regionprops (specifically, to the 
    # example that is shown in the documentation) to get the properties
    # of the region that you need to get the descriptors of shape_features 
    # (mentioned above)
    # ====================== YOUR CODE HERE ======================
    props = regionprops(image_region)
    # ============================================================


    # Convex Area: Number of pixels of convex hull image, which is the 
    # smallest convex polygon that encloses the region
    # ====================== YOUR CODE HERE ======================
    shape_features[0]=props[0].convex_area
    # ============================================================

    # Eccentricity: Eccentricity of the ellipse that has the same second-
    # moments as the region
    # ====================== YOUR CODE HERE ======================
    shape_features[1]=props[0].eccentricity
    # ============================================================

    # Perimeter: Perimeter of object which approximates the contour as a 
    # line through the centers of border pixels using a 4-connectivity
    # ====================== YOUR CODE HERE ======================
    shape_features[2]=props[0].perimeter
    # ============================================================

    # Equivalent Diameter: The diameter of a circle with the same area as 
    # the region
    # ====================== YOUR CODE HERE ======================
    shape_features[3]=props[0].equivalent_diameter
    # ============================================================
    
    # Extent: Ratio of pixels in the region to pixels in the total 
    # bounding box
    # ====================== YOUR CODE HERE ======================
    shape_features[4]=props[0].extent
    # ============================================================

    # Filled Area: Number of pixels of the region will all the holes 
    # filled in
    # ====================== YOUR CODE HERE ======================
    shape_features[5]=props[0].filled_area
    # ============================================================

    # Minor Axis Length: The length of the minor axis of the ellipse that 
    # has the same normalized second central moments as the region.
    # ====================== YOUR CODE HERE ======================
    shape_features[6]=props[0].minor_axis_length
    # ==========================================================

    # Major Axis Length: The length of the major axis of the ellipse that 
    # has the same normalized second central moments as the region.
    # ====================== YOUR CODE HERE ======================
    shape_features[7]=props[0].major_axis_length
    # ============================================================

    # R: Ratio between the major and minor axis of the ellipse that has 
    # the same second central moments as the region.
    minor_len=props[0].minor_axis_length # length of the minor axis of the ellipse
    major_len=props[0].major_axis_length # # length of the major axis of the ellipse
    R=major_len/minor_len
    shape_features[8]=R
    # ============================================================

    # Solidity: Ratio of pixels in the region to pixels of the convex 
    # hull image.
    # ====================== YOUR CODE HERE ======================
    shape_features[9]=props[0].solidity
    # ============================================================

    return (shape_features)

# ------------------------
# ----- MAIN PROGRAM -----
# ------------------------

dir_base = "./Images"
dir_edges = "Edges"
dir_images_edges = dir_base + "/" + dir_edges
dir_output = "Output"
features_path = dir_output + "/features_geometric.h5"
labels_path = dir_output + "/labels_high-low.h5"

image_labels = os.listdir(dir_images_edges)

labels = []
# variables to hold features and labels
X = np.empty((0, 10))
Y = np.array([])

for i, lab in enumerate(image_labels):
    cur_path = dir_images_edges + '/' + lab

    print(cur_path)
    # Check how many files there are, together with their extensions.
    # The images are in .png format in this lab
    for image_path in glob.glob(cur_path + "/*.png"):

        #print("[INFO] Processing image " + image_path)

        img_ori = imread(image_path)
        img = util.img_as_ubyte(img_ori)

        features = get_shape_features(img)

        #print("[INFO] ...Storing desctiptors of image " + image_path)

        # To simplify the problem, in this lab we are going to make it 
        # binary. 
        # Therefore, the high wear level will be class 1, whereas 
        # the low and medium levels will be class 0
        if lab == "2_High":
            lab_num = 1
        else:
            lab_num = 0

        # The descriptors will be stored in an array in which each row is 
        # a different descriptor. 
        # The labels will be stored in a vector
        X = np.append(X, np.array([np.transpose(features)]), axis=0)
        Y = np.append(Y, lab_num)

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

h5f_data.create_dataset("dataset_inserts_geometric", data=X)

h5f_label = h5py.File(labels_path, 'w')
h5f_label.create_dataset("dataset_inserts_geometric", data=Y)

h5f_data.close()
h5f_label.close()
