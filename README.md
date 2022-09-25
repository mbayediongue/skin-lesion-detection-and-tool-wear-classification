# skin-lesion-detection-from-segmented-images-and-superpixels

This work uses the database of the ISIC Challenge 2017: [Skin Lesion Analysis Towards Melanoma Detection](https://ieeexplore.ieee.org/document/8993219) [1].


It focuses on the third and fourth parts of the challenge (feature extraction and disease classification). Considering that we already have the segmented images and the superpixels of the original images. The main challenge is the small size of the data set (only 200 images). The proposed solution did not use directly the images but used morphological, intensity and texture descriptors extracted from the images to build a SVM classifier.


Here is a [link](https://drive.google.com/drive/folders/1vFrGFBBJygRIfYiHEQeuzUlyBJDm8MO-?usp=sharing) to directly download the database containing images (original images, superpixels and segmendted images) from Google Drive.

Python is used for different classification approaches using morphological features, texture descriptors and intensity descriptors extracted from the images. At the end, all this features are combined to build the final SVM classifier.

Attached files contain two Python scripts, one for feature extraction and the other for the classifier building.

Please read the report (file report_skin_lesion_detection.pdf, written in French) for more details about the methods used.



[1] S. Sreena and A. Lijiya, "Skin Lesion Analysis Towards Melanoma Detection," 2019 2nd International Conference on Intelligent Computing, Instrumentation and Control Technologies (ICICICT), 2019, pp. 32-36, doi: 10.1109/ICICICT46008.2019.8993219.
