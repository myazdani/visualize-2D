
# coding: utf-8

# In[12]:

import os
from pylab import *
from sklearn import (manifold, datasets, decomposition, ensemble, lda,
                     random_projection)
from skimage.feature import hog
from skimage.filters import gaussian_filter
from skimage import color
from scipy import misc
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd

get_ipython().magic(u'pylab inline')


# ## Specify image (source) path and feature write (target) path

# In[13]:

#src_path = "/Users/jaja/Documents/SD-digital-globe-slices/slices/"
src_path = "/Users/myazdaniUCSD/Documents/paintings/ROTHKO_MONDRIAN/Rothko_Mondrian_Complete image sets/Rothko_images/"

target_path = "../features"
if not os.path.exists(target_path):
    os.makedirs(target_path)


# ##Specify number of images to use per chunk of feature extraction

# In[37]:

num_imgs_per_chunk = 10


# ## Get image paths and setup chunk paths:

# In[38]:

image_type = (".png", ".jpg", ".JPG", ".jpeg", ".PNG")
 
image_paths = []  
for root, dirs, files in os.walk(src_path):
    image_paths.extend([os.path.join(root, f) for f in files if f.endswith(image_type)])

print "Number of images:", len(image_paths)


# In[39]:

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]
        
image_paths_chunks = chunks(image_paths, num_imgs_per_chunk)


# ## Extract Features

# In[43]:

# def return_HOG(img_path):
#     img = misc.imread(img_path)
#     hogs = []
#     for i in range(3):
#         img_c = misc.imresize(img[:,:,i], (80,100))
#         hog_c = hog(img_c, orientations=16, pixels_per_cell=(2, 2),cells_per_block=(1, 1), visualise=False)
#         hogs.extend(hog_c)
#     return array(hogs)

def return_HOG(img_path):
    img = misc.imread(img_path)
    new_img = misc.imresize(color.rgb2gray(img), (100,100))
    return hog(gaussian_filter(new_img,.1), orientations=8, pixels_per_cell=(4, 4),cells_per_block=(1, 1), visualise=False)


# In[44]:

for i, image_paths_chunk in enumerate(list(image_paths_chunks)):
    print "working on chunk", i
    df = pd.DataFrame({"image_paths": array(image_paths_chunk)})
    df["hogs"] = df["image_paths"].apply(lambda x: return_HOG(x))
    df.to_pickle(os.path.join(target_path, "hogs_chunk" + str(i) + ".pkl"))


# In[41]:

list(image_paths_chunks)


# In[36]:

image_paths


# In[ ]:



