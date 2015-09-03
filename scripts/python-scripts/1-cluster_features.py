
# coding: utf-8

# In[1]:

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
import gc


# ## Specify Number of Clusters

# In[2]:

num_clusters = 10


# ##Specify feature path

# In[3]:

src_path = "../features/"
file_type = ".pkl"

feature_paths = []  
for root, dirs, files in os.walk(src_path):
    feature_paths.extend([os.path.join(root, f) for f in files if f.endswith(file_type)])


# In[4]:

hog_features = pd.read_pickle(feature_paths[0])
for feature_path in feature_paths[1:]:
    hog_features_temp = pd.read_pickle(feature_path)
    hog_features = pd.concat([hog_features, hog_features_temp])
    del hog_features_temp
gc.collect()


# In[5]:

image_features = array(list(hog_features["hogs"]))
image_features.shape


# # Reduce dimensions and apply clustering

# In[6]:

pca = PCA(n_components=50)
X = pca.fit_transform(array(image_features))


# In[7]:

kmeans = KMeans(n_clusters = num_clusters, n_init = 100, n_jobs=1)
kmeans.fit(X)
clusters = kmeans.predict(X)
clusters_space = kmeans.transform(X)


# In[8]:

image_paths_hack = list(hog_features["image_paths"])
image_paths_rel = [image_path_hack.split("/")[-2] + "/" + image_path_hack.split("/")[-1] for image_path_hack in image_paths_hack]


# In[9]:

res_df = pd.DataFrame({'file_names' : image_paths_rel})


# In[10]:

res_df["clusters"] = clusters
#res_df = pd.concat([res_df, pd.DataFrame(clusters_space)], axis = 1)
res_df["cluster_dist"] = amin(clusters_space, axis = 1)


# In[11]:

res_df.to_csv("../features/hogs_clusters.csv", header = True, index = False)


# ## Save cluster centroids (optional)

# In[12]:

pd.DataFrame(kmeans.cluster_centers_).to_csv("../features/kmeans_centroids.csv", header = True, index = False)


# In[ ]:



