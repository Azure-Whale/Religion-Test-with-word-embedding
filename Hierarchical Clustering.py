# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn import *
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage

correlation_matrix=pd.read_csv('Regreesion&lists/Corr_matrix.csv')
correlation_matrix = np.sqrt(-correlation_matrix+1)
print(correlation_matrix)
linked = linkage(correlation_matrix, 'single')
plt.figure(figsize = (40,10))
dendrogram(linked,
                    orientation = 'top',
                    labels = list(correlation_matrix.columns),
                    distance_sort = 'descending',
                    show_leaf_counts = True
)
#plt.figure(figsize = (20,1))
plt.savefig('hierarchical map.png')
plt.show()
