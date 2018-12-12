# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 00:40:58 2018

@author: Atul
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 23:55:18 2018

@author: Atul
"""

#PCA

import numpy as np
import pandas as pd
A = np.random.random((1000,40))


cov_B = np.cov(A , rowvar = 0)  # covariance of matrix

eigen_val,eigen_vec = np.linalg.eig(cov_B)  # eigen values and eigen vectors

index_eigval = np.argsort(eigen_val)        # sorting the index of eigen values

sorted_eigval=np.zeros([len(index_eigval)]) # creating sorted eigen values array

sorted_eigvec=np.zeros(np.shape(eigen_vec)) # creating sorted eigen vectors array

# sorting eigen values with index position
for i in range (len(index_eigval)):
   sorted_eigval[len(index_eigval)-i-1] = eigen_val[index_eigval[i]]
   sorted_eigvec[:,len(index_eigval)-i-1]=eigen_vec[:,index_eigval[i]]
   
   
# calculating weight of all eigen values
commulative_weight=np.zeros([len(index_eigval)])
count=0
for i in range (len(index_eigval)):
    commulative_weight[i]=(sorted_eigval[i]+count)/np.sum(sorted_eigval)
    count=count+sorted_eigval[i]

# To check how much data we need to save   
v=0.9   
a=np.where(commulative_weight>=0.9)
a=a[0]
a=a[0]

# important eigen vectors 
imp_eigvec = sorted_eigvec[:,0:a+1]

new_data = np.zeros([1000,40])

# append old data value into new data after appying pca 
for i in range(0,35):
    new_data[:,index_eigval[i]] = A[:,index_eigval[i]]
    
 # convert new data into dataframe   
new_data = pd.DataFrame(new_data)

# replace 0 by nan and drop nan
new_data = new_data.replace(0,np.nan)  
new_data = new_data.dropna(axis = 1) 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    