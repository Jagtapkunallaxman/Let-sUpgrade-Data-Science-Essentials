#!/usr/bin/env python
# coding: utf-8

# # <u>Eigenvalues and Eigenvectors</u>

# Eigen vector of a square matrix A is a vector represented by a matrix X such that when X is multiplied with matrix A, then the direction of the resultant matrix remains same as vector X.
# 
# The corresponding eigenvalue, denoted by λ, is the factor by which the eigenvector is scaled.
# 
# **AX = λX**,
# 
# where A is any arbitrary matrix, λ are eigen values and X is an eigen vector corresponding to each eigen value.
# 
# **Syntax**: w,v = np.linalg.eig()
# 
# **Return**: w : eigen values v : eigen vectors
# 
# Application Whenever there is a complex system having large number of dimensions with a large number of data, eigenvectors and eigenvalues concepts help in transforming the data in a set of most important dimensions (principal components). This will result in processing the data in a faster manner.

# In[1]:


import numpy as np 
a = np.array([[50, 30], [23, 33]])
eig_Va, eig_Ve = np.linalg.eig(a) 

print("Printing the Eigen values of the given square array:\n", eig_Va)
print("Printing the Eigenvectors of the given square array:\n", eig_Ve)


# # <u>digitize()</u>

# With the help of np.digitize() method, we can get the indices of the bins to which the each value is belongs to an array by using np.digitize() method.
# 
# **Syntax**: np.digitize(Array, Bin, Right)
# 
# **Return**: Return an array of indices of the bins.

# In[2]:


import numpy as np 
a = np.array([1.23, 2.4, 3.6, 4.8]) 
bins = np.array([1.0, 1.3, 2.5, 4.0, 10.0]) 
b = np.digitize(a, bins) 

print(b)


# # <u>repeat()</u>

# numpy.repeat(a, repeats, axis=None)<br>
# Repeat elements of an array.
# 
# Parameters<br>
# a : array_like **Input array**.
# 
# **repeats**: int or array of ints<br>
# The number of repetitions for each element. repeats is broadcasted to fit the shape of the given axis.
# 
# **axis**: int, optional<br> 
# The axis along which to repeat values. By default, use the flattened input array,
# and return a flat output array.
# 
# **Returns**:
# repeated_array: ndarray<br> 
# Output array which has the same shape as a, except along the given axis.

# In[4]:


import numpy as np
a = np.repeat([[2,3,4],[2,3,4]],2, axis=0)
print(a)


# # <u>squeeze()</u>
# 
# numpy.squeeze(a, axis=None)<br>
# Remove single-dimensional entries from the shape of an array.
# 
# **Parameters**:<br>
# a : array_like
# 
# **Input data**:<br>
# **axis**: None or int or tuple of ints, optional<br>
# Selects a subset of the single-dimensional entries in the shape. If an axis is selected with shape entry greater than one, an error is raised.
# 
# **Returns**:
# **squeezed**: ndarray<br>
# The input array, but with all or a subset of the dimensions of length 1 removed.<br>
# This is always a itself or a view into a.<br>
# Note that if all axes are squeezed, the result is a 0d array and not a scalar.
# 
# **Raises**: ValueError<br>
# If axis is not None, and an axis being squeezed is not of length 1

# In[6]:


import numpy as np 

in_arr = np.array([[[2, 2, 2], [2, 2, 2]]]) 

print ("Input array : ", in_arr)  
print("Shape of input array : ", in_arr.shape)   
  
out_arr = np.squeeze(in_arr)  
  
print ("output squeezed array : ", out_arr) 
print("Shape of output array : ", out_arr.shape)


# # <u>linspace()<u/>
#     
# **syntax()**:<br>
# numpy.linspace(start,stop,num = 50,endpoint = True,retstep = False,dtype = None)
# 
# * start : [optional] start of interval range. By default start = 0<br>
# * stop : end of interval range<br>
# * restep : If True, return (samples, step). By deflut restep = False<br>
# * num : [int, optional] No. of samples to generate<br>
# * dtype : type of output array<br>
# 
# **Return**:
# * ndarray<br>
# * step: [float, optional], if restep = True

# In[9]:


import numpy as np 
import pylab as p
x1 = np.linspace(0, 2, 10, endpoint = False) 
y1 = np.ones(10) 
  
p.plot(x1, y1, '*') 
p.xlim(-0.2, 1.9)


# # <u>clip()<u/>
#     
# **Syntax**: numpy.clip(a, a_min, a_max, out=None)
# 
# **Parameters**:<br>
# a: Array containing elements to clip.
# 
# a_min : Minimum value.<br>
# * If None, clipping is not performed on lower interval edge. Not more than one of a_min and a_max may be None.
# 
# a_max : Maximum value.<br>
# * If None, clipping is not performed on upper interval edge. Not more than one of a_min and a_max may be None.<br>
# * If a_min or a_max are array_like, then the three arrays will be broadcasted to match their shapes.<br>
# out : Results will be placed in this array. It may be the input array for in-place clipping. out must be of the right shape to hold the output. Its type is preserved.<br>
# Return : clipped_array

# In[10]:


import numpy as np 
  
in_array = [1, 2, 3, 4, 5, 6, 7, 8 ] 
print ("Input array : ", in_array) 
  
out_array = np.clip(in_array, a_min = 1, a_max = 6) 
print ("Output array : ", out_array)


# # <u>extract()<u/>
# **Syntax**: numpy.extract(condition, array)
# 
# **Parameters**:<br>
# **array**: Input array. User apply conditions on input_array elements<br>
# **condition**: [array_like]Condition on the basis of which user extract elements. Applying condition on input_array, if we print condition, it will return an array filled with either True or False. Array elements are extracted from the Indices having True value.<br>
# **Returns**:
# Array elements that satisfy the condition.

# In[11]:


import numpy as np

array = np.arange(10).reshape(5, 2)
print("Original array : \n", array)

condition = np.mod(array, 4) == 0

# This will show element status of satisfying condition
print("\nArray Condition : \n", condition)

# This will return elements that satisy condition "a" condition
print("\nElements that satisfies the condition: \n",
      np.extract(condition, array))


# # <u>argpartition()<u/>
# 
# argpartition() function is used to create a indirect partitioned copy of input array with its elements rearranged in such a way that the value of the element in k-th position is in the position it would be in a sorted array. All elements smaller than the k-th element are moved before this element and all equal or greater are moved behind it. The ordering of the elements in the two partitions is undefined.It returns an array of indices of the same shape as arr, i.e arr[index_array] yields a partition of arr.<br>
#     
# **Syntax**: numpy.argpartition(arr, kth, axis=-1, kind=’introselect’, order=None)<br>
# 
# **Parameters**:<br>
# **arr**: [array_like] Input array.<br>
# **kth**: [int or sequence of ints ] Element index to partition by.<br>
# **axis**: [int or None] Axis along which to sort. If None, the array is flattened before sorting. The default is -1, which sorts along the last axis.<br>
# **kind**: Selection algorithm. Default is ‘introselect’.<br>
# **order**: [str or list of str] When arr is an array with fields defined, this argument specifies which fields to compare first, second, etc.<br>
# 
# **Return**: [index_array, ndarray] Array of indices that partition arr along the specified axis.

# In[12]:


import numpy as np 
  
# input array 
in_arr = np.array([ 2, 0,  1, 5, 4, 3]) 
print ("Input array : ", in_arr)  
  
out_arr = np.argpartition(in_arr, (0, 2)) 
print ("Output partitioned array indices: ", out_arr)


# # <u>setdiff1d()<u/>
#     
# **Syntax**: numpy.setdiff1d(arr1, arr2, assume_unique = False)
# 
# **Parameters**:<br>
# **arr1**: [array_like] Input array.<br>
# **arr2**: [array_like] Input comparison array.<br>
# **assume_unique**: [bool] If True, the input arrays are both assumed to be unique, which can speed up the calculation.
# Default is False.<br>
# 
# **Return**: [ndarray] 1D array of values in arr1 that are not in arr2. The result is sorted when assume_unique = False, but otherwise only sorted if the input is sorted.

# In[13]:


import numpy as np  
   
arr1 = [5, 6, 2, 3, 4, 6, 7, 8] 
arr2 = [4, 1, 3] 
   
a = np.setdiff1d(arr1, arr2) 
   
print (a)


# # <u>itemsize()<u/>
# 
# **Syntax**: numpy.ndarray.itemsize(arr)
#     
# **Parameters**:<br>
# **arr**: [array_like] Input array.
# 
# **Return**: [int] The length of one array element in bytes

# In[15]:


import numpy as np 
arr = np.array([1, 2, 3, 4], dtype = np.float64) 
a = arr.itemsize 
print (a)


# # <u>hstack()<u/>
# 
# **Syntax**: numpy.hstack(tup)<br>
# 
# **Parameters**:<br>
# **tup**: [sequence of ndarrays] Tuple containing arrays to be stacked. The arrays must have the same shape along all but the second axis.
# 
# **Return**: [stacked ndarray] The stacked array of the input arrays.

# In[16]:


import numpy as np
in_arr1 = np.array([ 1, 2, 3] ) 
print ("1st Input array : \n", in_arr1)  
  
in_arr2 = np.array([ 4, 5, 6] ) 
print ("2nd Input array : \n", in_arr2)  

out_arr = np.hstack((in_arr1, in_arr2)) 
print ("Output horizontally stacked array:\n ", out_arr)


# # <u>vstack()<u/>
# **Syntax**: numpy.vstack(tup)
# 
# **Parameters**:<br>
# **tup**: [sequence of ndarrays] Tuple containing arrays to be stacked. The arrays must have the same shape along all but the first axis.
# 
# **Return**: [stacked ndarray] The stacked array of the input arrays.

# In[17]:


in_arr1 = np.array([ 3, 4, 3] ) 
print ("1st Input array : \n", in_arr1)  
  
in_arr2 = np.array([ 80, 90, 60] ) 
print ("2nd Input array : \n", in_arr2)  
  
# Stacking the two arrays vertically 
out_arr = np.vstack((in_arr1, in_arr2)) 
print ("Output vertically stacked array:\n ", out_arr)


# # <u>hsplit()<u/>
#     
# **Syntax**: numpy.hsplit(arr, indices_or_sections)
#     
# **Parameters**:<br>
# **arr**: [ndarray] Array to be divided into sub-arrays.<br>
# **indices_or_sections**: [int or 1-D array] If indices_or_sections is an integer, N, the array will be divided into N equal arrays along axis.<br>
# If indices_or_sections is a 1-D array of sorted integers, the entries indicate where along axis the array is split Return : [ndarray] A list of sub-arrays.

# In[18]:


import numpy as np 
arr = np.arange(16.0).reshape(4, 4) 
a = np.hsplit(arr, 2) 

print (a)


# # <u>vsplit()<u/>
#     
# **Syntax**: numpy.vsplit(arr, indices_or_sections)
#     
# **Parameters**:<br>
# **arr**: [ndarray] Array to be divided into sub-arrays.
# 
# **indices_or_sections**: [int or 1-D array] If indices_or_sections is an integer, N, the array will be divided into N equal arrays along axis.<br>
# If indices_or_sections is a 1-D array of sorted integers, the entries indicate where along axis the array is split Return : [ndarray] A list of sub-arrays.

# In[19]:


import numpy as np 
arr = np.arange(9.0).reshape(3, 3) 
a = np.vsplit(arr, 1) 
print (a)


# # <u>View vs Shallow Copy<u/>
#     
# A view of a NumPy array is a shallow copy , i.e. it references the same data buffer as the original, so changes to the original data affect the view data and vice versa.
# 
# A shallow copy constructs a new compound object and then (to the extent possible) inserts references into it to the objects found in the original.
# 
# A shallow copy means constructing a new collection object and then populating it with references to the child objects found in the original. The copying process does not recurse and therefore won’t create copies of the child objects themselves. In case of shallow copy, a reference of object is copied in other object. It means that any changes made to a copy of object **do reflect** in the original object.

# # <u>Deep Copy<u/>
# Deep copy is a process in which the copying process occurs recursively. It means first constructing a new collection object and then recursively populating it with copies of the child objects found in the original. In case of deep copy, a copy of object is copied in other object. It means that any changes made to a copy of object do not reflect in the original object.

# # <u>copy()<u/>
#     
# **Syntax**: numpy.ndarray.copy(order='C')
# 
# **Parameters**:<br>
# **order**: Controls the memory layout of the copy.<br>
# ‘C’ means C-order,<br>
# ‘F’ means F-order,<br>
# ‘A’ means ‘F’ if a is Fortran contiguous, ‘C’ otherwise.<br>
# ‘K’ means match the layout of a as closely as possible.

# In[20]:


import numpy as np   
x = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], 
                                 order ='F')
  
y = x.copy() 
print("y is :\n", y)
print("\nx is equal to y :\n",x==y)


# # <u>meshgrid<u/>
#     
# The numpy.meshgrid function is used to create a rectangular grid out of two given one-dimensional arrays representing the Cartesian indexing or Matrix indexing.

# In[21]:


import numpy as np

x = np.linspace(-4, 4, 9) 
y = np.linspace(-5, 5, 11)
x_1, y_1 = np.meshgrid(x, y) 
  
print("x_1 = ") 
print(x_1) 
print("y_1 = ") 
print(y_1)


# # <u>swapaxes()<u/>
#     
# With the help of matrix.swapaxes() method, we are able to swap the axes a matrix by using the same method.
# 
# **Syntax**: matrix.swapaxes()<br>
# **Return**: Return matrix having interchanged axes

# In[22]:


import numpy as np       

a = np.matrix('[4, 1; 12, 3]') 

b = a.swapaxes(1,1) 
   
print(b)


# # <u>column_stack()<u/>
#     
# **Syntax**: numpy.column_stack(tup)
# 
# **Parameters**:<br>
# **tup**: [sequence of ndarrays] Tuple containing arrays to be stacked. The arrays must have the same first dimension.<br>
# **Return**: [stacked 2-D array] The stacked 2-D array of the input arrays.

# In[23]:


import numpy as np 
 
in_arr1 = np.array(( 1, 2, 3 )) 
print ("1st Input array : \n", in_arr1)  
  
in_arr2 = np.array(( 4, 5, 6 )) 
print ("2nd Input array : \n", in_arr2)  

out_arr = np.column_stack((in_arr1, in_arr2)) 
print ("Output stacked array:\n ", out_arr)

