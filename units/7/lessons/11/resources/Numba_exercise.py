#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Source code reference: https://numba.pydata.org/numba-doc/latest/cuda/examples.html#matrix-multiplication
#Source code reference: https://nyu-cds.github.io/python-numba/

from numba import cuda, float32
import numpy
import numba
import math

# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
TPB = 16

@cuda.jit
def kernel_op(A, B, C):

    #cuda.grid returns the absolute position of the current thread in the entire grid of blocks
    x = cuda.grid(1)

    if x >= C.shape[0] :
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    C[x]=A[x]+B[x]


# In[6]:



# Initialite the data array
A = numpy.ones(48, dtype = float)
B = numpy.ones(48, dtype =float)

#copy the host variables to device
A_global_mem = cuda.to_device(A)
B_global_mem = cuda.to_device(B)

#Create memory for C in device
C_global_mem = cuda.device_array((48)) 

# Configure the blocks
threadsperblock = TPB
blockspergrid_x = int(math.ceil(len(A) / threadsperblock))


# In[7]:


# Start the kernel 
kernel_op[blockspergrid_x, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
#copy the result to CPU
res = C_global_mem.copy_to_host()


# In[8]:


print(res)


# In[9]:


print("The sum of the results is ",numpy.sum(res))


# In[ ]:





# In[ ]:




