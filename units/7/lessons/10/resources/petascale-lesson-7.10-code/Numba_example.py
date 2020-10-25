#!/usr/bin/env python
# coding: utf-8

# Blue Waters Petascale Semester Curriculum v1.0
# Unit 7: CUDA
# Lesson 10: Numba for CUDA GPUs
# File: Numba_example.ipynb
# Developed by Sanish Rai for the Shodor Education Foundation, Inc.
#
# Copyright (c) 2020 The Shodor Education Foundation, Inc.
#
# Browse and search the full curriculum at
# <http://shodor.org/petascale/materials/semester-curriculum>.
#
# We welcome your improvements! You can submit your proposed changes to this
# material and the rest of the curriculum in our GitHub repository at
# <https://github.com/shodor-education/petascale-semester-curriculum>.
#
# We want to hear from you! Please let us know your experiences using this
# material by sending email to petascale@shodor.org
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# In[ ]:





# In[1]:

#Instructions to run
#This python file is extracted from the jupyter notebook version
#Before running the program make sure CUDA, Python and Numba are installed correctly
#To run the program, type: python Numba_example.py
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
    x, y = cuda.grid(2)

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    C[x,y]=A[x,y]-B[x,y]


# In[2]:



# Initialite the data array
A = numpy.ones([48,48], dtype = float)
B = numpy.ones([48,48], dtype =float)

#copy the host variables to device
A_global_mem = cuda.to_device(A)
B_global_mem = cuda.to_device(B)

#Create memory for C in device
C_global_mem = cuda.device_array((48,48)) 

# Configure the blocks
threadsperblock = (TPB, TPB)
blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[1]))
blockspergrid_y = int(math.ceil(B.shape[1] / threadsperblock[0]))
blockspergrid = (blockspergrid_x, blockspergrid_y)


# In[3]:


# Start the kernel 
kernel_op[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
#copy the result to CPU
res = C_global_mem.copy_to_host()


# In[4]:


print(res)


# In[5]:


print("The sum is ",numpy.sum(res))


# In[ ]:




