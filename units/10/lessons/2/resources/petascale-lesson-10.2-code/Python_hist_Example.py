# Blue Waters Petascale Semester Curriculum v1.0
# Unit 10: Productivity and Visualization
# Lesson 2: Python Scripting 2
# File: Python_hist_Example.py
# Developed by Michael N. Groves for the Shodor Education Foundation, Inc.
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


import numpy as np      # Importing NumPy
import matplotlib.pyplot as plt     # Importing MatPlotLib

# Setting up the data and the bins
### np.random.random_sample generates n random numbers in [0,1)
### This is multiplied by 10 to get [0,10)
data=np.random.random_sample(1000)*10
### Setting up the bins so that the data is grouped into integers
### Must include the minimum limit on the bottom bin
### and the maximum limit on the top bin
mybins = [0,1,2,3,4,5,6,7,8,9,10]

# Plotting data
plt.hist(x=data,bins=mybins)

# Axis labeling.  Argument is a string
plt.xlabel('Data')
plt.ylabel('Frequency')

####### Only uncomment one at a time
### Show the plot
# plt.show()

### Save the plot
# plt.savefig('My_Figure.pdf')
