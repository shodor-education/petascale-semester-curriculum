# Blue Waters Petascale Semester Curriculum v1.0
# Unit 10: Productivity and Visualization
# Lesson 2: Python Scripting 2
# File: Python_Plot_Example.py
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

# Setting up x and y data
x=[1,1.5,2,2.5,3,3.5]
y=[1.0,3.0,3.9,5.05,5.8,7.9]

# Fitting the x and y data to a line
fit=np.polyfit(x,y,1)  #np.polyfit is a NumPy fitting function 
fitfunction=np.poly1d(fit)   #Take the output of polyfit and makes a function

# Plotting data
plt.plot(x,y,'bo')   # Plot the original x,y data
plt.plot(x,fitfunction(x),'r-')     # Plot the fitted data

# Axis labeling.  Argument is a string
plt.xlabel('Concentration (M)')
plt.ylabel('Amplitude (arb.)')

####### Only uncomment one at a time
### Show the plot
# plt.show()

### Save the plot
# plt.savefig('My_Figure.pdf')
