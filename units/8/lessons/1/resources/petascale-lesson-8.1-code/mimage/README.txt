# Blue Waters Petascale Semester Curriculum v1.0
# File: mimage/README.txt
# Developed by R. Phillip Bording for the Shodor Education Foundation, Inc.
# Included in the following lessons:
# - Unit 5 (MPI) Lesson 10: Wave Propagation in MPI
# - Unit 6 (Hybrid MPI + OpenMP) Lesson 3: Pebble in Pond Wave Equation
# - Unit 8 (OpenACC) Lesson 1: Accelerating Scientific Applications
# - Unit 10 (Productivity and Visualization) Lesson 3: Visualization 1
#
# This file by The Shodor Education Foundation, Inc. is licensed under
# CC BY-SA 4.0. To view a copy of this license, visit
# <https://creativecommons.org/licenses/by-sa/4.0>.
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


This project contains a small image library. See mimage.h for the
image structure definition, but to summarize an image is an array
of pixels in memory, where a pixel is ARGB stored in an unsigned int.

The library supports reading and writing to PNG and JPEG file formats.
There is no guarantee that all PNG and all JPEG files are supported,
only the ones that have been tested. See examples subdirectory for
test images.

The library depends on libpng and jpeg, see INSTALL.txt for instructions.

Several test programs are provided, including a mandelbrot example and
programs to read and write PNG and JPEG images.



