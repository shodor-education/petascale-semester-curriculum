#!/bin/bash
# Blue Waters Petascale Semester Curriculum v1.0
# Unit 9: Optimization
# Lesson 1: Cache Efficient Matrix Multiplication
# File: runMatMult.sh
# Developed by Paul F. Hemler for the Shodor Education Foundation, Inc.
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


# BASH shell script to automate timing results for
# various size matrices and various number of threads

# make sure the program is current
make matMult

# Output file variable, in case I want to run the
# script again with a different name.  Change once
# use often!
OUTFILE=mmTranspose.txt

# Remove the output file if it exists because the
# loop below appends to the file.
if [ -f $OUTFILE ]; then
   rm -f $OUTFILE
fi

# Run the program for matrices for size 100, 200, ... 1500
# and variuos number of threads.  Go have a cup a coffee
# and let the machine do the work!!
COUNTER=100
while [ $COUNTER -le 1500 ]; do
    echo $COUNTER		#Progress Visualization
    echo $COUNTER >> $OUTFILE
    ./matMult $COUNTER >> $OUTFILE
    let COUNTER=COUNTER+100
    echo >> $OUTFILE
done
