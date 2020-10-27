#!/bin/bash
# Blue Waters Petascale Semester Curriculum v1.0
# Unit 9: Optimization
# Lesson 3: Cache Memory Efficiency
# File: runMemStrideT.sh
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
# various strides through a linear array

# make sure the program is made and ready to go
make memStrideT


# Output file variable, in case I want to run the
# script again with a different name.  Change once
# use often!
OUTFILE=memStrideUnsignedChar


# Remove the output file if it exists because the
# loop below appends to the file.
if [ -f $OUTFILE.csv ]; then
   rm -f $OUTFILE.csv
fi

#Run the program using stride of 1 to 132 with 100Mibe elements
COUNTER=1
while [ $COUNTER -le 133 ]; do
	echo $COUNTER		#Progress Visualization
	./memStrideT 100 $COUNTER >> $OUTFILE.csv
	let COUNTER=COUNTER+1
done


#Run the program using stride of 128,256,512,1024,248,4096 with 100Mib of memory
COUNTER=256
while [ $COUNTER -le 4096 ]; do
	echo $COUNTER
	./memStrideT 100 $COUNTER >> $OUTFILE.csv
	let COUNTER=COUNTER*2
done
