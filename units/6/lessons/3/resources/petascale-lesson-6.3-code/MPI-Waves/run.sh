#!/bin/bash 
# Blue Waters Petascale Semester Curriculum v1.0
# Unit 5: MPI
# Lesson 10: Wave Propagation in MPI
# File: MPI-Waves/run.sh
# Developed by R. Phillip Bording for the Shodor Education Foundation, Inc.
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

rm -f data/pebble.mvlp
rm -f data/pebble_mpi


f77 pebble_mpi.f -o pebble_mpi -O0

./pebble_mpi

xmovie  <pebble.mvlp  n1=512  n2=512 clip=0.03  height=512 width=512 loop=1 title=" Pond Wave Frame %g"