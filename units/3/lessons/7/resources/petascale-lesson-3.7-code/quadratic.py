#!/usr/bin/env python
# Blue Waters Petascale Semester Curriculum v1.0
# Unit 3: Using a Cluster
# Lesson 7: Scaling on a Cluster 1
# File: quadratic.py
# Developed by Linh B. Ngo for the Shodor Education Foundation, Inc.
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

## quadratic.py is a simple Python script that, given a b, and c,
## solves the two roots of ax^2+bx+c.

import sys, math

a = float(sys.argv[1])
b = float(sys.argv[2])
c = float(sys.argv[3])
d = math.sqrt(b**2 - 4. * a * c)
print ('x1 = {}'.format((-b + d) / (2. * a)))
print ('x2 = {}'.format((-b - d) / (2. * a)))
