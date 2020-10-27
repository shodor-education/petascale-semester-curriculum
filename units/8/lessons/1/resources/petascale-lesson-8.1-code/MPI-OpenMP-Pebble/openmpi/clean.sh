#!/bin/bash
# Blue Waters Petascale Semester Curriculum v1.0
# File: MPI-OpenMP-Pebble/openmpi/clean.sh
# Developed by R. Phillip Bording for the Shodor Education Foundation, Inc.
# Included in the following lessons:
# - Unit 5 (MPI) Lesson 10: Wave Propagation in MPI
# - Unit 6 (Hybrid MPI + OpenMP) Lesson 3: Pebble in Pond Wave Equation
# - Unit 8 (OpenACC) Lesson 1: Accelerating Scientific Applications
# - Unit 10 (Productivity and Visualization) Lesson 3: Visualization 1
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



rm -f data/*

rm -f temp_fort/*
