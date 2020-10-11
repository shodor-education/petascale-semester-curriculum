# Blue Waters Petascale Semester Curriculum v1.0
# Unit 3: Using a Cluster
# Lesson 6: Running Code on a Cluster 2
# File: Makefile
# Developed by Mobeen Ludin for the Shodor Education Foundation, Inc.
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

# Makefile
#------- Select your Compier
#COMPTYPE="Cray"
#COMPTYPE="Intel"
#COMPTYPE="GNU"
COMPTYPE="PGI"

#------ Select known target machine
#SYSTYPE="Earlham"
#SYSTYPE="Stampede"
#SYSTYPE="BW"
#SYSTYPE="Mac"
#SYSTYPE="Linux"
SYSTYPE="Cedar"

ifeq ($(SYSTYPE), "Earlham")
CC          = pgcc
MPICC       = mpicc
OMPFLAGS    = -openmp
ACCFLAGS    = -acc -Minfo
LIBS        = -lm
endif

#compiling mpi and openmpi hybrid program with intel:  mpiicc -qopenmp

ifeq ($(SYSTYPE), "Cedar")
    ifeq ($(COMPTYPE), "Intel")
    CC          = icc
    ACCFLAGS    = -h pragma=acc,msgs
    MPICC       = mpicc
    OMPFLAGS    = -qopenmp
    LIBS        = -lm
    else ifeq ($(COMPTYPE), "PGI")
    CC          = pgcc
    MPICC       = mpicc 
    ACCFLAGS    = -acc -Minfo
    OMPFLAGS    = -mp
    LIBS        = -lm
    else ifeq ($(COMPTYPE), "GNU")
    CC          = gcc
    MPICC       = mpicc 
    ACCFLAGS    = -fopenaccc
    OMPFLAGS    = -fopenmp
    LIBS        = -lm
    endif
endif

ifeq ($(SYSTYPE), "BW")
    ifeq ($(COMPTYPE), "Cray")
    CC          = cc
    ACCFLAGS    = -h pragma=acc,msgs
    MPICC       = cc
    OMPFLAGS    = -fopenmp
    LIBS        = -lm
    else ifeq ($(COMPTYPE), "PGI")
    CC          = pgcc
    MPICC       = mpicc 
    ACCFLAGS    = -acc -Minfo
    OMPFLAGS    = -mp=nonuma
    LIBS        = -lm
    else ifeq ($(COMPTYPE), "GNU")
    CC          = gcc
    MPICC       = mpicc 
    ACCFLAGS    = -fopenaccc
    OMPFLAGS    = -fopenmp
    LIBS        = -lm
    endif
endif

mpi_pi_area:
	$(MPICC) -o mpi_pi_area.exe mpi_pi_area.c   
acc_laplace:
	$(CC) $(ACCFLAGS) -o acc_laplace.exe acc_laplace.c
omp_pi_area:
	$(CC) $(OMPFLAGS) -o omp_pi_area.exe omp_pi_area.c
mpi_example:
	$(MPICC) -o mpi_example.exe mpi_example.c   

all:
	make clean
	make mpi_pi_area acc_laplace omp_pi_area mpi_example
clean:
	rm -rf *.exe *.ptx *.cub
