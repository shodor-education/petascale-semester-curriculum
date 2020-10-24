# Blue Waters Petascale Semester Curriculum v1.0
# Unit 5: MPI
# Lesson 2: Collective vs. Point-to-Point Communication
# Makefile
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
SYSTYPE="BW"
#SYSTYPE="Mac"
#SYSTYPE="Linux"

ifeq ($(SYSTYPE), "Earlham")
CC			= pgcc
MPICC		= mpicc
OMPFLAGS	= -openmp
ACCFLAGS	= -acc -Minfo
LIBS		= -lm
endif

ifeq ($(SYSTYPE), "BW")
    ifeq ($(COMPTYPE), "Cray")
    CC			= cc
    ACCFLAGS    = -h pragma=acc,msgs
    MPICC		= cc
    OMPFLAGS	= -fopenmp
    LIBS		= -lm
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

mpi_allgather:
	$(MPICC) -o mpi_allgather.exe mpi_allgather.c
mpi_allgatherv:
	$(MPICC) -o mpi_allgatherv.exe mpi_allgatherv.c
mpi_allreduce:
	$(MPICC) -o mpi_allreduce.exe mpi_allreduce.c
mpi_bcast:
	$(MPICC) -o mpi_bcast.exe mpi_bcast.c
mpi_gather:
	$(MPICC) -o mpi_gather.exe mpi_gather.c
mpi_gatherv:
	$(MPICC) -o mpi_gatherv.exe mpi_gatherv.c
mpi_pi_area:
	$(MPICC) -o mpi_pi_area.exe mpi_pi_area.c
mpi_reduce:
	$(MPICC) -o mpi_reduce.exe mpi_reduce.c   
mpi_scatter:
	$(MPICC) -o mpi_scatter.exe mpi_scatter.c
mpi_scatterv:
	$(MPICC) -o mpi_scatterv.exe mpi_scatterv.c
all:
	make clean
	make mpi_reduce mpi_scatter mpi_gather mpi_bcast mpi_allreduce mpi_allgather mpi_pi_area
clean:
	rm -rf *.exe *.out
