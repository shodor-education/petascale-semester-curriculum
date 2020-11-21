# Blue Waters Petascale Semester Curriculum v1.0
# Unit 4: OpenMP
# Lesson 6: When Should You Use OpenMP?
# File: pandemic/Makefile
# Developed by Aaron Weeden for the Shodor Education Foundation, Inc.
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

# Code prefix
PROGRAM_PREFIX=pandemic

# Compilers and flags
CC=gcc
MPICC=mpicc
#CFLAGS+=-DTEXT_DISPLAY # Uncomment to show text display
#CFLAGS+=-DX_DISPLAY -L/usr/X11R6/lib -lX11 -lm # Uncomment to show X display
CFLAGS+=-DSHOW_RESULTS # Uncomment to make the program print its results

# Source files
SRCS=$(PROGRAM_PREFIX).c

# OpenMP
OPENMP_FLAGS=-fopenmp

# MPI
MPI_FLAGS=-DMPI

# Hybrid MPI/OpenMP
#HYBRID_FLAGS=$(MPI_FLAGS) $(OPENMP_FLAGS)
HYBRID_FLAGS=$(OPENMP_FLAGS) $(MPI_FLAGS)

EXECUTABLES=$(PROGRAM_PREFIX).{serial,openmp,mpi,hybrid}

# Make rules
$(PROGRAM_PREFIX).serial: $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(PROGRAM_PREFIX).serial

$(PROGRAM_PREFIX).openmp: $(SRCS)
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) $(SRCS) -o $(PROGRAM_PREFIX).openmp

$(PROGRAM_PREFIX).mpi: $(SRCS)
	$(MPICC) $(CFLAGS) $(MPI_FLAGS) $(SRCS) -o $(PROGRAM_PREFIX).mpi


$(PROGRAM_PREFIX).hybrid: $(SRCS)
	$(MPICC) $(CFLAGS) $(HYBRID_FLAGS) $(SRCS) -o $(PROGRAM_PREFIX).hybrid

clean:
	rm -f $(EXECUTABLES) *.o

serial:
	make $(PROGRAM_PREFIX).serial

openmp:
	make $(PROGRAM_PREFIX).openmp

mpi:
	make $(PROGRAM_PREFIX).mpi

hybrid:
	make $(PROGRAM_PREFIX).hybrid

all:
	make clean
	make $(EXECUTABLES)
