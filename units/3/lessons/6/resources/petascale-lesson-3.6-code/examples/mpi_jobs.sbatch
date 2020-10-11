#!/bin/bash
# Blue Waters Petascale Semester Curriculum v1.0
# Unit 3: Using a Cluster
# Lesson 6: Running Code on a Cluster 2
# File: mpi_jobs.sbatch
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

### This script will allocate a total of 64 cores, 16 MPI processes (2 node * 8 ntask-per-node = 16), If is parallelized with OpenMP
### then each process will spawn 4 threads, one per core. 
### By design, Cedar supports multiple simultaneous parallel jobs of up to 1024 broadwell cores (32 nodes) or 1536 skylake cores 
### (32 nodes) or 1536 cascade lake cores (32 nodes)in a fully non-blocking manner.
#SBATCH --account=def-mludin  		# account to charge the allocation to
#SBATCH --job-name=mpi_example		# simple core affeninty example
#SBATCH --constraint=[skylake|cascade] 	# select the type of nodes, for any: --constraint=[skylake|cascade]
#SBATCH --nodes=2			# number of nodes for the job
#SBATCH --ntasks=16			# number of MPI process 
#SBATCH --ntasks-per-node=8		# number of MPI process to run per node
#SBATCH --cpus-per-task=4		# number of cpus to user per task/MPI process, with openmp this should be larger
#SBATCH --mem-per-cpu=1024M		# amount of RAM per cpu core, default is megabytes, --mem=0 reserves all available memory
#SBATCH --time=00:00:05			# time in (HH:MM:SS), you could also do(DD-HH:MM), this will run for 30min

echo "Current working directory is: " pwd
time srun ./mpi_example.exe|sort 		# mpirun or mpiexec also works
