#!/bin/bash
# Blue Waters Petascale Semester Curriculum v1.0
# Unit 3: Using a Cluster
# Lesson 5: Running Code on a Cluster 1
# File: runbatch.sh
# Developed by Juan R. Perilla for the Shodor Education Foundation, Inc.
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

################################################################################
# This script will allocate a total of 64 cores, 16 MPI processes (2 node * 8
# ntask-per-node = 16), If is parallelized with OpenMP then each process will
# spawn 4 threads, one per core.
################################################################################



export mypath=`pwd`
sbatch --job-name=mpi_example --nodes=2	--ntasks=16 --ntasks-per-node=8 --cpus-per-task=4 --time=00:30:00 --partition=normal -o slurm%j.out -e slurm%j.err <<ENDINPUT
#!/bin/bash

module load impi/18.0.2
export OMP_NUM_THREADS=8

# Set executable fille
export myfile=$1
# Go to working directory (submission directory)
cd $mypath
echo "Current working directory is: " `pwd`
# Execute job
time ibrun \${myfile}
ENDINPUT
