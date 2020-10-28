#!/bin/bash
# Blue Waters Petascale Semester Curriculum v1.0
# Unit 10: Productivity and Visualization
# Lesson 4: Visualization 2
# File: runbatch_mpi.sh
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


TIME=01:00:00
HOSTS=2
PROCSPERNODE=1
export IBRUN_TASKS_PER_NODE=1
BINDIR=/work/06295/fabiogon/stampede2/VMD_MPI/vmd
RUNDIR=`pwd`

sbatch --job-name=render_mpi --nodes=${HOSTS} --ntasks=$((${HOSTS}*${PROCSPERNODE})) --ntasks-per-node=$PROCSPERNODE --time=${TIME} --partition=normal -o slurm%j.out -e slurm%j.err << ENDINPUT
#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/06295/fabiogon/stampede2/netcdf-c-4.7.4/build/lib
export myfile=$1

# Go to working directory (submission directory)
cd $RUNDIR
echo "Current working directory is: " `pwd`
# Execute job
time ibrun $BINDIR/vmd_mpi -e \${myfile} 

ENDINPUT
