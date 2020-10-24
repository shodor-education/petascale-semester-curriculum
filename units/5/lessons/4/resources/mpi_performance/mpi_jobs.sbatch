#!/bin/bash
### This script will allocate a total of 64 cores, 16 MPI processes (2 node * 8 ntask-per-node = 16), If is parallelized with OpenMP
### then each process will spawn 4 threads, one per core. 
### By design, Cedar supports multiple simultaneous parallel jobs of up to 1024 broadwell cores (32 nodes) or 1536 skylake cores 
### (32 nodes) or 1536 cascade lake cores (32 nodes)in a fully non-blocking manner.
#SBATCH --account=def-mludin  		# account to charge the allocation to
#SBATCH --job-name=GalaxSee-v2-pppm	# simple core affeninty example
#SBATCH --output=pppm.out		# output file for job
#SBATCH --constraint=[skylake|cascade] 	# select the type of nodes, for any: --constraint=[skylake|cascade]
#SBATCH --nodes=2			# number of nodes for the job
#SBATCH --ntasks=16			# number of MPI process 
#SBATCH --ntasks-per-node=8		# number of MPI process to run per node
#SBATCH --cpus-per-task=4		# number of cpus to user per task/MPI process, with openmp this should be larger
#SBATCH --mem-per-cpu=1024M		# amount of RAM per cpu core, default is megabytes, --mem=0 reserves all available memory
#SBATCH --time=00:00:30			# time in (HH:MM:SS), you could also do(DD-HH:MM), this will run for 30min

echo "Current working directory is: `pwd`"
time srun GalaxSee-v2/Galaxsee-v2.c-mpi pppm.gal		# mpirun or mpiexec also works
