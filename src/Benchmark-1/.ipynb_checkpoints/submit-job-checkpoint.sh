#!/bin/bash -l

#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
#SBATCH -D ./
#SBATCH -J julia_dist

# Note: Full node and shared node are mutually exclusive! Uncomment either one or the other!

### Full node configuration:
##SBATCH --nodes=1
##SBATCH --tasks-per-node=1
##SBATCH --cpus-per-task=32  # 32 on DRACO, use 40 on COBRA
##SBATCH --partition=general  # necessary on DRACO

### Shared node configuration:
#SBATCH --partition=express
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=10
##SBATCH --mem=32000

#SBATCH --mail-type=none
#SBATCH --time=00:20:59

module load julia/1.4.2
module load anaconda/3/2019.03 

# each worker can use 1 thread
export JULIA_NUM_THREADS=10

# we spawn 1 master and N-1 workers from the master
julia run-mpcdf.jl

