using Distributed, ClusterManagers

ENV["JULIA_NUM_THREADS"] = ENV["SLURM_CPUS_PER_TASK"]
slurm_ntasks = parse(Int, ENV["SLURM_NTASKS"])
addprocs(SlurmManager(slurm_ntasks))

@everywhere begin 
    using IntervalSets
    using Distributions
    using Random, LinearAlgebra, Statistics, Distributions, StatsBase, ArraysOfArrays
    using JLD2
    using ValueShapes
    using TypedTables
    using Measurements
    using TypedTables
    using HypothesisTests
    using BAT
    using HDF5
    using CSV
end

@everywhere begin
    JLD2.@load "../../data/mixture-9D-nc.jld" means cov_m n_clusters
    
    mixture_model = MixtureModel(MvNormal[MvNormal(means[i,:], Matrix(Hermitian(cov_m[i,:,:])) ) for i in 1:n_clusters]);
    
    likelihood = let model = mixture_model
        params -> begin
             return LogDVal(logpdf(model, params.a))
        end
    end
    prior = NamedTupleDist(a = [Uniform(-100,100) for i in 1:size(means)[2]])
    posterior = PosteriorDensity(likelihood, prior);
    log_volume = BAT.log_volume(BAT.spatialvolume(posterior.parbounds))
end

SAVE_PATH = "../../data/Benchmark-1/"
IND = "1"

sampler = MetropolisHastings() # AHMC()

burnin_1 = MCMCBurninStrategy(
        max_nsamples_per_cycle = 10000,
        max_nsteps_per_cycle = 10000,
        max_time_per_cycle = 25,
        max_ncycles = 40
    )

tuning = AdaptiveMetropolisTuning(
    λ = 0.5,
    α = 0.05..0.15,
    β = 1.5,
    c = 1e-4..1e2,
    r = 0.5
)

sampling_kwargs = (burnin = burnin_1, tuning=tuning)
 
exploration_sampler = MetropolisHastings()

burnin_2 = MCMCBurninStrategy(
        max_nsamples_per_cycle = 6000,
        max_nsteps_per_cycle = 6000,
        max_time_per_cycle = 25,
        max_ncycles = 20
    )

exploration_kwargs = (burnin = burnin_2,)
n_exploration = (10^2, 40)

partitioner = KDTreePartitioning( partition_dims = :auto, extend_bounds=true)

integrator = AHMIntegration(  
        whitening= CholeskyPartialWhitening(), #CholeskyPartialWhitening(),
        autocorlen= GeyerAutocorLen(),
        volumetype = :HyperRectangle,
        max_startingIDs = 10000,
        max_startingIDs_fraction = 2.5,
        rect_increase = 0.1,
        warning_minstartingids = 16,
        dotrimming = true,
        uncertainty= [:cov]
    )

algorithm = PartitionedSampling(
        sampler = sampler,
        exploration_sampler = exploration_sampler,
        partitioner = partitioner,
        integrator = integrator,
        exploration_kwargs = exploration_kwargs,
        sampling_kwargs = sampling_kwargs,
        n_exploration = n_exploration
    )

n_chains = 10 
n_samples = 10^5
n_subspaces = 15

try
    output_sp_ms = bat_sample(posterior, (n_samples, n_chains, n_subspaces), algorithm);
    samples = output_sp_ms.result
    posterior_integral = -log(sum(output_sp_ms.info.density_integral))
    
    @show "Saving Samples:"
    BAT.bat_write(SAVE_PATH*"samples"*IND*".hdf5", unshaped.(samples))
    CSV.write(SAVE_PATH*"table"*IND*".csv", output_sp_ms.info)
    
finally
   rmprocs.(workers())
end

@info "Done."