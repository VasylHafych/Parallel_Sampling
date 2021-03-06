using Distributed 

slurm_ntasks = 5
Distributed.addprocs(slurm_ntasks)

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

SAVE_PATH = "../../data/Benchmark-3/"

sampler = MetropolisHastings() # AHMC()

burnin_1 = MCMCBurninStrategy(
        max_nsamples_per_cycle = 10000,
        max_nsteps_per_cycle = 10000,
        max_time_per_cycle = 25,
        max_ncycles = 50
    )

tuning = AdaptiveMetropolisTuning(
    λ = 0.5,
    α = 0.05..0.15,
    β = 1.5,
    c = 1e-4..1e2,
    r = 0.5
)

init = MCMCInitStrategy(
    init_tries_per_chain = 8..128,
    max_nsamples_init = 25,
    max_nsteps_init = 250,
    max_time_init = 25
)

max_nsteps = 10^10
 
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

n_chains = 10 
n_samples = 10^6
n_subspaces = slurm_ntasks

try 
    # Test run to precompile BAT on workers
    
    max_time = 5
    sampling_kwargs = (burnin = burnin_1, tuning=tuning, max_nsteps=max_nsteps, max_time=max_time, init=init)
    algorithm = PartitionedSampling(
            sampler = sampler,
            exploration_sampler = exploration_sampler,
            partitioner = partitioner,
            integrator = integrator,
            exploration_kwargs = exploration_kwargs,
            sampling_kwargs = sampling_kwargs,
            n_exploration = n_exploration
        )
    
    test_run = bat_sample(posterior, (10^3, 2, slurm_ntasks), algorithm);
catch 
    @show "Precompilation Run Failed"
end

try
    
    times = range(3, stop=15, length=3)
    n_repetitions = 2
    
    for time_ind in Base.OneTo(length(times))

        max_time = times[time_ind]
        sampling_kwargs = (burnin = burnin_1, tuning=tuning, max_nsteps=max_nsteps, max_time=max_time, init=init)
        algorithm = PartitionedSampling(
            sampler = sampler,
            exploration_sampler = exploration_sampler,
            partitioner = partitioner,
            integrator = integrator,
            exploration_kwargs = exploration_kwargs,
            sampling_kwargs = sampling_kwargs,
            n_exploration = n_exploration
        )
        
        for rep_ind in Base.OneTo(n_repetitions)
            
            file_name_ = "$slurm_ntasks-$time_ind-$rep_ind"
            
            output_sp_ms = bat_sample(posterior, (n_samples, n_chains, n_subspaces), algorithm);
            samples = output_sp_ms.result
            posterior_integral = -log(sum(output_sp_ms.info.density_integral))
            @show "Saving Samples:"
            BAT.bat_write(SAVE_PATH*file_name_*".hdf5", unshaped.(samples))
            CSV.write(SAVE_PATH*file_name_*".csv", output_sp_ms.info)
            
        end
    end


    
finally
   rmprocs.(workers())
end

@info "Done."