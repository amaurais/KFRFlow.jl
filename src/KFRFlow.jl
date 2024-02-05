module KFRFlow

using KernelFunctions 
using LinearAlgebra, Statistics 
using Distances: pairwise, Euclidean 
using Distributions: MvNormal, sample 

import ForwardDiff 

export generateDataKernels, KFRFlowEuler, KFRFlowI, EKI, SVGD, KFRParams, vKFRFlow!, dKFRFlow!, vEKI!, dEKI!, EKIParams, SVGDParams, vSVGD!

"""
	generateDataKernels(X; k=RationalQuadraticKernel(α=1/2), pKernels=1.0, bwSet="full")

Generate kernel functions centered at a subset of the columns of the ``d × N`` matrix `X`. 

`pKernels` is the fraction of the columns of `X` that will be randomly selected and used for kernel centers. `k` is a kernel on ℝᵈ available in KernelFunctions. The kernel bandwidth is set according to the median heuristic (Liu & Wang 2016). The switch `bwSet` controls whether all columns of `X` are used to compute the bandwidth (`bwSet="full"`) or just the subset used as the kernel centers. 

Returns a vector of kernel functions.
"""
function generateDataKernels(X::Matrix; k::Kernel=RationalQuadraticKernel(α=1/2), pKernels::Real=1.0, bwSet::String="full")

	Nx = size(X, 2)
	Nkernels = Int64(floor(pKernels*Nx)) 

	# decide on loci, subsampling if pKernels < 1.0  
	if Nkernels < Nx 
		zIdx = sample(1:Nx, Nkernels, replace=false) #? where did this come from? 
	else
		zIdx = 1:Nx  
	end 

	loci = [ X[:, idx] for idx in zIdx]
	Xloci = hcat(loci...) 

	
	# Set of particles used to compute bandwidth. 
	if bwSet == "full" # use the entire ensemble 
		Xbw = X 
		Nbw = Nx 
	else # use only particles currently active as kernel loci 
		Xbw = Xloci 
		Nbw = Nkernels 
	end

	# median heuristic from Liu & Wang 2016
	dMat = pairwise(Euclidean(), eachcol(Xbw); symmetric=true)
	dists = [dMat[i,j] for i=1:Nbw for j=i+1:Nbw][:]  
	med = median(dists) 
	bw = 0.5*sqrt(med^2 / log(Nbw))		

	# generate a kernel function at each locus 
	features = [ x -> (k ∘ ScaleTransform(1/bw))(x, z) for z in loci ]
	return features[:] 
end


"""
	KFRFlowEuler(Xprior, logratio; <keyword arguments>)

Apply a forward Euler discretization of KFRFlow (Maurais & Marzouk 2024) to the columns of `Xprior` according to log density ratio `logratio`. 

# Arguments 
- `Xprior::Matrix`: d × N matrix containing N samples from a distribution π₀ on ℝᵈ 
- `logratio::Function`: function to compute the logarithm of the density ratio π₁/π₀, where π₁ is the desired endidng distribution. 
- `dt::Real=0.1`: timestep for the iteration 
- `savehistory::Bool=false`: whether to save the history of the evolution of `X` or only report the samples at t = 1. 
- `pKernels::Real=1.0`: fraction of the samples to use to define kernel features on each iteration. 
- `k::Kernel=RationalQuadraticKernel(α=1/2)` kernel function to use in the algorithm. Defaults to inverse multiquadric. 
- `bwSet::String="full"`: switch controlling whether all samples are used to compute the bandwidth at each step (`bwSet="full"`) or just the subset used as the kernel centers.
- `verbose::Bool=false`: whether to print a status update after each step. 
- `nugget::Real=0.0`: level of inflation for Mₜ
- `ϵ::Real=0.0`: level of stochasticity. Note that ϵ > 0 requires ∇logπ0 and ∇logπ1 to be input 
- `∇logπ0::Function=x->zeros(size(Xprior, 1))` score of π₀. Note that the default value is *not* physically meaningful. 
- `∇logπ1::Function=x->zeros(size(Xprior, 1))` score of π₁. Note that the default value is *not* physically meaningful. 

If `savehistory` is `true`, a d×N×(number of steps) matrix containing the intermediate states of `X` during the flow and a (number of steps) vector of time waypoints is returned. If not, only a d×N matrix containing the samples at t = 1 is returned.
"""
function KFRFlowEuler(Xprior::Matrix, logratio::Function; dt::Real=0.1, savehistory::Bool=false, pKernels::Real=1.0, k::Kernel=RationalQuadraticKernel(α=1/2), bwSet::String="full", verbose::Bool=false, nugget::Real=0.0, ϵ::Real=0.0, ∇logπ0::Function=x->zeros(size(Xprior, 1)), ∇logπ1::Function=x->zeros(size(Xprior, 1)))

	# dimensions 
	Nx = size(Xprior, 1) 
	Ne = size(Xprior, 2) 

	# initialize the ensemble container 
	X = Xprior  

	if savehistory 
		Xhist = Xprior 
		tArr = [0.0] 
	end

	tCurrent = 0.0 

	while tCurrent < 1.0 

		#* generate kernel at each ensemble member  
		features = generateDataKernels(X, pKernels=pKernels, k=k, bwSet=bwSet)  
		fgrads = [x -> ForwardDiff.gradient(f, x) for f in features]

		F(x) =[f(x) for f in features] 
		A(x) = vcat([g(x)' for g in fgrads]...) 

		Fevals = hcat(F.(eachcol(X))...) # feature evaluations at each particle, J x J 
		logRatioEvals = hcat(logratio.(eachcol(X))...)[:] # density ratio evaluations at each particle, Jx1 

		meanLRE = mean(logRatioEvals)  
		weights = (logRatioEvals .- meanLRE) # Jx1 
		

		# precompute gradients 
		Aarr = [A(X[:, i]) for i = 1:Ne] # gradient evaluations at each particle, each is Jxd  
		gradH0 =  Symmetric(sum( [ Aarr[i]*Aarr[i]' for i=1:Ne] )) # this is Mt, JxJ  

		# precompute update 
		dX = zeros(Nx, Ne)
		rhs = Fevals * weights 

		perturbation = (gradH0 + nugget*I) \ rhs

		for i = 1:Ne 
			# compute the update for particle i 
			#* updated to include stochasticity 
			dX[:, i] =  Aarr[i]' * perturbation + ϵ*( (1 - tCurrent)*∇logπ0(X[:, i]) + tCurrent*∇logπ1(X[:, i]) ) 
		end

		# Euler Maruyama 
		Δt = min(dt, 1-tCurrent)
		X = X + Δt*dX  +  sqrt(2*ϵ)*randn(Nx, Ne)*sqrt(Δt) 

		if savehistory 
			Xhist = cat(Xhist, X, dims=3)
			tArr = [tArr; tArr[end] + Δt] 
		end

		tCurrent = tCurrent + Δt  

		if verbose 
			println("finished step with dt=$(Δt). Current time is $tCurrent.")
		end
	end

	if savehistory 
		return (Xhist, tArr)  
	else
		return X 
	end
end

"""
	KFRParams(<keyword arguments>)

Struct for holding the parameters required for using DifferentialEquations.jl to implement KFRFlow. 

# Arguments 
- `logratio::Function`: function to compute the logarithm of the density ratio π₁/π₀, where π₁ is the desired endidng distribution. Required.  
- `d::Integer`: dimension of the problem. Required. 
- `pKernels::Real=1.0`: fraction of the samples to use to define kernel features on each iteration. 
- `k::Kernel=RationalQuadraticKernel(α=1/2)` kernel function to use in the algorithm. Defaults to inverse multiquadric. 
- `bwSet::String="full"`: switch controlling whether all samples are used to compute the bandwidth at each step (`bwSet="full"`) or just the subset used as the kernel centers.
- `verbose::Bool=false`: whether to print a status update after each step. 
- `nugget::Real=0.0`: level of inflation for Mₜ
- `ϵ::Real=0.0`: level of stochasticity. Note that ϵ > 0 requires ∇logπ0 and ∇logπ1 to be input 
- `∇logπ0::Function=x->zeros(size(Xprior, 1))` score of π₀. Note that the default value is *not* physically meaningful. 
- `∇logπ1::Function=x->zeros(size(Xprior, 1))` score of π₁. Note that the default value is *not* physically meaningful. 
"""
@kwdef struct KFRParams
	logratio::Function 
	d::Integer  
	pKernels::Real = 1.0 
	k::Kernel = RationalQuadraticKernel(α=1/2)
	bwSet::String = "full" 
	nugget::Real = 0.0 
	ϵ::Real = 0.0 
	∇logπ0::Function = x-> zeros(d)  
	∇logπ1::Function = x-> zeros(d) 
end

"""
	vKFRFlow!(dX, X, p, t)

Compute the velocity for KFRFlow in-place.  `X` is a d x N matrix containing `N` samples (samples are stored in columns), `dX` is a d x N matrix container for the velocity, `p` is an instance of `KFRParams`, and `t` is a time. 
"""
function vKFRFlow!(dX::Matrix, X::Matrix, p::KFRParams, t::Real)
	features = generateDataKernels(X, pKernels=p.pKernels, k=p.k, bwSet=p.bwSet)  
	fgrads = [x -> ForwardDiff.gradient(f, x) for f in features]

	F(x) =[f(x) for f in features] 
	A(x) = vcat([g(x)' for g in fgrads]...) 

	Fevals = hcat(F.(eachcol(X))...) # feature evaluations at each particle, J x J 
	logRatioEvals = hcat(p.logratio.(eachcol(X))...)[:] # density ratio evaluations at each particle, Jx1 

	meanLRE = mean(logRatioEvals)  
	weights = (logRatioEvals .- meanLRE) # Jx1 
	
	# precompute gradients 
	Ne = size(X, 2) 
	Aarr = [A(X[:, i]) for i = 1:Ne] # gradient evaluations at each particle, each is Jxd  
	gradH0 =  Symmetric(sum( [ Aarr[i]*Aarr[i]' for i=1:Ne] )) # this is Mt, JxJ  

	# precompute update 
	rhs = Fevals * weights 

	perturbation = (gradH0 + p.nugget*I) \ rhs

	for i = 1:Ne 
		# compute the update for particle i 
		dX[:, i] =  Aarr[i]' * perturbation  + p.ϵ*( (1 - t)*p.∇logπ0(X[:, i]) + t*p.∇logπ1(X[:, i]))
	end
end

"""
	dKFRFlow!(dX, X, p, t)

Compute the diffusion coefficient for KFRFlow in-place.  `X` is a d x N matrix containing `N` samples (samples are stored in columns), `dX` is a d x N matrix container for the velocity, `p` is an instance of `KFRParams`, and `t` is a time. 
"""
function dKFRFlow!(dX::Matrix, X::Matrix, p::KFRParams, t::Real)
	dX[:, :] = sqrt(2*p.ϵ)*ones(size(dX))
end 

"""
	KFRFlowI(Xprior, logratio; <keyword arguments>)

Apply KFRFlow-I (Maurais & Marzouk 2024) to the columns of `Xprior` according to log density ratio `logratio`. 

# Arguments 
- `Xprior::Matrix`: d × N matrix containing N samples from a distribution π₀ on ℝᵈ 
- `logratio::Function`: function to compute the logarithm of the density ratio π₁/π₀, where π₁ is the desired endidng distribution. 
- `dt::Real=0.1`: timestep for the iteration 
- `savehistory::Bool=false`: whether to save the history of the evolution of `X` or only report the samples at t = 1. 
- `pKernels::Real=1.0`: fraction of the samples to use to define kernel features on each iteration. 
- `k::Kernel=RationalQuadraticKernel(α=1/2)` kernel function to use in the algorithm. Defaults to inverse multiquadric. 
- `bwSet::String="full"`: switch controlling whether all samples are used to compute the bandwidth at each step (`bwSet="full"`) or just the subset used as the kernel centers.
- `verbose::Bool=false`: whether to print a status update after each step. 
- `nugget::Real=0.0`: level of inflation for Mₜ.

If `savehistory` is `true`, a d×N×(number of steps) matrix containing the intermediate states of `X` during the flow and a (number of steps) vector of time waypoints is returned. If not, only a d×N matrix containing the samples at t = 1 is returned.
"""
function KFRFlowI(Xprior::Matrix, logratio::Function; dt::Real=0.1, savehistory::Bool=false, pKernels::Real=1.0, k::Kernel=RationalQuadraticKernel(α=1/2), bwSet::String="full", verbose::Bool=false, nugget::Real=0.0 )

	# dimensions 
	Ne = size(Xprior, 2) 
	Nx = size(Xprior, 1) 

	# initialize the ensemble container 
	X = Xprior  

	Nsteps = Int(ceil(1/dt))

	if savehistory 
		# if old
		# 	Xhist = Xprior 
		# 	tArr = [0.0] 
		#else
			Xhist = zeros(Nx, Ne, Nsteps + 1)
			Xhist[:, :, 1] = Xprior[:, :] 
			tArr = zeros(Nsteps+1)  
		#end
	end

	tCurrent = 0.0 
	logIdx = 2 

	while tCurrent < 1.0 
		features = generateDataKernels(X, pKernels=pKernels, k=k, bwSet=bwSet)
		fgrads = [x -> ForwardDiff.gradient(f, x) for f in features]
	
		# evaluate features 
		F(x) =[f(x) for f in features] 
		A(x) = vcat([g(x)' for g in fgrads]...)  


		Fevals = hcat(F.(eachcol(X))...) # feature evaluations at each particle, JxJ 

		# importance weights 
		Δt = min(dt, 1-tCurrent)
		W = exp.(Δt * hcat(logratio.(eachcol(X))...))[:] 
		w = W/sum(W) 

		a = sum(Fevals, dims=2)/Ne 

		# precompute gradients 
		Aarr = [A(X[:, i]) for i = 1:Ne]
		gradH0 = 1/Ne * Symmetric(sum( [ Aarr[i]*Aarr[i]' for i=1:Ne] ) )

		# target feature means 
		b = Fevals*w 

		# map coefficients 
		sopt = - ((gradH0 + nugget*I)\(a - b)) 

		# update samples   
		X = X + hcat( [(sopt'*Amat)[:] for Amat in Aarr]... )

		if savehistory 
			# if old 
			# 	Xhist = cat(Xhist, X, dims=3)
			# 	tArr = [tArr; tArr[end] + Δt] 
			# else
				Xhist[:, :, logIdx] = X[:, :] 
				tArr[logIdx] = tArr[logIdx-1] + Δt 
			#end 
		end

		tCurrent = tCurrent + Δt
		logIdx = logIdx + 1 

		if verbose 
			println("finished step with dt=$(Δt). Current time is $tCurrent.")
		end
	end

	if savehistory 
		return (Xhist, tArr)  
	else
		return X 
	end

end


"""
	EKI(Xprior, ystar, G; <keyword arguments>)

Apply Ensemble Kalman Inversion (Iglesias et al. 2013) to approximately sample the posterior distribution of X given y = G(x) + ϵ = `ystar`, where ``ϵ ∼ \\mathcal{N}(0, Γ) ``.

# Arguments 
- `Xprior::Matrix`: d×N matrix containing N samples from a prior distribution on ℝᵈ.
- `ystar::Vector`: m-dimensional observation.  
- `G::Function`: Forward observation operator mapping ℝᵈ to ℝᵐ 
- `dt::Real=0.1`: timestep for the iteration  
- `Γ::Matrix=Float64.(Matrix(I, size(ystar, 1), size(ystar, 1)))`: Observational covariance matrix of ϵ. Defaults to the identity. 
- `savehistory::Bool=false`: whether to save the history of the evolution of the samples or only report the samples at t = 1.
- `verbose::Bool=false`: whether to print a status update after each step. 

If `savehistory` is `true`, a d×N×(number of steps) matrix containing the intermediate states of the samples during the iteration and a (number of steps) vector of time waypoints is returned. If not, only a d×N matrix containing the samples at t = 1 is returned.
"""
function EKI(Xprior::Matrix, ystar::Vector, G::Function; dt::Real=0.1, Γ::Matrix=Float64.(Matrix(I, size(ystar, 1), size(ystar, 1))), savehistory::Bool=false, verbose::Bool=false)

	# dimensions 
	Ne = size(Xprior, 2) 

	# initialize the ensemble container 
	X = Xprior  

	if savehistory 
		Xhist = Xprior 
		tArr = [0.0] 
	end

	tCurrent = 0.0 

	while tCurrent < 1.0

		Δt = min(dt, 1-tCurrent) 

		# incremental likelihood 
		Γn = Γ*1/Δt 
		πϵ_N = MvNormal(Γn)
		
		# simulate noiseless forward process 
		GX = hcat(G.(eachcol(X))...)  

		# compute ensemble Kalman update 
		Cᵘᵖ = cov(X, GX, dims=2)
		Cᵖᵖ = cov(GX, dims=2)

		X = X + Cᵘᵖ*((Cᵖᵖ + Γn)\(ystar .+ rand(πϵ_N, Ne) - GX))

		if savehistory 
			Xhist = cat(Xhist, X, dims=3)
			tArr = [tArr; tArr[end] + Δt] 
		end

		tCurrent = tCurrent + Δt 

		if verbose 
			println("finished step with dt=$(min(dt, 1-tCurrent)). Current time is $tCurrent.")
		end
	end

	if savehistory 
		return (Xhist, tArr)  
	else
		return X 
	end
end

"""
	EKIParams(<keyword arguments>)

Struct for holding the parameters required for using DifferentialEquations.jl to implement Ensemble Kalman Inversion (Iglesias et al. 2013). 

# Arguments 
- `ystar::Vector`: m-dimensional observation. Required. 
- `G::Function`: Forward observation operator mapping ℝᵈ to ℝᵐ. Required. 
- `Γ::Matrix=Float64.(Matrix(I, size(ystar, 1), size(ystar, 1)))`: Observational covariance matrix of ϵ. Defaults to the identity.
- `Γhalf::Matrix=Float64.(Matrix(I, size(ystar, 1), size(ystar, 1)))`: A square root of Γ. Defaults to the identity  
"""
@kwdef struct EKIParams
	ystar::Vector
	G::Function 
	Γ::Matrix = Float64.(Matrix(I, size(ystar, 1), size(ystar, 1)))
	Γhalf::Matrix = cholesky(Γ) 
end

"""
	vEKI!(dX, X, p, t)

Compute the velocity for Ensemble Kalman Inversion (Iglesias et al. 2013) in-place.  `X` is a d x N matrix containing `N` samples (samples are stored in columns), `dX` is a d x N matrix container for the velocity, `p` is an instance of `EKIParams`, and `t` is a time. 
"""
function vEKI!(dX::Matrix, X::Matrix, p::EKIParams, t::Real)
	GX = hcat(p.G.(eachcol(X))...)  

	# compute ensemble Kalman update 
	Cᵘᵖ = cov(X, GX, dims=2)
	dX[:, :] = Cᵘᵖ*(p.Γ\(p.ystar .- GX))
end

"""
	dEKI!(dX, X, p, t)

Compute the diffusion coefficients for Ensemble Kalman Inversion (Iglesias et al. 2013) in-place.  `X` is a d x N matrix containing `N` samples (samples are stored in columns), `dX` is a d x N matrix container for the velocity, `p` is an instance of `EKIParams`, and `t` is a time. 
"""
function dEKI!(dX::Matrix, X::Matrix, p::EKIParams, t::Real)
	GX = hcat(p.G.(eachcol(X))...)
	Cᵘᵖ = cov(X, GX, dims=2)
	dX[:, :] = Cᵘᵖ/p.Γhalf
end

"""
	SVGD(Xprior, ∇logπ1; <keyword arguments>)

Use Stein Variational Gradient Descent (SVGD) (Liu & Wang 2016)to sample from a distribution π₁ beginning with the samples in `Xprior`. 

# Arguments 
- `Xprior::Matrix`: d×N matrix containing N samples from a starting distribution on ℝᵈ
- `∇logπ1::Function`: score of the target distribution  
- `dt::Real=0.1`: timestep for the iteration 
- `Tstop::Real=1.0`: stopping time of the iteration 
- `k::Kernel=RationalQuadraticKernel(α=1/2)`: choice of kernel, from KernelFunctions.jl. Defaults to inverse multiquadric. 
- `bwSet::String="full"`: switch controlling whether all samples are used to compute the bandwidth at each step (`bwSet="full"`) or just the subset used as the kernel centers. 
- `verbose::Bool=false`: whether to print a status update after each step. 
- `savehistory::Bool=false`: whether to save the history of the evolution of the samples or only report the samples at t = `Tstop`.

If `savehistory` is `true`, a d×N×(number of steps) matrix containing the intermediate states of the samples during the iteration and a (number of steps) vector of time waypoints is returned. If not, only a d×N matrix containing the samples at t = 1 is returned.
"""
function SVGD(Xprior::Matrix, ∇logπ1::Function; dt::Real=0.1, Tstop::Real=1.0, k::Kernel=RationalQuadraticKernel(α=1/2), bwSet::String="full", verbose::Bool=false, savehistory::Bool=false)

	# initialize the ensemble container 
	X = Xprior  

	if savehistory 
		Xhist = Xprior 
		tArr = [0.0] 
	end

	tCurrent = 0.0 

	while tCurrent < Tstop  

		#* generate kernel at each ensemble member  
		features = generateDataKernels(X, pKernels=1.0, k=k, bwSet=bwSet)  
		fgrads = [x -> ForwardDiff.gradient(f, x) for f in features]

		Nk = length(features) 

		F(x) =[f(x) for f in features] 
		A(x) = hcat([g(x) for g in fgrads]...) 

		Fevals = hcat(F.(eachcol(X))...) # kernel evaluations at each particle, Nk x J  
		Aarr = cat(A.(eachcol(X))..., dims=3) # gradient evaluations at each particle. d x J x Nk 

		potentials = hcat(∇logπ1.(eachcol(X))...) # d x Nk 

		# dX = zeros(Nx, Ne)

		# for i = 1:Ne 
		# 	dX[:, i] =  1/Nk * potentials*Fevals[:, i] + 1/Nk * sum(Aarr[:, i, :], dims=2)
		# end

		dX = dropdims(1/Nk * potentials*Fevals + 1/Nk * sum(Aarr, dims=3), dims=3)

		Δt = min(dt, Tstop-tCurrent)
		X = X + Δt*dX 

		if savehistory 
			Xhist = cat(Xhist, X, dims=3)
			tArr = [tArr; tArr[end] + Δt] 
		end

		tCurrent = tCurrent + Δt   

		if verbose 
			println("finished step with dt=$(Δt). Current time is $tCurrent.")
		end
	end

	if savehistory 
		return (Xhist, tArr)  
	else
		return X 
	end
end

"""
	SVGDParams(<keyword arguments>)

Struct for holding the parameters required for using DifferentialEquations.jl to implement SVGD (Liu & Wang 2016). 

# Arguments 
- `∇logπ1::Function`: score of the target distribution  
- `k::Kernel=RationalQuadraticKernel(α=1/2)`: choice of kernel, from KernelFunctions.jl. Defaults to inverse multiquadric. 
"""
@kwdef struct SVGDParams 
	∇logπ1::Function
	k::Kernel=RationalQuadraticKernel(α=1/2) 
end

"""
	vSVGD!(dX, X, p, t)

Compute the velocity for SVGD (Iglesias et al. 2013) in-place.  `X` is a d x N matrix containing `N` samples (samples are stored in columns), `dX` is a d x N matrix container for the velocity, `p` is an instance of `SVGDParams`, and `t` is a time. 
"""
function vSVGD!(dX::Matrix, X::Matrix, p::SVGDParams, t::Real)
	features = generateDataKernels(X, pKernels=1.0, k=p.k, bwSet="full")  
	fgrads = [x -> ForwardDiff.gradient(f, x) for f in features]

	Nk = length(features) 

	F(x) =[f(x) for f in features] 
	A(x) = hcat([g(x) for g in fgrads]...) 

	Fevals = hcat(F.(eachcol(X))...) # kernel evaluations at each particle, Nk x J  
	Aarr = cat(A.(eachcol(X))..., dims=3) # gradient evaluations at each particle. d x J x Nk 

	potentials = hcat(p.∇logπ1.(eachcol(X))...) # d x Nk 

	dX[:, :] = dropdims(1/Nk * potentials*Fevals + 1/Nk * sum(Aarr, dims=3), dims=3)
end

end # module KFRFlow