using KFRFlow   
using Distributions 
using Random 
using KernelFunctions
using LinearAlgebra
using ForwardDiff 
using DifferentialEquations 

## plotting dependencies  
using CairoMakie 
bigSize = 30 
smallSize = 24
fontsize_theme = Theme(Axis=(titlesize=bigSize, xlabelsize=bigSize, ylabelsize=bigSize, xticklabelsize=smallSize, yticklabelsize=smallSize, ylabelfont=:bold, xlabelfont=:bold, xticklabelfont=:bold, yticklabelfont=:bold, titlefont=:bold), Legend=(labelfont=:bold, labelsize=smallSize, margin=(10, 20, 10, 10)))
set_theme!(fontsize_theme)
ms = 8

## * set up problem   

Ne = 300   # Number of samples 
Nx = 2     # Dimension of the state


# standard normal reference  
σ0 = 1.0 
μ0 = [0.0; 0.0]
Σ0 = σ0*[1.0 0; 0 1.0]  
Γ0 = (1/σ0) * [1.0 0; 0 1.0] 

πx = MvNormal(Σ0)
logπ0(x) = -0.5*(x - μ0)'*Γ0*(x - μ0)
π0(x) = exp(logπ0(x))

# choose example 
example = "spaceships" # "donut", "butterfly" 

if example == "donut" 
	G(x) = norm(x) 
	ystar = 2 

	σϵ = 0.25 

	logRatio(x) = -0.5*(G(x) - ystar)^2/(σϵ^2)
	densityRatio(x) = exp(logRatio(x))

	π1(x) = π0(x)*densityRatio(x) 
	logπ1(x) = logπ0(x) + logRatio(x)

	nugget = 0.0  
	importanceNugget = 1e-11

	kfrKernel = MaternKernel() 
	kfrBw = 45 

	dt = 0.01

	xmin = -3.0 
	xmax = 3.0 
	ymin = -3.0
	ymax = 3.0 

elseif example == "spaceships"
	G(x) = sin(x[1]*x[2]) + cos(x[1]*x[2]) 
	ystar = -1.0

	σϵ = 0.5 
	logRatio(x) = -0.5*(G(x) - ystar)^2/(σϵ^2)
	densityRatio(x) = exp(logRatio(x))

	π1(x) = π0(x)*densityRatio(x) 
	logπ1(x) = logπ0(x) + logRatio(x)

	dt = 0.01  

	nugget = 0.0  
	importanceNugget = 0.0  

	kfrKernel = MaternKernel() 
	kfrBw = 30 

	xmin = -3.0
	xmax = 3.0
	ymin = -3.0 
	ymax = 3.0

elseif example == "butterfly" 
	G(x) = sin(x[2]) + cos(x[1]) 
	ystar = -1.0 

	σϵ = 0.6
	logRatio(x) = -0.5*(G(x) - ystar)^2/(σϵ^2)
	densityRatio(x) = exp(logRatio(x))

	π1(x) = π0(x)*densityRatio(x) 
	logπ1(x) = logπ0(x) + logRatio(x) 

	dt = 0.01 

	# inflation levels 
	nugget = 0.0  
	importanceNugget = 0.0  

	kfrKernel = MaternKernel() 
	kfrBw = 15 

	xmin = -3.0 
	xmax = 3.0 
	ymin = -3.0
	ymax = 3.0 
else
	error("unknown example!")
end

∇logπ1(x) = ForwardDiff.gradient(logπ1, x)
∇logπ0(x) = ForwardDiff.gradient(logπ0, x)

## helper plotting function  

ms = 15
Nlog = 300 


xrange = range(xmin; stop = xmax, length = Nlog)
yrange = range(ymin; stop = ymax, length = Nlog)

function plotTimeSeries(tArr, Xpost, tIdx, title)
	for idx = tIdx  
		# true posterior for noise level 
		t = tArr[idx] 
		πt(x) = exp((1-t)*logπ0(x) + t*logπ1(x))
		postVals = [πt([x1; x2]) for x1 in xrange, x2 in yrange]
	
		# plot
		f = Figure()
		ax = Axis(f[1,1], xlabel="x₁", ylabel="x₂", ygridvisible=false, xgridvisible=false, title="$title, t=$t")
	
		contourf!(xrange, yrange, postVals, label="true PDF", levels=8)
		scatter!(Xpost[1, :, idx], Xpost[2, :, idx], color = (:red, 0.5), markersize = ms) 
		xlims!(xmin, xmax)
		ylims!(ymin, ymax)
		display(f)
	end
end

##* KFRFlow    
Xprior = rand(πx, Ne) # initalize particles 
tspan = (0.0, 1.0) 
params = KFRParams(logratio=logRatio, d=Nx, nugget=nugget, k=kfrKernel, bw=kfrBw)
prob = ODEProblem(vKFRFlow!, Xprior, tspan, params)

## Example: solve with Euler 

alg = Euler() 
solEuler = solve(prob, alg, dt=dt, adaptive=false)

Xposterior_Euler = cat(solEuler.u..., dims=3) 
tArr_Euler = solEuler.t 

# plot final time 
plotTimeSeries(tArr_Euler, Xposterior_Euler, [length(tArr_Euler)], "KFRFlow, Euler, dt=$dt")

# to plot entire series, use
#plotTimeSeries(tArr_Euler, Xposterior_Euler, 1:length(tArr_Euler), "Euler, dt=$dt")  

## Example: solve with multistep 
alg = AB4()  
solAB = solve(prob, alg, adaptive=false, dt=dt, progress=true, dense=false, calck=false) 

Xposterior_AB = cat(solAB.u..., dims=3) 
tArr_AB = solAB.t 
plotTimeSeries(tArr_AB, Xposterior_AB, [length(tArr_AB)], "KFRFlow, AB4, dt=$dt")

## Example: use standalone implementation of Euler 
(Xposterior_ES, tArr_ES) = KFRFlowEuler(Xprior, logRatio, savehistory=true, pKernels=1.0, dt=dt, bwSet="full", k=kfrKernel, bw=kfrBw, verbose=false, nugget=nugget)

plotTimeSeries(tArr_ES, Xposterior_ES, [length(tArr_ES)], "KFRFlow, Euler, dt=$dt")

##* KFRFlow-I 
(Xposterior_I, tArr_I) = KFRFlowI(Xprior, logRatio, savehistory=true, pKernels=1.0, dt=dt, k=kfrKernel, bw=kfrBw, verbose=false, nugget=importanceNugget)

plotTimeSeries(tArr_I, Xposterior_I, [length(tArr_I)], "KFRFlow-I, dt=$dt")

##* KFRD
ϵ = 0.1    
stochParams = KFRParams(d=Nx, logratio=logRatio, ϵ=ϵ, ∇logπ0=∇logπ0, ∇logπ1=∇logπ1, k=kfrKernel, bw=kfrBw)
stochProb = SDEProblem(vKFRFlow!, dKFRFlow!, Xprior, tspan, stochParams)

alg = EM() # Euler Maruyama 
solEM = solve(stochProb, alg, dt=dt, adaptive=false, progress=true, dense=false, calck=false)

Xposterior_EM = cat(solEM.u..., dims=3) 
tArr_EM = solEM.t 

plotTimeSeries(tArr_EM, Xposterior_EM, [length(tArr_EM)], "KFRD, Euler-Maruyama, dt=$dt, ϵ=$ϵ")

##* Example usage of EKI 

Ny = 1
pEKI = EKIParams(G=G, ystar=[ystar], Γ=Matrix(σϵ^2*I, 1, 1))
alg = EM()  
EKInoise = WienerProcess(0.0, zeros(Ny, Ne)) #, sqrt(dt)*randn(Nx, Ne)
EKIprob = SDEProblem(vEKI!, dEKI!, Xprior, tspan, pEKI, noise=EKInoise, noise_rate_prototype=zeros(Nx,Ny))

solEKI_EM = solve(EKIprob, alg, dt=dt, adaptive=false, progress=true)

XposteriorEKI_EM = cat(solEKI_EM.u..., dims=3) 
tArrEKI_EM = solEKI_EM.t 

plotTimeSeries(tArrEKI_EM, XposteriorEKI_EM, [length(tArrEKI_EM)], "EKI, Euler-Maruyama, dt=$dt")

## Standalone EKI 

(Xposterior_EKI, tArr_EKI) = EKI(Xprior, [ystar], G, dt=dt, Γ=σϵ^2*Matrix(I, size(ystar, 1), size(ystar, 1)), savehistory=true, verbose=false)

plotTimeSeries(tArr_EKI, Xposterior_EKI, [length(tArr_EKI)], "EKI, dt=$dt")

##* SVGD without adaptive bandwidth selection, using DifferentialEquations 
Tstop = 50.0
tspanSVGD = (0.0, Tstop)

Nsteps = 1/dt 
dtSVGD = Tstop/Nsteps  

pSVGD = SVGDParams(∇logπ1=∇logπ1, k=kfrKernel, bw=kfrBw)

svgdProb = ODEProblem(vSVGD!, Xprior, tspanSVGD, pSVGD)
alg = Euler() 
solSVGD_Euler = solve(svgdProb, alg, dt=dtSVGD, adaptive=false, progress=true) 

XposteriorSVGD_Euler = cat(solSVGD_Euler.u..., dims=3) 
tArrSVGD_Euler = solSVGD_Euler.t

# different plotting routine 
postVals = [π1([x1; x2]) for x1 in xrange, x2 in yrange]

f = Figure()
ax = Axis(f[1,1], xlabel="x₁", ylabel="x₂", ygridvisible=false, xgridvisible=false, title="SVGD, dt=$dtSVGD, T=$Tstop")

contourf!(xrange, yrange, postVals, label="true PDF", levels=8)
scatter!(XposteriorSVGD_Euler[1, :, end], XposteriorSVGD_Euler[2, :, end], color = (:red, 0.5), markersize = ms) 
xlims!(xmin, xmax)
ylims!(ymin, ymax)
display(f)

## Standalone SVGD 

#* SVGD with adaptive bandwidth selection, standalone implementation 
(Xposterior_SVGD, tArr_SVGD) = SVGD(Xprior, ∇logπ1,; dt=dtSVGD, Tstop=Tstop, k=kfrKernel, verbose=false, savehistory=true, bw=nothing)

f = Figure()
ax = Axis(f[1,1], xlabel="x₁", ylabel="x₂", ygridvisible=false, xgridvisible=false, title="SVGD, dt=$dtSVGD, T=$Tstop")

contourf!(xrange, yrange, postVals, label="true PDF", levels=8)
scatter!(Xposterior_SVGD[1, :, end], Xposterior_SVGD[2, :, end], color = (:red, 0.5), markersize = ms) 
xlims!(xmin, xmax)
ylims!(ymin, ymax)
display(f)