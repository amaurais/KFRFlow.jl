using KFRFlow   
using Distributions 
using Random 
using KernelFunctions
using LinearAlgebra
using ForwardDiff 
using DifferentialEquations 
using JLD2
using NamedColors 
using ColorSchemes 
using CairoMakie 
using Makie.Colors 
using Statistics 

using SteinDiscrepancy 
#* Note: This example uses SteinDiscrepancy.jl, available at https://github.com/jgorham/SteinDiscrepancy.jl/tree/master. The original package is old and does not work with newer versions of Julia, but there is an updated version available at https://github.com/Red-Portal/SteinDiscrepancy.jl/tree/master which should work with newer versions of Julia.  


example = "funnel"


# parameter ranges 
NxArr = [5; 10; 15; 20]
λArr = [1e-2; 1e-3; 1e-4]
ϵArr = [1.0; 5.0; 10.0]
Tarr = [25; 50; 100]
Nrepeat = 30 


nNx = length(NxArr) 
nλ = length(λArr) 
nϵ = length(ϵArr) 
nT = length(Tarr) 

#* Test KFRFlow-AB4, KFRD, KFRFlow-I, and SVGD 

# fixed parameters 
dt = 0.01 
Nsteps = Int(1/dt) 

Ne = 100 
σ0 = 1.0 
tspan = (0.0, 1.0) 


samplesSVGD = Dict( [(Nx, zeros(nT, Nrepeat, Nx, Ne)) for Nx in NxArr] )
samplesAB4 = Dict( [(Nx, zeros(nλ, Nrepeat, Nx, Ne)) for Nx in NxArr] )
samplesI = Dict( [(Nx, zeros(nλ, Nrepeat, Nx, Ne)) for Nx in NxArr] )
samplesEM = Dict( [(Nx, zeros(nϵ, Nrepeat, Nx, Ne)) for Nx in NxArr] )

ksdSVGD = zeros(nNx, nT, Nrepeat, Nsteps+1)
ksdAB4 = zeros(nNx, nλ, Nrepeat, Nsteps+1)
ksdI = zeros(nNx, nλ, Nrepeat, Nsteps+1)
ksdEM = zeros(nNx, nϵ, Nrepeat, Nsteps+1)

ksdKernel = SteinInverseMultiquadricKernel()

function calcStoreKSD(iNx, iParam, iRep, storage, ∇logπ1, u, t, integrator)
	storage[ iNx, iParam, iRep, integrator.iter + 1 ] = sqrt(ksd(points=Matrix(u'), gradlogdensity=∇logπ1, kernel=ksdKernel).discrepancy2) 
end

savepath = "" 
if savepath == "" 
	error("Please specify a location to save the data")
end

##
mainRNG = Xoshiro(10331033) 
seedMax = 10^8 
seedArr = [rand(mainRNG, 1:seedMax) for i=1:nNx]

processRNG = Xoshiro(123456) 

## 
rerun = true  

if rerun 
	for (iNx, Nx) in enumerate(NxArr) 
		logπ1(x) = -0.5*(x[1]^2/9 + x[2:Nx]'*x[2:Nx]/exp(x[1])) 
		
		μ0 = zeros(Nx)
		Σ0 = Diagonal(σ0* Matrix(I, (Nx, Nx))) 
		Γ0 = Diagonal((1/σ0) * Matrix(I, (Nx, Nx)))
		
		logπ0(x) = -0.5*(x - μ0)'*Γ0*(x - μ0)
		πx = MvNormal(μ0, Σ0)
		π0(x) = exp(logπ0(x))
		
		logRatio(x) = logπ1(x) - logπ0(x) 
		densityRatio(x) = exp(logRatio(x))
		
		∇logπ0(x) = ForwardDiff.gradient(logπ0, x)
		∇logπ1(x) = ForwardDiff.gradient(logπ1, x)

		Xprior_arr = [ rand(mainRNG, πx, Ne) for i=1:Nrepeat ]

		for (iλ, λ) in enumerate(λArr)  
			Threads.@threads for iRep=1:Nrepeat 
				try
					(Xkfr, tkfr) = KFRFlowI(Xprior_arr[iRep], logRatio; dt=dt, savehistory=true, nugget=λ)
					samplesI[Nx][iλ, iRep, :, :] = Xkfr[:, :, end]
					ksdI[iNx, iλ, iRep, :] = [ sqrt(ksd(points=Matrix(Xkfr[:, :, j]'), gradlogdensity=∇logπ1, kernel=ksdKernel).discrepancy2) for j=1:length(tkfr) ]
				catch
					samplesI[Nx][iλ, iRep, :, :] .= NaN 
					ksdI[iNx, iλ, iRep, :] .= NaN 
					println("Importance errored on Nx=$Nx, λ=$λ, repeat=$iRep")
					flush(stdout) 
				end
			end
			println("Finished importance, Nx=$Nx, λ=$λ")
			flush(stdout) 
		end
	

		fnameI = "testImportance_$(example)_d$Nx.jld2"	
		jldsave("$savepath/$fnameI"; ksdI, samplesI)
		println("Finished importance, Nx=$Nx, saved data")
		flush(stdout)

		for (iλ, λ) in enumerate(λArr)  
			params = KFRParams(logratio=logRatio, d=Nx, nugget=λ)  
			alg = AB4() 
	
			Threads.@threads for iRep = 1:Nrepeat 
				base_prob = ODEProblem(vKFRFlow!, Xprior_arr[iRep], tspan, params)
	
				cbfunc(u, t, integrator) = calcStoreKSD(iNx, iλ, iRep, ksdAB4, ∇logπ1, u, t, integrator)
				cb = FunctionCallingCallback(cbfunc, func_everystep=true)
				try 
					solRep = solve(base_prob, alg, adaptive=false, dt=dt, saveat=[1.0], callback=cb) 
					samplesAB4[Nx][iλ, iRep, :, :] = solRep.u[1] 
	
					if solRep.t[1] != 1.0 
						# throw an error 
						error("early exit due to instability")
					end
				catch 
					samplesAB4[Nx][iλ, iRep, :, :] .= NaN 
					ksdAB4[iNx, iλ, iRep, :] .= NaN 
					println("AB4 errored on Nx=$Nx, λ=$λ, repeat=$iRep")
					flush(stdout) 
				end
			end
			println("Finished AB4, Nx=$Nx, λ=$λ")
			flush(stdout) 
		end
	
		fnameAB4 = "testAB4_$(example)_d$Nx.jld2"
		jldsave("$savepath/$fnameAB4"; ksdAB4, samplesAB4)
		println("Finished AB4, Nx=$Nx, saved data")
		flush(stdout) 


		for (iϵ, ϵ) in enumerate(ϵArr) 
			params = KFRParams(logratio=logRatio, d=Nx, ϵ=ϵ, ∇logπ0=∇logπ0, ∇logπ1=∇logπ1)  
			alg = EM() 
	
			repSeeds = [rand(processRNG, 1:seedMax) for i=1:Nrepeat]
	
			Threads.@threads for iRep = 1:Nrepeat 
				SDE_noise = WienerProcess(0.0, zeros(Nx, Ne), rng=Xoshiro(repSeeds[iRep]))
				base_prob = SDEProblem(vKFRFlow!, dKFRFlow!, Xprior_arr[iRep], tspan, params, noise=SDE_noise)
	
				cbfunc(u, t, integrator) = calcStoreKSD(iNx, iϵ, iRep, ksdEM, ∇logπ1, u, t, integrator)
				cb = FunctionCallingCallback(cbfunc, func_everystep=true)
				try 
					solRep = solve(base_prob, alg, adaptive=false, dt=dt, saveat=[1.0], callback=cb) 
					samplesEM[Nx][iϵ, iRep, :, :] = solRep.u[1] 
	
					if solRep.t[1] != 1.0 
						# throw an error 
						error("early exit due to instability")
					end
				catch 
					samplesEM[Nx][iϵ, iRep, :, :] .= NaN 
					ksdEM[iNx, iϵ, iRep, :] .= NaN 
					println("EM errored on Nx=$Nx, ϵ=$ϵ, repeat=$iRep")
					flush(stdout) 
				end
			end
			println("Finished EM, Nx=$Nx, ϵ=$ϵ")
			flush(stdout)
		end
	 
		fnameEM = "EM_$(example)_d$Nx.jld2"	
		jldsave("$savepath/$fnameEM"; ksdEM, samplesEM)
		println("Finished EM, Nx=$Nx, saved data")
		flush(stdout) 

		for (iT, T) in enumerate(Tarr) 
			# svgd 
			dtSVGD = T/Nsteps 
			tspanSVGD = (0.0, T)
	
			params = SVGDParams(∇logπ1=∇logπ1) 
			#base_prob = ODEProblem(vSVGD!, Xprior_arr[1], tspanSVGD, params)  
			alg = Euler() 
	
			Threads.@threads for iRep = 1:Nrepeat 
				# shoot shoot shoot 
				base_prob = ODEProblem(vSVGD!, Xprior_arr[iRep], tspanSVGD, params)  
	
				cbfunc(u, t, integrator) = calcStoreKSD(iNx, iT, iRep, ksdSVGD, ∇logπ1, u, t, integrator)
				cb = FunctionCallingCallback(cbfunc, func_everystep=true)
				#cb = FunctionCallingCallback(cbfunc, funcat=0.0:dtSVGD:T)
				try 
					solRep = solve(base_prob, alg, adaptive=false, dt=dtSVGD, saveat=[T], callback=cb) 
					samplesSVGD[Nx][iT, iRep, :, :] = solRep.u[1] 
	
					if solRep.t[1] != T 
						# throw an error 
						error("early exit due to instability")
					end
				catch 
					samplesSVGD[Nx][iT, iRep, :, :] .= NaN 
					ksdEM[iNx, iT, iRep, :] .= NaN 
					println("SVGD errored on Nx=$Nx, T=$T, repeat=$iRep")
					flush(stdout) 
				end
			end
			println("Finished SVGD, Nx=$Nx, T=$T")
			flush(stdout) 
		end
	
		fnameSVGD = "testSVGD_$(example)_d$Nx.jld2"
		jldsave("$savepath/$fnameSVGD"; ksdSVGD, samplesSVGD)
		println("Finished SVGD, Nx=$Nx, saved data")
		flush(stdout) 
	end
end 

## reload data 

reloaddata = true 

if reloaddata 
	SVGDdata = load("$savepath/SVGD_funnel_d20.jld2")
	AB4data = load("$savepath/AB4_funnel_d20.jld2")
	EMdata = load("$savepath/EM_funnel_d20.jld2")
	Idata = load("$savepath/importance_funnel_d20.jld2")
end

samplesSVGD = SVGDdata["samplesSVGD"]
samplesAB4 = AB4data["samplesAB4"]
samplesI = Idata["samplesI"]
samplesEM = EMdata["samplesEM"]

ksdSVGD = SVGDdata["ksdSVGD"]
ksdAB4 = AB4data["ksdAB4"]
ksdI = Idata["ksdI"]
ksdEM = EMdata["ksdEM"]

## 

ksdEndSVGD = dropdims(mean(ksdSVGD[:, :, :, end], dims=3), dims=3) 
ksdEndAB4 = dropdims(mean(ksdAB4[:, :, :, end], dims=3), dims=3) 
ksdEndEM = dropdims(mean(ksdEM[:, :, :, end], dims=3), dims=3) 
ksdEndI = dropdims(mean(ksdI[:, :, :, end], dims=3), dims=3) 

SVGDparam = 3 
AB4param = 2
EMparam = 2  
Iparam = 2 

SVGDevol = dropdims(mean(ksdSVGD[:, SVGDparam, :, :], dims=2), dims=2)
EMevol = dropdims(mean(ksdEM[:, EMparam, :, :], dims=2), dims=2)
Ievol = dropdims(mean(ksdI[:, Iparam, :, :], dims=2), dims=2)
AB4evol = dropdims(mean(ksdAB4[:, AB4param, :, :], dims=2), dims=2)

##
bigSize = 30 
smallSize = 24
fontsize_theme = Theme(Axis=(titlesize=bigSize, xlabelsize=bigSize, ylabelsize=bigSize, xticklabelsize=smallSize, yticklabelsize=smallSize, ylabelfont=:bold, xlabelfont=:bold, xticklabelfont=:bold, yticklabelfont=:bold, titlefont=:bold), Legend=(labelfont=:bold, labelsize=smallSize, margin=(10, 20, 10, 10)))
set_theme!(fontsize_theme)

ms = 30 
lw = 6 
##

fig = Figure(resolution=(1300, 400), figure_padding=0) # resolution=(1600, 800)
ax = Axis(fig[1,1], xlabel="d", ylabel="KSD", xreversed=false, xticks=NxArr, yscale=identity)

labels = ["SVGD", "KFRD", "KFRFlow-I"]
linestyles = [:dash, :dash, :solid,]
markers = [:diamond, :diamond, :circle, ]
colors = [colorant"maroon", colorant"navy blue", colorant"blue"]

datasets = [ksdEndSVGD[:, SVGDparam], ksdEndEM[:, EMparam], ksdEndI[:, Iparam],]

for (idx, d) in enumerate(datasets)
	scatterlines!(NxArr[1:3], d[1:3], linewidth=lw, color=colors[idx], marker=markers[idx], linestyle=linestyles[idx], markersize=ms, label=labels[idx])
end

axislegend(position=:rb, nbanks=3) 
ylims!(0.0, 1.1)
#ylims!(2^-2.2, 1)
display(fig) 

## 

fig = Figure(resolution=(1300, 400), figure_padding=0) # resolution=(1600, 800)
ax = Axis(fig[1,1], xlabel="d", ylabel="KSD", xreversed=false, xticks=NxArr, yscale=log2, yticks=2.0 .^ (-2:1:0), ytickformat=values -> [rich("2", superscript("$(Int(log2(tl)))")) for tl in values])

labels = ["SVGD", "KFRD", "KFRFlow-I"]
linestyles = [:dash, :dash, :solid,]
markers = [:diamond, :diamond, :circle, ]
colors = [colorant"maroon", colorant"navy blue", colorant"blue"]

datasets = [ksdEndSVGD[:, SVGDparam], ksdEndEM[:, EMparam], ksdEndI[:, Iparam],]

for (idx, d) in enumerate(datasets)
	scatterlines!(NxArr[1:3], d[1:3], linewidth=lw, color=colors[idx], marker=markers[idx], linestyle=linestyles[idx], markersize=ms, label=labels[idx])
end

axislegend(position=:rb, nbanks=3) 
#ylims!(0.0, 1.0)
ylims!(2^-2.2, 1.1)
display(fig) 

## 

##* Plot evolutions grouped by method type, colored by dimension 

color_theme = Theme(Axis=(titlesize=bigSize, xlabelsize=bigSize, ylabelsize=bigSize, xticklabelsize=smallSize, yticklabelsize=smallSize, ylabelfont=:bold, xlabelfont=:bold, xticklabelfont=:bold, yticklabelfont=:bold, titlefont=:bold, xgridvisible=false), Legend=(labelfont=:bold, labelsize=smallSize, margin=(10, 20, 10, 10)), palette=(color=ColorSchemes.Johnson.colors[3:-1:1],)) 

set_theme!(color_theme)

ms = 20
lw = 6

titles = true       

for (idx, d) in enumerate(evolData) 
	if titles 
		titleStr = labels[idx+1]
	else
		titleStr = "" 
	end

	fig = Figure(resolution=(800, 400), figure_padding=0) # resolution=(1600, 800)
	ax = Axis(fig[1,1], xlabel="t", ylabel="KSD", title=titleStr, xreversed=false, xgridvisible=false, ygridvisible=true, yscale=identity)

	for (iNx, Nx) in enumerate(NxArr[1:3])
		lines!(0:dt:1, d[iNx, :], linewidth=lw, label="d = $Nx")
	end

	axislegend() 
	ylims!(0.0, 5.5)
	display(fig) 

end

## * log scaled 
titles = true         

for (idx, d) in enumerate(evolData) 
	if titles 
		titleStr = labels[idx+1]
	else
		titleStr = "" 
	end

	fig = Figure(resolution=(800, 400), figure_padding=0) # resolution=(1600, 800)
	ax = Axis(fig[1,1], xlabel="t", ylabel="KSD", title=titleStr, xreversed=false, xgridvisible=false, ygridvisible=true, yscale=log2)

	for (iNx, Nx) in enumerate(NxArr[1:3])
		lines!(0:dt:1, d[iNx, :], linewidth=lw, label="d = $Nx")
	end

	axislegend() 
	ylims!(2^-2, 2^2.9)
	display(fig) 

end