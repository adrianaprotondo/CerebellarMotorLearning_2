#####
# Produce graphs of learning speed and steady state loss 
# for different input expansions Ns and learning steps
# We test two different ways of computing the learning parameters
# 1. Fit the task loss as an exponential decay c_1 e^(-c_2 t )+c_3. 
#       the learning speed is c_2 and steady state loss is c_3
# 2. Learning speed is the mean over first t_ls epochs of νₜ = -ΔFₜ/(δₜFₜ) (where F is the task loss)
#       Steady state loss is the mean of the last task loss over the last t_ls
# Use data produced by testSize_static_Ls-SS_simulate saved in datadir(path)
# Set simModel, path and pretrain used in testSize_static_Ls-SS_simulate 
# Plots produced:
# - optimal learning speed and optimal ss vs Ns and for different gammas
# - scatter ls vs SS for different Ns 
# - line plots ls and ss as a function of gammas or Ns (different color lines)
#####

using CerebellarMotorLearning
using Distributions
using Plots
using Random
using LaTeXStrings
using LinearAlgebra
using LsqFit
using Printf
using DrWatson

using DataFrames

include("sizeSim_functions.jl")
simModel = "LMS_size_test_seed9"
path = string(simModel,"_",10,"_",90)
preTrain = true

############
# Load simulation data as a Dataframe using DrWatson
############
# dfLsSS has a columns named :postStores, mu, Ns,seed
dfLsSS = collect_results(datadir("simulations",path);verbose=true) # get data in the path with simulations
sort!(dfLsSS,:seed)

# values of parameters used for simulations
seeds = unique(dfLsSS[!,:seed])
mus = unique(dfLsSS[!,:mu])[1]
Ns = unique(dfLsSS[!,:Ns])[1] # get net sizes
t_save = dfLsSS[!,:t_save][1]

NsNorm = Ns./Ns[1] # expansion ratio
Nmus = length(mus[1]) # number of different mus for each size


############
## Extract poststores and trainstores according to seed
############
# get postStores an array of arrays size length(seeds)xlength(Ns)xlength(mus)
postStores = map(seeds) do s # get postStore for each net size for a given seed
    filter(row -> (row.seed==s),dfLsSS)[:,:postStores][1]
end
if preTrain # get post-stores to compute SS loss if there was pretrain
    postStoresSS = map(seeds) do s # get postStore for each net size for a given seed
        filter(row -> (row.seed==s),dfLsSS)[:,:postStoresSS][1]
    end
    loop_3(postStoresSS,computeDynamicSS!,[100,false]) # compute dynamic ss
end

# Compute learning speed and steady state loss from mean (described above)
loop_3(postStores,computeDynamicLs!,[t_save,50]) # compute dynamic ls 
loop_3(postStores,computeDynamicSS!,[50]) # compute dynamic ss

############
# Extract values of the learning speed and steady state loss
############

# All values
lsAll = loop_3(postStores,getValF,[:learningSpeed,mean])
lsAllD = loop_3(postStores,getValF,[:dynamicLs,mean])
ssAll = loop_3(postStoresSS,getValF,[:steadyStateE,mean])
if preTrain
    ssAllD = loop_3(postStoresSS,getValF,[:dynamicSS,mean])
else
    ssAllD = loop_3(postStores,getValF,[:dynamicSS,mean])
end

# Select values for a subset of learning steps gammas and seeds 
gammaI=1:1:Nmus # interval of gammas to select
gammaIVar = [gammaI for i=1:length(NsNorm)]

int =1:length(seeds) # simulations to select

# arrays of learning performance variables for subset of gammas 
lsAllS = [[lsAll[i][j][gammaIVar[j]]  for j=1:length(Ns)] for i=int]
ssAllS = [[ssAll[i][j][gammaIVar[j]]  for j=1:length(Ns)] for i=int] 

# selected learning steps
musS = [mus[i][gammaIVar[i]] for i=1:length(Ns)] # learning steps selected


############
# Prepare labels and paths for plots
############
mkpath(plotsdir(path)) # make directory to save plots

xlbl = "expansion ratio (q)" # xlabel for plots 

# label of expansion ratio for plots
lbl = string.("q=",NsNorm)
lbl = reshape(lbl,(1,length(lbl)))

############
# optimal values lsOpt and ssOpt muOpt for each size each sim 
############

# optimal over all learning steps gammas for each sim and N
lsOpt = [[maximum(lsAllS[j][i][:]) for i=1:length(Ns)] for j=1:length(lsAllS)]
ssOpt = [[minimum(ssAllS[j][i][:]) for i=1:length(Ns)] for j=1:length(ssAllS)]
ssOpt2 = [[ssAllS[j][i][argmax(lsAllS[j][i][:])] for i=1:length(Ns)] for j=1:length(ssAllS)]

# colorsLines =  ["#66c2a5","#fc8d62","#8da0cb"]
colorsLines=["#a6cee3","#1f78b4","#b2df8a","#33a02c"]

# maximal learning speed over all gammas
lt = :scatter
saveLbl = plotsdir(path,"lsOpt.pdf")
llbl = [L"$\nu^*$"]
plot_mean(NsNorm,lsOpt,llbl,xlbl,L"optimal learning speed ($\nu^*$)",saveLbl,false,lt,colorsLines,7)

# minimal ss over all gammas
saveLbl = plotsdir(path,"ssOpt.pdf")
llbl = [L"$\xi^*$"]
plot_mean(NsNorm,ssOpt,llbl,xlbl,L"optimal steady state loss ($\xi^*$)",saveLbl,false,lt,colorsLines,7)

# optimal ss value for the gamma that gives optimal learning speed
saveLbl = plotsdir(path,"ss_atOptls.pdf")
llbl = [L"$\xi^*$"]
plot_mean(NsNorm,ssOpt2,llbl,xlbl,"steady state loss at\n optimal learning speed ",saveLbl,false,lt,colorsLines,7)

## Values of learning step gamma for optimal ls and ss
muOptLs = [[musS[i][argmax(lsAllS[j][i][:])] for i=1:length(Ns)] for j=1:length(lsAllS)]
muOptss = [[musS[i][argmin(ssAllS[j][i][:])] for i=1:length(Ns)] for j=1:length(ssAllS)]

saveLbl = plotsdir(path,"gammaOptLs.pdf")
llbl = [L"$\gamma^*_{\nu}$"]
plot_mean(NsNorm,muOptLs,llbl,xlbl,L"optimal learning step ($\gamma^*_{\nu}$)",saveLbl,false,lt,false,7)

saveLbl = plotsdir(path,"gammaOptss.pdf")
llbl = [L"$\gamma^*_{\xi}$"]
plot_mean(NsNorm,muOptss,llbl,xlbl,L"optimal learning step ($\gamma^*_{\xi}$)",saveLbl,false,lt,false,7)

############
# scatter plot ls vs ss for different subsets of Ns
############
"""
    plot scatter plot of ss vs ls for a subset of sizes with indeces sizeIs
"""
function plot_scatter_Nint(ssAll,lsAll,sizeIs,slbl,stdErr=false,gammaI=false,ms=4)
    ssM = map(1:length(ssAll)) do i
        ssAll[i][sizeIs]
    end
    lsM = map(1:length(lsAll)) do i
        lsAll[i][sizeIs]
    end
    plot_scatterMean(ssM,lsM,lbl[sizeIs],gammaI,plotsdir(path,slbl),false,ms,stdErr)
end

# multiple scatter plots for different subset of Ns. If we select all, the graph is too crowded
plot_scatter_Nint(ssAllS,lsAllS,[1],"ss_vs_ls_1.pdf",true)
plot_scatter_Nint(ssAllS,lsAllS,1:5:length(Ns),"ss_vs_ls_5.pdf",true)
plot_scatter_Nint(ssAllS,lsAllS,[1,9],"ss_vs_ls_1_9.pdf",true)
plot_scatter_Nint(ssAllS,lsAllS,[1,3,length(Ns)],"ss_vs_ls_1_3_end.pdf",true,false,6)
plot_scatter_Nint(ssAllS,lsAllS,[4],"ss_vs_ls_4.pdf",true,false,6)


############
# Line plots of mean ls and ss (both method of computation) for all Ns and gammas
############
int =1:length(seeds)
cm = palette(:heat,length(NsNorm)+1)[2:end]

xlbl=L"\gamma"
plotLines(lsAll,int,mus,lbl,xlbl,"Learning speed","ls_vs_mus.pdf",cm)
plotLines(lsAllD,int,mus,lbl,xlbl,"Learning speed","ls_vs_musD.pdf",cm)
plotLines(ssAll,int,mus,lbl,xlbl,"Steady state loss","ss_vs_mus.pdf",cm)
plotLines(ssAllD,int,mus,lbl,xlbl,"Steady state loss","ss_vs_musD.pdf",cm)
############
# Plot task loss for different gammas for one size
############

# label for plots of different learning steps
lbl2 = string.(L"\gamma=",mus[1])
lbl2 = reshape(lbl2,(1,length(lbl2))) 
lblMuV = map(1:length(Ns)) do i
    string.(L"\gamma=",round.(mus[i],digits=4))
end

# Plot task loss for different gammas for one size
postStore = postStores[1][1]
Fs= map(postStore) do s
    s[:taskError]
end
plot(t_save,vcat(Fs...)',lw=3,label=lbl2)
plot!(xlabel="trajectory time")
plot!(ylabel="task loss")
savefig(plotsdir(path,"taskErrors.pdf")) 
