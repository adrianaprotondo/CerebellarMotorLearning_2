#######
# Helper functions for simulations of motor control system: 
## functions to build expanded systems with size N given original network with some parameters
## function build_systems_sim returns systems with different net sizes given by Ns with random seed randomSeed
## function build_systems_sim_K: build all the systems with sizes Ns and input sparsity Kss but same ref, plant... keep same W for different K. expand W with zeros for different N. returns vector of vector of systems with length(Kss)xlength(Ns)
## function build_systems_sim_K_KNconst: same as above but keeping the number of input connections to Kss[end]*N (i.e. to the largest size)
## plotting functions
## helper functions
######



###############
# Build systems 
###############
"""
build system of size N, and K indegree given parameters and matrices of smallest net Z0 and W0
params = [plant,pid,nnDims,ref,trajTime,lookahead_times,Z0,W0] 
"""
function build_system_N(params,N,K)
    params = systemParams!(params,N,K)
    build_system(params...)
end

"""
build inputsystem with K inputs per gc and N GCs given parameters and matrices of smallest net Z0 and W0
params = [nnDims,ref,trajTime,lookahead_times,Z0,W0] 
"""
function build_inputSystem_N(params,N,K)
    params = systemParams!(params,N,K)
    build_inputSystem(params...)
end

"""
    build parameters for system of size N,and K indegree 
    given parameters and matrices of smallest net Z0 and W0
    params = [plant,pid,nnDims,ref,trajTime,lookahead_times,Z0,W0] 
"""
function systemParams!(params,N,K)
    Z0 = params[end-1]
    W0 = params[end]
    nnDims = params[3]
    K0 = size(Z0,2)-sum(Z0[1,:].==0)
    if K0!=K # if K is changed
        Z = createZ(nnDims[1],N,K) # change input matrix with new
        params[end-1] = Z
    elseif size(Z0,1) < N # if K is not changed but expansion
        Z = createZ(nnDims[1],N,K)
        Z[1:size(Z0,1),:] = Z0
        params[end-1] = Z
    end
    if size(W0,2) < N # if net is larger than weight matrices
        W = zeros(nnDims[3],N)
        W[:,1:size(W0,2)] = W0
        params[3] = (nnDims[1],N,nnDims[3])
        params[end] = W 
    end
    return params
end

"""
    build systems for simulation randomSeed sets seed for weight matrices and reference 
    build all the systems with sizes Ns but same ref, plant... 
"""
function build_systems_sim(Ns,plantMatrices,Ks,refF,trajTime,lookahead_times,num_nn_inputs,num_nn_outputs,K,randomSeed) 
    Random.seed!(randomSeed)
    nnDims = (num_nn_inputs,Ns[1],num_nn_outputs)
    Z0 = createZ(num_nn_inputs,Ns[1],K)
    d = Normal(0,1/sqrt(Ns[1]))
    W0 = rand(d,(num_nn_outputs,Ns[1]))

    system = build_system(plantMatrices,Ks,nnDims,refF,trajTime,lookahead_times,Z0,W0);
    plant = system.plant
    pid = system.pid
    ref = system.ref
    # reuse plant, pid and ref
    paramsN = [plant,pid,nnDims,ref,trajTime,lookahead_times,Z0,W0]
    systems = map(Ns) do N 
        build_system_N(paramsN,N,K)
    end
    return systems
end

"""
    build systems for simulation randomSeed sets seed for weight matrices and reference 
    build all the systems with sizes Ns and input sparsity Kss but same ref, plant...
    keep same W for different K 
    expand W with zeros for different N
    return vector of vector of systems with length(Kss)xlength(Ns) 
"""
function build_systems_sim_K(Ns,plantMatrices,Ks,refF,trajTime,lookahead_times,num_nn_inputs,num_nn_outputs,Kss,randomSeed) 
    Random.seed!(randomSeed)
    nnDims = (num_nn_inputs,Ns[1],num_nn_outputs)
    Z0 = createZ(num_nn_inputs,Ns[1],Kss[1])
    d = Normal(0,1/sqrt(Ns[1]))
    W0 = rand(d,(num_nn_outputs,Ns[1]))
    system = build_system(plantMatrices,Ks,nnDims,refF,trajTime,lookahead_times,Z0,W0);
    plant = system.plant
    pid = system.pid
    ref = system.ref
    # reuse plant, pid and ref
    # paramsN = [plant,pid,nnDims,ref,trajTime,lookahead_times,Z0,W0]
    systems = map(Kss) do K
        Z0 = createZ(num_nn_inputs,Ns[1],K) # new input weight matrix for K
        map(Ns) do N 
            build_system_N([plant,pid,nnDims,ref,trajTime,lookahead_times,Z0,W0],N,K)
        end
    end
    return systems
end

"""
    build systems for simulation randomSeed sets seed for weight matrices and reference 
    build all the systems with sizes Ns and input sparsity Kss but same ref, plant...
    keep number of input connections to Kss[end]*N
    keep same W for different K 
    expand W with zeros for different N
    return vector of vector of systems with length(Kss)xlength(Ns) 
"""
function build_systems_sim_K_KNconst(N,plantMatrices,Ks,refF,trajTime,lookahead_times,num_nn_inputs,num_nn_outputs,Kss,randomSeed) 
    Random.seed!(randomSeed)
    Ns = trunc.(Int,(Kss[end]*N)./Kss)# keep the number of input connections constant to K[end]*N
    nnDims = (num_nn_inputs,Ns[end],num_nn_outputs)
    Z0 = createZ(num_nn_inputs,Ns[end],Kss[end])
    d = Normal(0,1/sqrt(Ns[end]))
    W0 = rand(d,(num_nn_outputs,Ns[end]))
    system = build_system(plantMatrices,Ks,nnDims,refF,trajTime,lookahead_times,Z0,W0);
    plant = system.plant
    pid = system.pid
    ref = system.ref
    # reuse plant, pid and ref
    # paramsN = [plant,pid,nnDims,ref,trajTime,lookahead_times,Z0,W0]
    systems = map(1:length(Kss)) do i
        # Z0 = createZ(num_nn_inputs,Ns[i],Kss[i]) # new input weight matrix for K
        build_system_N([plant,pid,nnDims,ref,trajTime,lookahead_times,Z0,W0],Ns[i],Kss[i])
    end
    return systems
end

"""
    build input systems for simulation randomSeed sets seed for weight matrices and reference 
    build all the inputsystems with sizes Ns but same ref 
"""
function build_systems_sim(Ns,fc,trajTime,lookahead_times,num_nn_inputs,num_nn_outputs,K,randomSeed) 
    Random.seed!(randomSeed)
    nnDims = (num_nn_inputs,Ns[1],num_nn_outputs)
    Z0 = createZ(num_nn_inputs,Ns[1],K)
    d = Normal(0,1/sqrt(Ns[1]))
    W0 = rand(d,(num_nn_outputs,Ns[1]))

    system = build_inputSystem(nnDims,fc,trajTime,lookahead_times,Z0,W0);
    ref = system.ref
    # reuse plant, pid and ref
    # paramsN = [nnDims,ref,trajTime,lookahead_times,Z0,W0]
    systems = map(Ns) do N 
        build_inputSystem_N([nnDims,ref,trajTime,lookahead_times,Z0,W0],N,K)
    end
    return systems
end

"""
    build input systems for simulation randomSeed sets seed for weight matrices and reference 
    build all the inputsystems with size N and Ks but same ref 
    if constantNumWeights keep the number of weights N*K constant, else keep N constant for different K
"""
function build_systems_sim_K(N,fc,trajTime,lookahead_times,num_nn_inputs,num_nn_outputs,Ks,randomSeed,constantNumWeights=false) 
    Random.seed!(randomSeed)
    if constantNumWeights # if keep the number of input connections constant to K[end]*N
        Ns = trunc.(Int,(Ks[end]*N)./Ks)
    else # if keep the number of granule cells constant
        Ns = trunc.(Int,N.*ones(length(Ks)))
    end
    # reference values for the smallest number of granule cells (Ns[end] and Ks[end])
    nnDims = (num_nn_inputs,Ns[1],num_nn_outputs) 
    Z0 = createZ(num_nn_inputs,Ns[1],Ks[1])
    d = Normal(0,1/sqrt(Ns[1]))
    W0 = rand(d,(num_nn_outputs,Ns[1]))

    system = build_inputSystem(nnDims,fc,trajTime,lookahead_times,Z0,W0);
    ref = system.ref
    # reuse plant, pid and ref
    # paramsN = [nnDims,ref,trajTime,lookahead_times,Z0,W0]

    systems = map(1:length(Ks)) do i
        build_inputSystem_N([nnDims,ref,trajTime,lookahead_times,Z0,W0],Ns[i],Ks[i])
    end
    return systems
end

"""
    Function that returns output weights of an expanded neural network with N weights
    based on the weight matrix W0 of smaller net 
"""
function expandW(W0,N)
    W = zeros(N,size(W0,2))
    W[1:size(W0,1),:] = W0
    return W 
end

""" Change ouptut weights of the system's nn """
function changeOutputWeights!(system,W)
    s = system.nn.fluxNet
    Wss = zeros(1,system.nn.dims[2])
    Wss[1,1:size(W,2)] = W
    Flux.params(s)[3] .= Wss
    system
end 

""" Pretrain an array of systems with different sizes. Return final weights and initial weights for each one"""
function preTrainF(systems,musVar,Ns,trajTimePTs,dt,trainErrorIndex,trajTimePP,path,plotsPath,funcType="GradientTrain",lossCutoff=0,ssWI=10)
    wss=map(1:length(Ns)) do j # for each size
        t_trainPT = 0.01:dt:trajTimePTs[j] # train times for pretraining 
        t_savePT = t_trainPT[1]:10:trajTimePTs[j] # save less often for comp efficiency
        recordsPT = [LearningPerformanceTask(t_savePT,trainErrorIndex)] # get task loss of pre-training 
        muPT = musVar[j][end] # use largest learning step
        if funcType=="GradientTrain" # pre-train with gradient descent
            uPT = [GradientTrain(muPT,t_trainPT,trainErrorIndex,trajTimePP,lossCutoff)];
        elseif funcType=="LMSTrain" # pre-train with LMS learning ruls
            uPT = [LMSTrain(muPT,t_trainPT,deltaTePT,deltaTrPT,deltaTrPT,trajErrorIndex)];
        end
        s1 = systems[j] # train all nets
        t,p=simulate(s1,uPT,t_savePT,recordsPT,trajTimePTs[j],trajTimePP,plotsdir(path,plotsPath))
        wInitial = t[:weights][:,1]' # store initial weights to set after pre-training to maintain initialisaiton
        wss = mean(t[:weights][:,end-ssWI:end]';dims=1) # ss weights are mean over interval 
        changeOutputWeights!(s1,wInitial) # change output weights back to initial for training 
        if p[:taskError][1,end]>lossCutoff # if the task loss at end of pretrain is larger than cutoff 
            lossCutoff=p[:taskError][1,end] # set as new cutoff for next net sizes this assures that nets achieve approximate same pre-train
        end
        return wInitial, wss
    end
    return wss
end

""" build array of array of updates for each learning step in musVar
    trainType is the function to create updates 
        arg is the arguments for trainType after the learning step should be [t_train,deltaTe,deltaTr,deltaTh,trajErrorIndex]
    SNR if larger than zero add gaussian noise with gamma = learning step/SNR 
    lmsNR if larger than zero add noise inside lms
"""
function buildUpdates(musVar,trainType,arg,SNR=0,lmsNR=0)
    mm = map(musVar) do d # all sizes
        map(d) do dd # all learning steps 
            if trainType == "LMSTrain"
                if lmsNR>0 # add noise to LMS
                    a = LMSTrain(dd,arg...,lmsNR) # training function  
                else
                    a = LMSTrain(dd,arg...) # training function 
                end
            elseif trainType == "OnlineGradientTrain"
                a = OnlineGradientTrain(dd,arg...); 
            end
            if SNR>0 # add gaussian noise
                gamma=dd/SNR
                b = GaussianNoise(gamma,arg[1]); # arg[1] should be trainTime
                m = [a,b]
            else
                m = [a]
            end
            m
        end
    end
    return mm
end

###############
# Save variables
###############
"""
    create file to save variables return filename generated from parameters
    save the parameters given into the file
"""
function createSaveFile(saveParams,filePath="../Variables/")
    mu = saveParams["mu = "]
    gamma = saveParams["gamma = "]
    Ns = saveParams["Ns = "]
    deltaTe = saveParams["deltaTe = "]
    fileName = string(filePath,"sizeTest_mu-",mu,"_gamma-",gamma,"_deltaTe-",deltaTe,"_Ns-",Ns[1],"-",Ns[end],".jl")
    output_file = open(fileName,"w")
    write(output_file,"# Parameters \n \n")
    for (key, value) in saveParams
        write(output_file,key)
        show(output_file,value)
        write(output_file, "; \n \n")
    end 
    close(output_file)
    return fileName
end

"""
    save variable tl with name in fileName
"""
function saveVar(fileName,name,tl)
    output_file = open(fileName,"a")
    write(output_file,name)
    show(output_file,tl)
    write(output_file, "; \n \n") 
    close(output_file)
end

###############
# Analyse simulation results
###############
""" 
loop over a vector of vectors with three layers and apply the function fc to each element
"""
function loop_3(trainStores,fc,args=[])
    map(trainStores) do t
        map(t) do tt
            map(tt) do ttt
                fc(ttt,args...)
            end
        end
    end
end

""" 
loop over a vector of vectors with two layers and apply the function fc to each element
"""
function loop_2(trainStores,fc,args=[])
    map(trainStores) do t
        map(t) do ttt
            fc(ttt,args...)
        end
    end
end

""" 
normalize p by p[1] if non_zero
"""
function normalize_by1(p)
    if p[1]>0.0
        return p./p[1]
    else
        return p 
    end
end 

"""
    extract train stores and post stores of simulation result mm 
    when simulating over two parameters (size and initialisation for example)
"""
function simAnsAnalisis_2(mm)
    trainStores = loop_2(mm,x -> x[1]) 
    postStores = loop_2(mm,x -> x[2]) 
    return trainStores,postStores
end 

"""
    extract trainstores and poststores from solution running simulations with three parameters changing
"""
function simAnsAnalisis_3(mm)
    trainStores = loop_3(mm,x -> x[1]) 
    postStores = loop_3(mm,x -> x[2]) 
    return trainStores,postStores
end 


"""
    extract train stores and post stores of simulation result mm 
    and extract learning speed and steady state error from postStores
    when simulating over two parameters (size and initialisation for example)
"""
function simAnsAnalyse(mm)
    trainStores, postStores = simAnsAnalisis_2(mm)
    ls = postStoresExtract_2(postStores,:learningSpeed)
    lsNorm = map(ls) do p
        normalize_by1(p)
    end
    ss = postStoresExtract_2(postStores,:steadyStateE) 
    ssNorm = map(ss) do p
        normalize_by1(p)
    end
    return trainStores,postStores,ls,ss, lsNorm,ssNorm
end


"""
    extract train stores and post stores of simulation result mm 
    and extract learning speed and steady state error from postStores
    when simulating over three parameters (size, initialisation and deltaTe for example)
"""
function simAnsAnalyse_3(mm)
    trainStores, postStores = simAnsAnalisis_3(mm)
    ls = postStoresExtract_3(postStores,:learningSpeed)
    lsNorm = loop_2(ls,normalize_by1)
    ss = postStoresExtract_3(postStores,:steadyStateE) 
    ssNorm = loop_2(ss,normalize_by1)
    return trainStores,postStores,ls,ss,lsNorm,ssNorm
end

"""
    extract from an array of postStores the variable with symbol symb
    array of postores has three levels
"""
function postStoresExtract_3(postStores,symb)
    map(postStores) do p
        postStoresExtract_2(p,symb)
    end
end
"""
    extract from an array of postStores the variable with symbol symb
    array of postores has two levels
"""
function postStoresExtract_2(postStores,symb)
    map(postStores) do p
        _postStoresExtract(p,symb)
        # map(p) do pp
        #     pp[symb]
        # end
    end
end

"""
    extract from an array of postStores the variable with symbol symb
    array of postores has one level
"""
function _postStoresExtract(postStores,symb)
    map(postStores) do p
        p[symb]
    end
end
# """
#     extract from an array of postStores the variable with symbol symb
#     array of postores has one level
# """
# function postStoresExtract(postStores,symb)
#     p = postStores
#     while typeof
#     map(postStores) do p
#         p[symb]
#     end
# end

"""
    get run function f on postStore[s]
"""
function getValF(postStore,s,f)
    return f(postStore[s])
    # dF = postStore[Symbol(:localTaskDifficulty_,s)]
    # return findmin(dF)[1]
end

"""
    get run function f on postStore[int][s]
"""
function getValF(postStore,s,f,int)
    return f(postStore[int][s])
end

"""
    return array of arrays of arrays with 
    elements from int selected in level 3
"""
function selectInd_3(p,int)
    map(1:length(p)) do i 
        map(int) do j
            map(1:length(p[i])) do k 
                p[i][k][j]
            end
        end
    end
end

""" 
    decompose vec into the direction of ref plus noise
    hat{vec} = γ₁hat{ref} + γ₂hat{n}
    return gamma1, gamma2, and n
"""
function decompose(vec,ref)
    gamma1 = zeros(1,size(vec,2))
    gamma2 = zeros(1,size(vec,2))
    n = zeros(size(vec))
    for i=1:size(vec,2)
        if norm(vec[:,i])>0
            nv = normalize(vec[:,i])
            nr = normalize(ref[:,i])
            gamma1[:,i] .= (nv'*nr)
            noise = nv.-gamma1[:,i].*nr
            gamma2[:,i] .= norm(noise)
            n[:,i] .= normalize(noise)
        end
    end
    return gamma1,gamma2,n 
end


"""
    reuturn norm of each slice of value of trainstore at symb
"""
function calculateNorm(trainStore,symb=:dw)
    mapslices(norm,trainStore[symb],dims=1)'
end


###############
# Plotting functions
###############
"""
    compute standard error of an array of arrays
"""
function my_std(ls)
    numSim = length(ls)
    lsM = hcat(ls...)
    map(1:size(lsM,1)) do m
        std(lsM[m,:])./sqrt(numSim)
    end
end

"""
    plot mean of ls vs Ns 
    label of each line is given by l 
    the xlabel is given by xl and ylabel by yl
    save figure with name saveLbl
"""
function plot_mean(Ns,ls,l,xl,yl,saveLbl,vertical=false,lt=:scatter,cm=false,ms=5)
    if cm==false
        my_colors = ["#D43F3AFF", "#5CB85CFF", "#EEA236FF", "#46B8DAFF","#357EBDFF", "#9632B8FF", "#B8B8B8FF"]
    else
        my_colors = cm
    end
    lsm = mean(ls)
    lss = my_std(ls)
    if length(l) == 1 && length(lsm) > 1# if only one line to plot
        if lt==:scatter
            plot(Ns,lsm,yerr=lss,lw=3,label=l[1],color=my_colors[1],seriestype=lt,legend=:outertopright,grid=false,markersize=ms)
        elseif ls==:line
            plot(Ns,lsm,ribbon=lss,lw=3,label=l[1],color=my_colors[1],seriestype=lt,legend=:outertopright,grid=false,markersize=ms)
        end
        if (vertical==argmax) || (vertical==argmin)
            vline!([Ns[vertical(lsm)]],lw=2,ls=:dash,color=my_colors[1],label = string("optimum ", l[1]))
        end 
    else
        if lt==:scatter
            plot(Ns,lsm[1],yerr=lss[1],lw=3,label=l[1],color=my_colors[1],seriestype=lt,legend=:outertopright,grid=false,markersize=ms)
        elseif lt==:line
            plot(Ns,lsm[1],ribbon=lss[1],lw=3,label=l[1],color=my_colors[1],seriestype=lt,legend=:outertopright,grid=false,markersize=ms)
        end
        if (vertical==argmax) || (vertical==argmin)
            vline!([Ns[vertical(lsm[1])]],lw=2,color=my_colors[1],ls=:dash,label = string("optimum ", l[1]))
        end
        if length(lsm)>1
            for i=2:length(lsm)
                if lt==:scatter
                    plot!(Ns,lsm[i],yerr=lss[i],lw=3,color=my_colors[i],seriestype=lt,label=l[i],markersize=ms)
                elseif lt==:line
                    plot!(Ns,lsm[i],ribbon=lss[i],lw=3,color=my_colors[i],seriestype=lt,label=l[i],markersize=ms)
                end
                if vertical==argmax || vertical==argmin
                    vline!([Ns[vertical(lsm[i])]],lw=2,color=my_colors[i],ls=:dash,label = string("optimum ", l[i]))
                end
            end
        end
    end
    plot!(xlabel=xl)
    plot!(ylabel=yl)
    savefig(saveLbl)
end

"""
    plot scatter of mean of ss and mean of ls at interval int
    assume ss and ls have shape numSim x length(Ns) x length(t_save)
    calculate mean over numSim for each Ns at t_save[int]
"""
function plot_scatterMean(ss,ls,l,int=false,saveLbl="../Figures/deltaTe/ssVsls_$int.pdf",cm=false,ms=4,stdErr=false)
    if int==false # take all the gammas
        ssS = ss
        lsS = ls
    else # select and interval of gammas
        ssS = map(1:length(ss)) do i
            map(1:length(ss[i])) do j
                ss[i][j][int]
            end
        end
        lsS = map(1:length(ls)) do i
            map(1:length(ls[i])) do j
                ls[i][j][int]
            end
        end
    end
    ssM = mean(ssS)
    lsM = mean(lsS)
    if cm == false
        namedColors = ["Blues","Greens","Oranges","Purples","Reds","Grays"]
    else 
        namedColors = cm
    end
    if stdErr ==true && length(ss)>1
        ssS = my_std(ssS)
        lsS = my_std(lsS)
        scatter(ssM[1],lsM[1],c=colormap(namedColors[1],length(ssM[1])+2)[3:end],label=l[1],colorbar=true,grid=false,legend=:outertopright,markersize = ms,yerr=lsS[1],xerr=ssS[1])
    else
        scatter(ssM[1],lsM[1],c=colormap(namedColors[1],length(ssM[1])+2)[3:end],label=l[1],colorbar=true,grid=false,legend=:outertopright,markersize = ms)
    end
    if length(lsM)>1
        for i=2:length(lsM)
            # scatter!(ssM[i],lsM[i],c=colormap(namedColors[i],length(ssM[i])+2)[3:end],label=l[i],markersize = ms)
            if stdErr == true && length(ss)>1
                scatter!(ssM[i],lsM[i],c=colormap(namedColors[i],length(ssM[i])+2)[3:end],label=l[i],markersize = ms,yerr=lsS[1],xerr=ssS[1])
            else
                scatter!(ssM[i],lsM[i],c=colormap(namedColors[i],length(ssM[i])+2)[3:end],label=l[i],markersize = ms)
            end
        end 
    end
    plot!(xlabel="steady state loss")
    plot!(ylabel="learning speed")
    savefig(saveLbl)
end

"""
    Plot mean of lsAll at indeces int vs mus for each N
    lsAll has shape length(numSim)xlength(Ns)xlenth(mus)
    lbl has label for each different line
"""
function plotLines(lsAll,int,mus,lbl,xlbl,ylbl,svlbl,cm)
    lsAllM = hcat(mean(lsAll[int])...)
    lsAllStd = hcat(my_std(lsAll[int])...)
   
    plot(mus,lsAllM,lw=3,label=lbl,legend=:outertopright,
        xlabel=xlbl,ylabel=ylbl,palette=cm,
        yerr = lsAllStd)
    savefig(plotsdir(path,svlbl))
end

#####
# Fitting models
#####
@. modelL(x, p) = p[1]*x+p[2]
function fitLinear(l,times)
    p0 = [1.0,0.0]
    fit = curve_fit(modelL,times,l, p0)
    return fit.param
end

@. modelE(x, p) = p[3].*x.^p[1].+p[2]
function fitExp(l,times)
    p0 = [0.5,0.0,1.0]
    fit = curve_fit(modelE,times,l, p0)
    return fit.param
end

@. modelSqrt(x, p) = p[1].*sqrt.(x).+p[2]
function fitSqrt(l,times)
    p0 = [0.0,0.0]
    fit = curve_fit(modelSqrt,times,l, p0)
    return fit.param
end
@. modelInv(x, p) = p[1]./x.+p[2]
function fitInv(l,times)
    p0 = [0.0,0.0]
    fit = curve_fit(modelInv,times,l, p0)
    return fit.param
end



"""
plot mean of v vs Ns with label lbl and save at saveLbl
plot fit as well
"""
function plotVWithFit(v,Ns,l,xlbl,ylbl,saveLbl,fit=:linear)
    function getP(f,vm,Ns,l)
        if length(l) == 1 && length(vm)>1
            p = [f(vm,Ns)]
            println(p)
        else
            p = map(vm) do vmm 
                f(vmm,Ns)
            end
        end
        return p
    end
    my_colors = ["#D43F3AFF", "#5CB85CFF", "#EEA236FF", "#46B8DAFF","#357EBDFF", "#9632B8FF", "#B8B8B8FF"]
    vm = mean(v)
    vss = my_std(v)
    if fit==:linear
        md = modelL
        p = getP(fitLinear,vm,Ns,l)
        lbl = ["fit: $(@sprintf("%.3f", p[i][1]))k+$(@sprintf("%.3f", p[i][2]))" for i=1:length(p)]
    elseif fit==:exp
        p = getP(fitExp,vm,Ns,l)
        md = modelE
        lbl = ["fit: $(@sprintf("%.3f", p[i][3]))k^$(@sprintf("%.3f", p[i][1]))+$(@sprintf("%.3f", p[i][2]))" for i=1:length(p)]
    elseif fit==:sqrt
        p = getP(fitSqrt,vm,Ns,l)
        md = modelSqrt
        lbl = ["fit: $(@sprintf("%.3f", p[i][1]))sqrt(k)+$(@sprintf("%.3f", p[i][2]))" for i=1:length(p)]
    elseif fit==:inv
        p = getP(fitInv,vm,Ns,l)
        md = modelInv
        lbl = ["fit: $(@sprintf("%.3f", p[i][1]))/k+$(@sprintf("%.3f", p[i][2]))" for i=1:length(p)]

    end

    # plot(Ns,vm,yerr=vss,seriestype=:scatter,label=l)
    # if fit!=:false
    #     plot!(Ns,md(Ns,p),lw=3,label=lbl)
    # end
    if length(l) == 1 && length(vm) > 1# if only one line to plot
        plot(Ns,vm,yerr=vss,lw=3,label=l[1],color=my_colors[1],seriestype=:scatter,legend=:outertopright,grid=false)
        if fit!=:false
            plot!(Ns,md(Ns,p[1]),lw=3,color=my_colors[1],label=lbl[1])
        end
    else
        plot(Ns,vm[1],yerr=vss[1],lw=3,label=l[1],color=my_colors[1],seriestype=:scatter,legend=:outertopright,grid=false)
        if fit!=:false
            plot!(Ns,md(Ns,p[1]),lw=3,color=my_colors[1],label=lbl[1])
        end
        if length(vm)>1
            for i=2:length(vm)
                plot!(Ns,vm[i],yerr=vss[i],lw=3,color=my_colors[i],seriestype=:scatter,label=l[i])
                if fit!=:false
                    plot!(Ns,md(Ns,p[i]),lw=3,color=my_colors[i],label=lbl[i])
                end
            end
        end
    end
    plot!(xlabel=xlbl)
    plot!(ylabel=ylbl)
    savefig(saveLbl) 
end


###############
# Calculate theoretical values 
###############

""" Calculate optimal network size for learning speed
given parameters in postStore for net with size N 
use values at interval of trainign given by interval 
"""
function getNopt(postStore,N,mu,rho,interval,s=:lms)
    # return ((2/(1+rho)).*(abs.(postStore[:gradCorr_gradO]).*postStore[:norm_gradO].*postStore[:norm_grad])./(gamma.*postStore[:hessProj_gradO].*postStore[:norm_gradO].^2)).^(1/rho)
    return N*(((1+rho)/2).*(abs.(postStore[Symbol(:gradCorr_,s)][interval]).*postStore[Symbol(:norm_,s)][interval].*postStore[:norm_grad][interval])./(mu.*postStore[Symbol(:hessProj_,s)][interval].*postStore[Symbol(:norm_,s)][interval].^2)).^(1/rho)
end

""" Calculate optimal network size for steady state error 
given parameters in postStore for net with size N 
use values at interval of trainign given by interval 
"""
function getNoptSS(postStore,N,mu,gamma,rho,interval,tr_N,s=:lms)
    # return ((2/(1+rho)).*(abs.(postStore[:gradCorr_gradO]).*postStore[:norm_gradO].*postStore[:norm_grad])./(gamma.*postStore[:hessProj_gradO].*postStore[:norm_gradO].^2)).^(1/rho)
    return N*(1/rho.*((gamma/mu)^2).*(tr_N[interval]./(postStore[Symbol(:hessProj_,s)][interval].*postStore[Symbol(:norm_,s)][interval].^2))).^(1/(rho+1))
end


""" Calculate optimal learning step 
    given parameters in postStore 
    use values at interval of training given by interval 
"""
function getmuOpt(postStore,interval,T=200,s=:lms)
    # return ((2/(1+rho)).*(abs.(postStore[:gradCorr_gradO]).*postStore[:norm_gradO].*postStore[:norm_grad])./(gamma.*postStore[:hessProj_gradO].*postStore[:norm_gradO].^2)).^(1/rho)
    return (abs.(postStore[Symbol(:gradCorr_,s)][interval]).*postStore[Symbol(:norm_,s)][interval].*postStore[:norm_grad][interval])./(postStore[Symbol(:hessProj_,s)][interval].*postStore[Symbol(:norm_,s)][interval].^2)
end


"""
    Compute the static learning speed 
    νₜ = -ΔFₜ/(δₜFₜ)
    at each training step 
"""
function computeStaticLs!(postStore,s,t_save)
    dt = hcat(vcat(1,t_save[2:end]-t_save[1:end-1])...)
    # postStore[Symbol(:staticLS_,s)] = -postStore[Symbol(:localTaskDifficulty_,s)][1,:]./(dt.*postStore[:taskError][1,:])
    postStore[Symbol(:staticLS_,s)] = -postStore[Symbol(:localTaskDifficulty_,s)]./(dt.*postStore[:taskError])
    # return postStore
end

"""
    Compute the static local task diff
    Gₜ = 1/2*γ*(||lmsₜ||/||∇Fₜ||)lmŝ∇²Fₜlmŝ
    at each training step 
"""
function computeStaticLT!(p,s,gamma)
    p[Symbol(:staticLT_,s)] = 1/2*gamma.*(p[Symbol(:norm_,s)]./p[:norm_grad]).*p[Symbol(:hessProj_,s)]
end

"""
    Compute dynamic ss error (mean over last i values of task error)
"""
function computeDynamicSS!(p,i=5,max=false)
    if max==true
        p[:dynamicSS] = maximum(p[:taskError])
    else 
        p[:dynamicSS] = mean(p[:taskError][1,end-i:end])
    end
end

"""
    Compute dynamic ls mean of νₜ = -ΔFₜ/(δₜFₜ) (measure change in task loss) over the firs i epochs 
"""
function computeDynamicLs!(p,t_save,i=5)
    dt = hcat(vcat(1,t_save[2:end]-t_save[1:end-1])...)
    dF = hcat(vcat(0,p[:taskError][1,2:end]-p[:taskError][1,1:end-1])...)
    # ls = -dF[1,:]./(dt[1,:].*p[:taskError][1,:])
    ls = -dF[1,:]./(dt[1,:])
    p[:dynamicLs] = mean(ls[3:3+i])
end


""" 
    decompose vec t[s] into the direction of p[s2] plus noise
    hat{vec} = γ₁hat{ref} + γ₂hat{n}
    set in postStore p [:n_s], the noise and :gamma2_s the gamma2 
"""
function computeDecomp!(p,t,s,s2=:grad)
    gamma1, gamma2, n = decompose(t[s],p[s2])
    p[Symbol(:n_,s)] = n 
    p[Symbol(:gamma2_,s)] = gamma2
    p[Symbol(:gamma1_,s)] = gamma1
end

function computeDecomp(p,t,s,s2=:grad)
    gamma1, gamma2, n = decompose(t[s],p[s2])
    pp = Dict()
    pp[Symbol(:n_,s)] = n 
    pp[Symbol(:gamma2_,s)] = gamma2
    pp[Symbol(:gamma1_,s)] = gamma1
    return pp
end

""" 
    compute normalized dot product between vec[:,i] and ref[:,i] 
    for each i return dot product
"""
function dp(vec,ref)
    dp = zeros(1,size(vec,2))
    for i=1:size(vec,2)
        dp[:,i] .= normalize(vec[:,i])'*normalize(ref[:,i])
    end
    return dp 
end

function computeDot(p,t,s,s2=:grad)
    return dp(t[s],p[s2])
end

""" 
    compute hess proj 
"""
function hessP(vec,hess)
    hp = zeros(1,size(vec,2))
    N = length(vec[:,1])
    for i=1:size(vec,2)
        hp[:,i] .= normalize(vec[:,i])'*reshape(hess[:,i],(N,N))*normalize(vec[:,i])
    end
    return hp 
end

function computeHP(p,t,s,s2=:hessian)
    return hessP(t[s],p[s2])
end


"""
    function to generate updates from the parameters in d and weights w
"""
function makeComputeUpdate(d::Dict,W,trajErrorIndex=2)
    @unpack mus, gammas, deltaTes, deltaTrs, deltaThs, t_train, method = d
    if method == "onlineGrad+Noise"
        a = OnlineGradientCompute(mus,t_train,deltaTes,deltaTrs,W);
        b = GaussianNoiseCompute(gammas,t_train,W);
        updates =[a,b]
    elseif method == "onlineGrad"
        a = OnlineGradientCompute(mus,t_train,deltaTes,deltaTrs,W);
        updates =[a]
    elseif method == "lms"  
        updates = [LMSCompute(mus,t_trains,deltaTes,deltaTrs,deltaThs,trajErrorIndex,W)]
    end
    fulld = copy(d)
    fulld["updates"] = updates
    return fulld
end

"""
analyse a simulation with trainstore t and poststore p 
to get the correlation and hess proj
"""
function analysePost(t,p,method="onlineGrad+Noise")
    if method=="onlineGrad+Noise"||method=="onlineGrad"
        s=:gradO
    elseif method=="lms" 
        s=:lms 
    end
    # compute gamma^oe and eta^oe (proj of gradO onto grad)
    pp = computeDecomp(p,t,s,:grad) 
    # compute grad correlation dw^T\nabla F 
    pp[Symbol(:corr_,:dw)] = computeDot(p,t,:dw,:grad)
    # compute hessian projection dw^THdw, and gradO^T H gradO 
    pp[Symbol(:hessProj_,:dw)] = computeHP(p,t,:dw)
    pp[Symbol(:hessProj_,s)] = computeHP(p,t,s)
    if haskey(t,:noise)
        pp[Symbol(:hessProj_,:noise)] = computeHP(p,t,:noise)
    end
    pp[Symbol(:hessProj_,:n_,s)] = computeHP(p,pp,Symbol(:n_,s))
    return pp
end