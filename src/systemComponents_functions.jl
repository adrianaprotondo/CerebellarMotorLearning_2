###
# helper functions for components of the system 
###
"""
get inputs to the system
u is reserved for inputs to any ODESystem
"""
function get_input(sys, i) 
    return sys.u[i]
end

"""
get inputs of system from index 1 to i 
below is broadcasting: each input should be an array/collection. 
the . operator then applies the function to each element of the array. 
We want the function to take the same sys each time, so we wrap it in a tuple: (sys,)
"""
get_inputs(sys, i) = get_input.((sys,), 1:i)

"""
get the outputs of the system 
o is reserved for outputs of any ODESystem
"""
function get_output(sys, i)
    return sys.o[i]
end

get_outputs(sys, i) = get_output.((sys,), 1:i)

"""
get the trajectory error of the system 
e is reserved for outputs of any ODESystem
"""
function get_error(sys, i)
    # return getproperty(sys,  Symbol(:e, Symbol('₀' + i)))
    return sys.e[i]
end

function get_errors(sys, i)
    get_error.((sys,), 1:i)
end


"""
get hidden layer of the NN
h is reserved for hidden layer of any mlp
"""
function get_hidden(sys,i)
    return sys.h[i]
end
function get_hiddens(sol,i)
    get_hidden.((sol,),1:i)
end
"""
get hidden layer activity of the nn in sys
from the  ODEsolution sol 
"""
function get_hidden(sol,sys,i,j)
    if j==0
        # return sol[getproperty(sys,name),end]
        return sol[sys.h[i],end]
    else
        # return sol[getproperty(sys,name),j]
        return sol[sys.h[i],j]
    end
end
function get_hiddens(sol,sys,i,j=0)
    map(1:i) do ii
        get_hidden(sol,sys,ii,j)
    end
    # return sol[sys.h[1:i],]
end

"""
    make reference function of sum of numSin sinusoidals
        with random frequency band limited to fc 
        and random phaseShifts and amplitudes
"""
function sinFunDef(fc,numSin=4,dt=0.001)
    omegas = rand(fc/(numSin+1):dt:fc,numSin)
    # phaseDif = zeros(numSin)
    phaseDif = rand(0:dt:pi/2,numSin)
    amp = rand(0.1:dt:1,numSin)
    function refSin(t)   # sum of sinusoidals 
        # sum(1/numSin*sin.(omegas.*t.+phaseDif))
        sum(1/numSin*amp.*sin.(omegas.*t.+phaseDif))
    end
    if phaseDif[1]!=0 # if non-zero phaseDif (translate to start at zero)
        ts = 0:dt:1000
        i = findfirst(refSin.(ts).<1e-5)
        t0 = ts[i]
        function refSinT(t)   # translation to start at zero
            refSin(t+t0)
        end 
        return refSinT
    else
        return refSin
    end
end

"""
Interpolate a series and return the interpolated value
"""
function interpolateSeries(t,time,values)
    # interpolator = CubicSplineInterpolation(time, values)
    interpolator = LinearInterpolation(time, values)
    interpolator(t)
end

# register the function for ModelingToolkit.jl to work
@register interpolateSeries(t, time::AbstractVector, values::AbstractVector)

"""
manual implementaition of linear interpolation
"""
function linInt(t,tInt,y)
    # find the index closest values to t in tInt: t1<t<t2 
    # get y values corresponding y1, y2
    # use liner interpolation y(t) = y1+(t-t1)*(y2-y1)/(t2-t1)
    i2 = findfirst(t.<tInt)
    i1 = i2-1
    return y[i1]+(t-tInt[i1])*(y[i2]-y[i1])/(tInt[i2]-tInt[i1])
end

@register linInt(t, tInt::AbstractVector, y::AbstractVector)


"""
create random reference from O-U process filtered and interpolatedValue
"""
function ouFunDef(θ,sigma,μ,T,dt,fc,w0S=0.5)
    # ornstein uhlenbeck process
    t0 = 0.0
    w0 = w0S*(rand()-0.5)
    W = OrnsteinUhlenbeckProcess(θ,μ,sigma,t0,w0)
    prob = NoiseProblem(W,(t0,T))
    sol = DifferentialEquations.solve(prob;dt=dt)

    # filter
    responsetype = Lowpass(fc) # type of filter
    designmethod = Butterworth(3) # window type
    filteredX = filt(digitalfilter(responsetype, designmethod), sol.u)

    function refF(t)
        linInt(t,sol.t,filteredX)
        interpolateSeries(t,sol.t,filteredX)
    end
    return refF
end

"""
ie if we call r(t) we get r.func(t)
"""
function (r::Reference)(t)
    r.func(t)
end

"""
ie expanded(r,t) = vector of outputs r(t .+ lookahead times)
"""
expanded(r::Reference, t::Num) = r.func.(t .+ collect(r.lookahead_times))
expanded(f,lookahead_times,t::Num) = f.(t .+ collect(lookahead_times))
expanded(r, t) = r.func.(t .+ r.lookahead_times)

"""
    make ModelingToolkitized reference controller
"""
function create_ref(func,lookahead_times; name = :ref)
    @parameters t 
    @variables o[1:length(lookahead_times)+1](t)
    eqs = vcat(
        o[1] ~ func(t),
        [o[i+1] ~ func(t+lookahead_times[i]) for i in 1:length(lookahead_times)]
        # collect(o[2:end] .~ expanded(func,lookahead_times,t))
    )
    ODESystem(eqs; name=name)
end



"""
    make ModelingToolkitized controller out of Flux neural network given as input
"""
function MLP_controller(nn; n=:MLP)
    @parameters t
    pAll, re = Flux.destructure(nn) # get all parameters and I-O function of the NN 
    pOutput, reOutput = Flux.destructure(nn[2:end]) # get output weights and hidden layer to output function
    pInput, reInput = Flux.destructure(nn[1:1]) # get input weights and input to hidden layer function
    input_dims = size(nn.layers[1].weight,2);  # number of inputs of flux nn
    output_dims = size(nn.layers[2].weight,1);  # number of outputs of flux nn
    N = size(nn.layers[1].weight,1); # number of hidden layers
    @parameters psymI[1:length(pInput)]
    @parameters psymO[1:length(pOutput)]
    @variables u[1:input_dims](t)
    @variables o[1:output_dims](t)
    @variables h[1:N](t)
    eqs = vcat(
        scalarize(h .~ reInput(psymI)(u)), # hidden layer variable from the flux net 
        scalarize(o .~ reOutput(psymO)(h)) # output from flux net
    )
    return ODESystem(eqs, t, vcat(u,o,h), vcat(psymI,psymO); name=n, defaults = Dict(vcat(psymI,psymO) .=> vcat(pInput,pOutput)))
    # return ODESystem(eqs,t,vcat(u,o,h),vcat(psymI,psymO))
end

"""
    make ModelingToolkitized linear plant from matrices
"""
function linear_plant(A,B,C,D; name=:lp)
    num_inputs = size(B,2)
    num_states = size(A,1)
    num_outputs = size(C,1)

    @parameters t
    dt = Differential(t)

    @variables x[1:num_states](t)
    @variables u[1:num_inputs](t)
    @variables o[1:num_outputs](t)
    
    # equations for the linear plant from the matrices
    eqs = vcat(
                dt.(x) .~ A*x .+ B*u,
                o .~  C*x .+ D*u
    )    
    return ODESystem(eqs; name=name)   
end

"""
    make ModelingToolkitized pid controller 
"""
function PID_controller(Pgain, Igain, Dgain, RC, num_inputs=1; name=:pid)
    @parameters t
    D = Differential(t)

    @variables u[1:num_inputs](t)
    @variables o[1:num_inputs](t)
    @variables int[1:num_inputs](t)
    @variables mv[1:num_inputs](t)
    eqs = vcat(
        D.(int) .~ Igain.*u, #integrator
        D.(mv) .~ u - RC.*(mv), # low pass filter 
        o .~ Dgain .* (u - RC.*(mv)) + Igain.*int + Pgain.*u
        # o .~ Dgain .* u + Igain.*int + Pgain.*u
    )
    return ODESystem(eqs; name = name)
end

"""
createZ(num_nn_inputs,N,K)
creates an input weight matrix for the NN
    the matrix has num_nn_inputs input units, N hidden units
    each hidden unit recieves K inputs
    each row of Z has K non-zero elements with values drawn from gaussian ditribution 
        with mean 0 and variance 1/num_nn_inputs 
"""
function createZ(num_nn_inputs,N,K)
    # generate an input weights matrix with I inputs, projecting onto N units
    # each of the N units receive K inputs at random form the I 
    # requires K<=num_nn_inputs
    # dI = Normal(0,1/sqrt(num_nn_inputs))
    dI = Normal(0,1/sqrt(K))
    Z0 = zeros(N,num_nn_inputs)
    for i=1:N
        Z0[i,:] = rand(dI,num_nn_inputs)
    end
    # Z0 = rand(dI,(N,num_nn_inputs)) # input weights
    allComb = collect(combinations(1:num_nn_inputs,K))
    granule_mix = allComb[rand(1:size(allComb,1),N)] # select a random combination of K inputs for each GC
    # granule_mix = rand(1:num_nn_inputs,(N,K))
    Z = zeros(N,num_nn_inputs)
    for i=1:N
        Z[i,granule_mix[i]] = Z0[i,granule_mix[i]] # set non-zero components to random weight
    end
    return Z
end

"""
connectFFFB(plant, ref, pid, mlp,lookahead_times)
returns an ODESystem representing system with mlp and pid controlling plant to track ref traj
"""
function connectFFFB(plant, ref, pid, mlp, lookahead_times, N; name=:FFFBsys)   
    @parameters t 
    dt = Differential(t)
    @variables intError(t)
    @variables e(t)
    @variables eI(t)
    # @variables lms(t)
    eqs = vcat(
            get_input(pid,1) ~ get_output(ref, 1)  - get_output(plant,1),
            get_output.((ref,), 2:length(lookahead_times)+1) .~ get_inputs(mlp,length(lookahead_times)),
            get_input(plant,1) ~ get_output(pid,1) + get_output(mlp,1),
            e ~  get_output(plant,1)-get_output(ref,1), # error is plant output-ref 
            dt(intError) ~  1/2*(e)^2, # for gradient calculation
            dt(eI) ~ e, # error integral
            # dt(lms) .~ e.*get_hiddens(mlp,N),
            # e ~ abs2(get_output(ref,1)  - get_output(plant,1)),
    ) 
    return ODESystem(eqs;systems=[plant,ref,pid,mlp],name=name)
end


"""
connectInputSystem(nn, ref, lookahead_times)
returns the functions that returns the input, hidden layer activity and output given some times
"""
function connectInputSystem(system::InputSystem)   
    function sysInput(t)
        # [system.ref(t),expanded(system.ref,system.lookahead_times,t)]
        system.ref.(t.+collect(system.lookahead_times))
    end
    function sysIAll(t_train)
        hcat(sysInput.(t_train)...)
    end
    function sysHidden(t)
        pInput, reInput = Flux.destructure(system.nn[1:1])
        reInput(pInput)(sysInput(t))
    end
    function sysHAll(t_train)
        hcat(sysHidden.(t_train)...)
    end
    function sysOutput(t)
        system.nn(sysInput(t))
    end 
    function sysOAll(t_train)
        hcat(sysOutput.(t_train)...)
    end
    return sysIAll,sysHAll,sysOAll
end