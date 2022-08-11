###
# Defines components for the motor control system 
###
abstract type InSimulationOperations end
abstract type PlasticityOperations <: InSimulationOperations end
abstract type ComputePlasticityOperations <: InSimulationOperations end
abstract type PostSimulationAnalysis end
abstract type PostSimulationReadout end
abstract type Plant end
abstract type TrainingProblem end

struct RandomInitialization end

""" 
System: trajectory tracking system with plant, fb controller, cerebellar-like net and reference
"""
struct System{A <: Plant,B,C,D,E,F,T,G}
    plant::A 
    pid::B
    nn::C
    ref::D
    trajTime::E
    lookahead_times::F
    system::T
    trainErrorIndex::G # integral of mse (compute task loss wrt to this)
    onlineErrorIndex::G # integral of e(t) = y(t)-r(t) for online training
    outputIndex::G # index of output in the system ode
end

""" 
Consturctor of the system given plant matrices, pidgains, neural network, reference function 
"""
function System(plantMatrices,pidGains,nn,refF,trajTime,lookahead_times)
    plant = LTIPlant(plantMatrices) # create plant
    pid = PID(pidGains) # create PID controller from the given gains 
    # refSin = sinFunDef(fc) # superposition of sins with cutoff frec fc
    ref = Reference(refF,lookahead_times) # reference system from the function refF
    N = nn.dims[2] # number of granule cells (i.e. adaptable parameters in the nn)
    all = connectFFFB(plant.system,ref.system,pid.system,nn.system,lookahead_times,N) # connect different systems for ODE
    system = structural_simplify(all) # simplify ODE system 
    System(plant,pid,nn,ref,trajTime,lookahead_times,system,1,1,3) # initialize System with default indeces
end

""" 
Consturctor of the system given plant matrices, pidgains, dimensions of the neural net (no nn given)
"""
function System(plantMatrices,pidGains,nnDims,K,refF,trajTime,lookahead_times)
    nn = NeuralNetwork(RandomInitialization(),nnDims,K)   # initialize NN with random weights
    System(plantMatrices,pidGains,nn,refF,trajTime,lookahead_times)
end

""" 
Consturctor of the system given plant matrices, pidgains, dimensions of the neural net
and the input Z and output weight W matrices of the neural net
"""
function System(plantMatrices,pidGains,nnDims,refF,trajTime,lookahead_times,Z,W)
    nn = NeuralNetwork(nnDims,Z,W)  # initialize NN with given weights
    System(plantMatrices,pidGains,nn,refF,trajTime,lookahead_times)
end

""" 
function to instantiate a system 
"""
function build_system(plantMatrices,Ks,nnDims,K,refF,trajTime,lookahead_times)
    System(plantMatrices,Ks,nnDims,K,refF,trajTime,lookahead_times)
end

""" 
function to instantiate a system with the weights matrices for NN
"""
function build_system(plantMatrices,pidGains,nnDims,refF,trajTime,lookahead_times,Z,W)
    System(plantMatrices,pidGains,nnDims,refF,trajTime,lookahead_times,Z,W)
end


""" 
function to instantiate a system given some plant, pid and ref already instantiated
"""
function build_system(plant::Plant,pid,nnDims,ref,trajTime,lookahead_times,Z,W)
    nn = NeuralNetwork(nnDims,Z,W)  
    all = connectFFFB(plant.system,ref.system,pid.system,nn.system,lookahead_times, nn.dims[2])
    system = structural_simplify(all)
    System(plant,pid,nn,ref,trajTime,lookahead_times,system,1,1,3)
end


""" 
InputSystem: system with only reference generator and neural net
"""
struct InputSystem{C ,D ,E ,F}
    nn::C ## flux net with two layers
    ref::D ## function generating input at time t
    trajTime::E
    lookahead_times::F
end
""" 
Consturctor of the system given neural network... 
"""
function InputSystem(nn,fc::Float64,trajTime,lookahead_times)
    refSin = sinFunDef(fc) # superposition of sins with cutoff frec fc
    # ref = Reference(refSin,lookahead_times)
    InputSystem(nn,refSin,trajTime,lookahead_times)
end
""" 
Consturctor of the system given dimensions of the neural net
"""
function InputSystem(nnDims,K,fc,trajTime,lookahead_times)
    Z = createZ(nnDims[1],nnDims[2],K)
    d = Normal(0,1/sqrt(nnDims[2]))
    W = rand(d,(nnDims[3],nnDims[2]))
    InputSystem(nnDims,fc,trajTime,lookahead_times,Z,W)
end
""" 
Consturctor of the system given dimensions of the neural net
and the input Z and output weight W matrices of the neural net
"""
function InputSystem(nnDims,fc,trajTime,lookahead_times,Z,W)
    nn = Chain(
        Dense(Z, true, tanh),
        Dense(W,false,identity)
    )
    InputSystem(nn,fc,trajTime,lookahead_times)
end
""" 
function to instantiate an input system 
"""
function build_inputSystem(nnDims,K,fc,trajTime,lookahead_times)
    InputSystem(nnDims,K,fc,trajTime,lookahead_times)
end
""" 
function to instantiate an input system 
"""
function build_inputSystem(nnDims,fc::Float64,trajTime,lookahead_times,Z,W)
    InputSystem(nnDims,fc,trajTime,lookahead_times,Z,W)
end
""" 
function to instantiate a system given some plant, pid and ref already instantiated
"""
function build_inputSystem(nnDims,ref,trajTime,lookahead_times,Z,W)
    nn = Chain(
        Dense(Z, true, tanh),
        Dense(W,false,identity)
    )
    InputSystem(nn,ref,trajTime,lookahead_times)
end

""" 
NeuralNetwork: cerebellar like network with one hidden layer
"""
struct NeuralNetwork{A,B,C,D,T}
    dims::A
    Z::B
    W::C
    fluxNet::D
    system::T
end
""" 
Neural net constructor for random initialisation with dim and K input degree of each hidden unit
"""
function NeuralNetwork(::RandomInitialization, nnDims,K)
    Z = createZ(nnDims[1],nnDims[2],K)
    d = Normal(0,1/sqrt(nnDims[2]))
    W = rand(d,(nnDims[3],nnDims[2]))
    NeuralNetwork(nnDims,Z,W)
end
""" 
Neural net constructor given input and output weights
"""
function NeuralNetwork(nnDims,Z,W)
    nn = Chain(
        Dense(Z, true, tanh),
        Dense(W,false,identity)
    )
    mlp = MLP_controller(nn) # make ODE system of the nn
    NeuralNetwork(nnDims,Z,W,nn,mlp)
end

"""
get all weights of the flux nn network
"""
function getAllWeights(nn::NeuralNetwork)
    pAll,re = Flux.destructure(nn.fluxNet)
    return pAll
end
"""
get input weights of the flux nn network
"""
function getInputWeights(sys::System) 
    getInputWeights(sys.nn)
end
function getInputWeights(nn::NeuralNetwork)
    pI,re = Flux.destructure(nn.fluxNet[1:end-1])
    return pI
end
"""
get number of input weights of the neural net
"""
function getNumInputWeigts(nn::NeuralNetwork)
    pI, reInput = Flux.destructure(nn.fluxNet[1:end-1]) # input weights
    length(pI)
end
"""
get number of hidden units N of the neural net of a syst
"""
function getN(syst::System)
    getN(syst.nn)
end
"""
get number of hidden units N of the neural net nn
"""
function getN(nn::NeuralNetwork)
    nn.dims[2]
end

""" 
Reference: generates reference trajectory and lookahead vector
"""
struct Reference{A,S,T}
    func::A 
    lookahead_times::S
    system::T
end
""" 
Reference: constructor given the function func(t) gives the reference at time
lookahead_times the array of times to generate the inputs to the neual net
"""
function Reference(func,lookahead_times)
   syst = create_ref(func,lookahead_times)
   Reference(func,lookahead_times,syst) 
end

""" 
Reference: given parameters for O-U process with filtering
lookahead_times the array of times to generate the inputs to the neual net
"""
function Reference(lookahead_times,θ,sigma,μ,T,dt,fc)
    func = ouFunDef(θ,sigma,μ,T,dt,fc)
    syst = create_ref(func,lookahead_times) 
    Reference(func,lookahead_times,syst) 
end 

""" 
Plant linear plant with state space given by plant matrices
"""
struct LTIPlant{A,T} <: Plant
    plantMatrices::A
    system::T
end
function LTIPlant(plantMatrices)
    sys = linear_plant(plantMatrices...)
    LTIPlant(plantMatrices,sys)
end

""" 
PID controller with gains = (Pgain, Igain, Dgain, RC)
"""
struct PID{A,T} 
    gains::A
    system::T
end
function PID(gains::Tuple)
   PID(gains, 1)
end
function PID(gains::Tuple,num_inputs::Int64)
    sys =  PID_controller(gains...,num_inputs)
    PID(gains, sys)
end
