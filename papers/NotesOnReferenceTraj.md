# Defining the reference trajectory 

The reference trajectory is the $r(t), t \in [0,T]$ that the system needs to learn to track. 

In any such problem there are signals that the plant won't be able to produce because of the limitations of its dynamics. Biologically, this makes sense as there are some movements that will be too fast or to big for the musculoskeletal system to produce. 
We can express this limitation by a **cut-off frequency** $f_c$. We assume that any signal that doesn't have any frequencies above $f_c$, can be actuated by the plant. 

We want the reference signal to satisfy both:
- $r(t)$ can be produced by the plant (i.e. it is band limited to $f_c$ and its magnitude is bound by some values). 
- it is a rich and unbiased signal for the learning performance to be unbiased or independent of the reference signal. 

## Sum of sinusoidals 
Initially we chose a reference signal as follows. 

The reference trajectories $r(t)$ are a sum of $4$ sinusoids with random frequencies, phase shifts and amplitudes.
$$
    r(t) = \frac{1}{4}\sum_{i=1}^4 A_i \sin(\omega_i t+\phi_i)
$$
$\omega_i$ is drawn from a random uniform distribution in the interval $[\omega_c/5,\omega_c]$, $\omega_c$ being the cut-off frequency. We set $\omega_c=0.5Hz$. The phase shifts $\phi_i$ are drawn from a random uniform distribution in the interval $[0,\pi/2]$ and the amplitudes $A_i$ from a random uniform distribution in $[0.1,4]$.

All tracking problems must be band limited to take into account the limits of the plant dynamics. This cut-off frequency ensures that the plant is able to produce the reference trajectory. 

We shift the reference trajectory to the first point where it is equal to zero, effectively redefining the reference trajectory as $r(t) \to r(t+\tau)$, such that $r(0)=0$. This is important as the plant initial condition is set to zero. Hence, a non-zero initial reference would lead to an error that the system could not reduce. 


## Random signal 

We find that the sum of sinusoidals might be too biased as a signal. Hence, some learning steps might be optimal for learning given some reference signal. 

We opt for a random signal. We want the value of the signal to be bound between $r(t) \in [-1,1], \quad \forall t \in [0,T]$, and the signal to be band limited to have frequencies below $\omega_c$. 

We chose an Ornstein-Uhlenbeck process passed through a low pass filter. 
The definition of the O-U process is 
$$
dx_t = \theta(\mu -x_t)dt+\sigma dW_t
$$ 
where $\theta>0$, $\sigma>0$ and $\mu$ are parameters and $W_t$ denotes a Wiener process.
The O-U process is a modification of the random walk to have mean-reverting properties. 
The parameter $\theta$ controls how fast the process drifts towards its mean $\mu$. $\sigma$ controls the effect of the random shocks. 

We generate an O-U process from the DifferentialEquations.jl library with parameters $\theta =0.1$, $\mu=0.0$ and $\sigma = 0.02$. We "solve" the process with $dt=0.01$ and $W_0$ is drawn from a random uniform distribution $\mathcal{U}[-0.5,0.5]$. 
The value of the parameters are chosen empirically to have the trajectory between $[-1,1]$ and mean reversion around the desired frequencies. Different parameters could be chosen with similar outcome. 

The signal is then passed through a low-pass filter (implemented with the DSP.jl library) with cutoff-frequency $\omega_c \times dt$, where $\omega_c=0.2$ and $dt=0.01$ as above. 
Note that the filter has an inherent delay, hence the filtered signal will start at $r(0)=0$. 

<!-- ![O-U process and filtered signal](../plots/O-U_process/original_filtered.png) -->

![O-U process and filtered signal](../plots/O-U_process/original_filtered_2.png)

Finally, we need to interpolate the signal to make it into a continuous function of time to pass to the solver of the trajectory tracking problem. 


The only issue is the ODE solver for LMS training and computing the task loss and the gradient is a lot slower than with the sinusoidals. This is probably and issue with the interpolated signal passed to the system. The time grows with the length of the reference signal. It takes around 10h to train and test for 1000s of trajectory. 

From a Discourse answer, it seems that this slowing down is a known issue with DifferentialEquations.jl and Interpolations. 


We try changing the ODE solver to a fixed timestep $dt$ with the time-step matching the values of the now discrete reference signal. We need to produce the reference signal 

$$r(t),\quad t\in [0,dt,2dt,...,\delta t_I,...,T]$$