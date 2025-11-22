###############################################################################
# Import required packages
import numpy as np
###############################################################################

###############################################################################
def stoch_dyn_CSA(states):
    
    """
    Function that simulates stochastic colloidal self-assembly dynamics
      
    states --> [xk, uk, xkw], shape = (3,1) or (3,)
    
    xk --> system state (C6)
    
    uk --> exogenous input (electric field voltage)
    
    xkw --> Gaussian white noise ~ N(0,1)
    
    """
    # Sampling time (s)
    dt = 1
    
    # Distribute states
    xk = states[0]
    uk = states[1]
    xkw = states[2]
    
    # Get diffusion coefficient
    g2 = 0.0045*np.exp(-(xk-2.1-0.75*uk)**2)+0.0005
    
    # Get drift coefficient
    #    F/KT = 10*(x-2.1-0.75*u)**2
    dFdx = 20*(xk-2.1-0.75*uk)
    dg2dx = -2*(xk-2.1-0.75*uk)*0.0045*np.exp(-(xk-2.1-0.75*uk)**2)
    g1 = dg2dx-g2*dFdx
    
    # Predict forward dynamics
    xkp1 = xk + g1*dt + np.sqrt(2*g2*dt)*xkw
    
    return [np.asarray([xkp1]), 
            np.asarray([g1]), 
            np.asarray([g2])]
###############################################################################
    
###############################################################################
def stoch_dyn_LVE(states):
    
    '''
    Function that simulates stochastic competitive Lotka-Volterra dynamics
    with coexistence equilbirum
    
    states = [xk, yk, xkw, ykw], shape = (4,1) or (4,)
    
    xk, yk --> species populations
    
    xkw, ykw --> independent Guassian white noise processes, ~ N(0,1)
    
    '''
    
    # Sampling time (s)
    dt = 0.01
    
    # Distribute states
    xk = states[0]
    yk = states[1]
    xkw = states[2]
    ykw = states[3]
    
    # Enter parameters
    k1 = 0.4
    k2 = 0.5
    xeq = 0.75
    yeq = 0.625
    d1 = 0.5
    d2 = 0.5
    
    # Get drift coefficients
    g1x = xk*(1 - xk - k1*yk)
    g1y = yk*(1 - yk - k2*xk)
    
    # Get diffusion coefficients
    g2x = 1/2*(d1*xk*(yk-yeq))**2
    g2y = 1/2*(d2*yk*(xk-xeq))**2
    
    # Predict forward dynamics
    xkp1 = xk + g1x*dt + np.sqrt(2*g2x*dt)*xkw
    ykp1 = yk + g1y*dt + np.sqrt(2*g2y*dt)*ykw
    
    return [np.asarray([[xkp1], [ykp1]]), 
            np.asarray([[g1x], [g1y]]), 
            np.asarray([[g2x], [g2y]])]
###############################################################################

###############################################################################
def stoch_dyn_SIR(states):
    
    '''
    Function that simulates stochastic Susceptible-Infectious-Recovered (SIR)
    dynamics
    
    states = [sk, ik, rk, skw, ikw, rkw], shape = (6,1) or (6,)
    
    sk, ik, rk --> susceptible, infectious, recovered populations
    
    skw, ikw, rkw --> independent Guassian white noise processes, ~ N(0,1)
    
    '''
    
    # Sampling time (s)
    dt = 1
    
    # Distribute states
    sk = states[0]
    ik = states[1]
    rk = states[2]
    skw = states[3]
    ikw = states[4]
    rkw = states[5]
    
    # Enter parameters
    b = 1
    d = 0.1
    k = 0.2
    alpha = 0.5
    gamma = 0.01
    mu = 0.05
    h = 2
    delta = 0.01
    sigma_1 = 0.2
    sigma_2 = 0.2
    sigma_3 = 0.1
    
    # Get nonlinear incidence rate
    g = (k*sk**h*ik)/(sk**h+alpha*ik**h)
    
    # Get drift coefficients
    g1s = b-d*sk-g+gamma*rk
    g1i = g-(d+mu+delta)*ik
    g1r = mu*ik-(d+gamma)*rk
    
    # Get diffusion coefficients
    g2s = 1/2*(sigma_1*sk)**2
    g2i = 1/2*(sigma_2*ik)**2
    g2r = 1/2*(sigma_3*rk)**2
    
    # Predict forward dynamics
    skp1 = sk + g1s*dt + np.sqrt(2*g2s*dt)*skw
    ikp1 = ik + g1i*dt + np.sqrt(2*g2i*dt)*ikw
    rkp1 = rk + g1r*dt + np.sqrt(2*g2r*dt)*rkw
    
    return [np.asarray([skp1, ikp1, rkp1]), 
            np.asarray([g1s, g1i, g1r]), 
            np.asarray([g2s, g2i, g2r]),
            np.asarray([g]),
            np.asarray([b-d*sk+gamma*rk]),
            np.asarray([(d+mu+delta)*ik])]
###############################################################################

###############################################################################
# Additional stochastic models (Black–Scholes and Heston)
#
# The functions `stoch_dyn_BS` and `stoch_dyn_Heston` implement the
# Black–Scholes and Heston SDEs, respectively. They were added by
# J. Cruz Araneda and B. Lagos Guerra for numerical experiments and 
# are not part of the original implementation of this repository.
###############################################################################

###############################################################################
def stoch_dyn_BS(states):
    """
    Stochastic dynamics for the Black-Scholes (Geometric Brownian Motion) model.

    states = [Sk, Skw]
      Sk  --> asset price at time k (float)
      Skw --> Gaussian white noise ~ N(0,1)
    """
    dt = 1/252   # one trading day
    mu = 0.05    # drift (5%)
    sigma = 0.2  # volatility (20%)

    # Unpack states safely (handles np arrays or lists)
    Sk = float(states[0])
    Skw = float(states[1])

    # Drift coefficient
    g1 = mu * Sk

    # Diffusion coefficient
    g2 = 0.5 * (sigma * Sk)**2

    # Euler-Maruyama step
    Skp1 = Sk + g1 * dt + np.sqrt(2 * g2 * dt) * Skw

    # Return scalar arrays (1D, consistent with SPINODE)
    return [np.asarray([Skp1], dtype=np.float32),
            np.asarray([g1], dtype=np.float32),
            np.asarray([g2], dtype=np.float32)]
###############################################################################

###############################################################################
def stoch_dyn_Heston(states):
    """
    Stochastic dynamics for the Heston model (stochastic volatility model).

    states = [Sk, vk, Skw, vkw]
      Sk  --> asset price at time k
      vk  --> variance at time k
      Skw --> Gaussian noise for price ~ N(0,1)
      vkw --> Gaussian noise for variance ~ N(0,1)
    """
    dt = 1/252     # one trading day
    mu = 0.05      # drift of the asset
    kappa = 2.0    # rate of mean reversion
    theta = 0.04   # long-term variance (so vol ~ 20%)
    xi = 0.3       # volatility of volatility
    rho = -0.5     # correlation between W1 and W2

    # Unpack states safely
    Sk  = float(states[0])
    vk  = float(states[1])
    Skw = float(states[2])
    vkw = float(states[3])

    # Correlated Brownian increments
    dW1 = Skw
    dW2 = rho * Skw + np.sqrt(1 - rho**2) * vkw

    # Drift terms
    g1 = mu * Sk
    g2 = kappa * (theta - vk)

    # Diffusion terms
    sigma1 = np.sqrt(max(vk, 0.0)) * Sk   # for S_t
    sigma2 = xi * np.sqrt(max(vk, 0.0))   # for v_t

    # Euler–Maruyama updates
    Skp1 = Sk + g1 * dt + sigma1 * np.sqrt(dt) * dW1
    vkp1 = vk + g2 * dt + sigma2 * np.sqrt(dt) * dW2

    # Pack outputs
    g1_vec = np.asarray([g1, g2], dtype=np.float32)
    g2_vec = np.asarray([0.5 * sigma1**2, 0.5 * sigma2**2], dtype=np.float32)
    Xkp1   = np.asarray([Skp1, vkp1], dtype=np.float32)

    return [Xkp1, g1_vec, g2_vec]
###############################################################################