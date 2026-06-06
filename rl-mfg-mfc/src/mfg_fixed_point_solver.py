# Iterative fixed-point solver for theoretical z-values (mean investments) using the formulas from the screenshot
# Uses the same model functions you provided earlier. m0 is set by the user below.
import numpy as np
import pandas as pd

# Model parameters (taken from your code snippet)
maxv = 12.0
step = 0.05
T = 4
gamma = 0.2
rho = 0.95
C = 3

supp_W = np.array([0.9, 1.3])
pmf_W = np.array([0.75, 0.25])

# compute pEWgamma = rho * E[ W^gamma ]
exp_W_gamma = np.sum(np.power(supp_W, gamma) * pmf_W)
pEWgamma = rho * exp_W_gamma

# G(z, W) as used in your env (depends on z scalar and noise W)
def G_of_z_w(z, w):
    return C * w / (pEWgamma * (1 + (C - 1) * (z ** 3)))

# Φ(z) = rho * E[ G(z,W)^gamma ]
def Phi(z):
    vals = np.pow(G_of_z_w(z, supp_W), gamma)
    return rho * np.dot(vals, pmf_W)

# φ(z) = Φ(z)^{1/(γ - 1)}  (note gamma < 1 so exponent is negative)
def phi_small(z):
    ph = Phi(z)
    if ph <= 0:
        return 0.0
    return ph ** (1.0 / (gamma - 1.0))

# Ψ(z) = E[ G(z, W) ]
def Psi(z):
    vals = G_of_z_w(z, supp_W)
    return np.dot(vals, pmf_W)

# Build Lambda vector from current z vector according to the boxed formulas in the screenshot
def Lambda(z_vec, m0):
    # z_vec length should be T, representing z0..z_{T-1}
    phis = [phi_small(z) for z in z_vec]    # φ(z_i)
    psis = [Psi(z) for z in z_vec]          # Ψ(z_i)
    
    # build list of cumulative products of φ from T-1 downwards for convenience
    # products_list[j] = φ_{T-1} * φ_{T-2} * ... * φ_{j}  (for j from 0..T-1)
    products_list = []
    for j in range(T):
        prod = 1.0
        for r in range(T-1, j-1, -1):
            prod *= phis[r]
        products_list.append(prod)
    
    # The denominator appearing in all Λ_k: 1 + φ_{T-1} + φ_{T-1}φ_{T-2} + ... + φ_{T-1}...φ_0
    denom_terms = [1.0] + [np.prod(phis[T-1: T-1-i:-1]) for i in range(1, T+0)]  # construct progressively
    # A clearer way: build powers by progressively multiplying from φ_{T-1} down
    denom = 1.0
    running = 1.0
    for r in range(T-1, -1, -1):
        running *= phis[r]
        denom += running
    
    Lamb = np.zeros(T)
    # Λ_0: numerator = 1 + φ_{T-1} + ... + φ_{T-1}...φ_1  (i.e., exclude the final product that includes φ_0)
    # equivalent: denom - (φ_{T-1}...φ_0)
    last_prod_all = np.prod(phis) if T>0 else 1.0
    numer_L0 = denom - last_prod_all
    Lamb[0] = (numer_L0 / denom) * m0
    
    # For k = 1..T-2:
    # Λ_k := (1 + φ_{T-1} + ... + φ_{T-1}...φ_{k+1}) / denom * Ψ(z_{k-1}) ... Ψ(z_0) * m0
    # The numerator is denom minus the tail products that include φ_k down to φ_0
    for k in range(1, T-1):
        # product φ_{T-1}...φ_k (includes φ_k) is tail_prod_from_k = prod_{r=T-1..k} φ_r
        tail_prod_from_k = 1.0
        for r in range(T-1, k-1, -1):
            tail_prod_from_k *= phis[r]
        numer = denom - tail_prod_from_k  # removes terms starting with φ_{T-1}...φ_k ... which include φ_k downward
        # multiply by product of Ψ(z_0) ... Ψ(z_{k-1})
        psi_prod = 1.0
        for j in range(0, k):
            psi_prod *= psis[j]
        Lamb[k] = (numer / denom) * psi_prod * m0
    
    # Λ_{T-1} := 1/denom * Ψ(z_{T-2}) ... Ψ(z_0) * m0
    if T >= 2:
        psi_prod_last = 1.0
        for j in range(0, T-1):
            psi_prod_last *= psis[j]
        Lamb[T-1] = (1.0 / denom) * psi_prod_last * m0
    else:
        # T == 1 special case
        Lamb[0] = (1.0 / denom) * m0
    
    return Lamb

# Fixed-point iteration: z_{new} = Lambda(z_old). Iterate until convergence.
def solve_fixed_point(m0, T, tol=1e-10, max_iter=2000, verbose=False):
    # initial guess for z: uniform small positive numbers
    z = np.full(T, 0.1)
    history = [z.copy()]
    for it in range(max_iter):
        z_new = Lambda(z, m0)
        err = np.max(np.abs(z_new - z))
        history.append(z_new.copy())
        z = z_new
        if verbose and (it % 100 == 0):
            print(f"iter {it}, err={err:.3e}, z={z}")
        if err < tol:
            break
    return z, np.array(history), err, it+1


m0 = 0.5

z_star, history, final_err, iterations = solve_fixed_point(m0=m0, T=T, tol=1e-12, max_iter=5000, verbose=True)


df = pd.DataFrame({
    't': np.arange(T),
    'z_star': z_star,
    'phi(z)': [phi_small(z) for z in z_star],
    'psi(z)': [Psi(z) for z in z_star]
})

print(df)
