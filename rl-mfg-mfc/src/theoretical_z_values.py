import numpy as np

# Given z values (for t=0,1,2,3)
z = [0.3897613315318339,
0.7374869368000772,
0.8025699054330236,
0.6497764126229566]
T = 4  # Time periods: t=0,1,2,3,4 (but only z_0 through z_3)

# Parameters
gamma = 0.2
rho = 0.95
C = 3
supp_W = [0.9, 1.3]
pmf_W = [0.75, 0.25]

# Function G(z,W) = C*W / (rho*E[W^gamma] * (1 + (C-1)*z^3))
def G(z_val, w):
    """Production function G(z,W)"""
    denominator = pEWgamma * (1 + (C - 1) * np.power(z_val, 3))
    return C * w / denominator

# Calculate E[W^gamma] first
exp_W_gamma = sum(np.power(w, gamma) * p for w, p in zip(supp_W, pmf_W))
pEWgamma = rho * exp_W_gamma

# Functions
def Phi(z_val):
    """Φ(z) = ρE[G^γ(z,W)]"""
    # E[G^γ(z,W)] = E[(C*W / denominator)^γ]
    exp_G_gamma = 0
    denominator = pEWgamma * (1 + (C - 1) * np.power(z_val, 3))
    for w, p in zip(supp_W, pmf_W):
        G_val = C * w / denominator
        exp_G_gamma += np.power(G_val, gamma) * p
    return rho * exp_G_gamma

def phi(z_val):
    return np.power(Phi(z_val), 1.0 / (gamma - 1))

# Calculate phi values
phi_vals = [phi(z_t) for z_t in z]

# Calculate D_t (backward)
D = np.zeros(T + 1)
D[T] = 1.0

for t in range(T - 1, -1, -1):
    D[t] = phi_vals[t] * D[t + 1] / (1 + phi_vals[t] * D[t + 1])

# Calculate optimal policy coefficients
print("="*50)
print("OPTIMAL INVESTMENT POLICY: α̂_t(x) = c_t × x")
print("="*50)

for t in range(T):
    c_t = 1 / (1 + phi_vals[t] * D[t + 1])
    print(f"α̂_{t}(x) = {c_t:.6f} × x")

print(f"α̂_{T}(x) = 0  (terminal time)")
print("="*50)