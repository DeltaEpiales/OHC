import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, cumulative_trapezoid
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# ==============================================================================
# OSACRA-HOLOGRAPHIC COSMOLOGY: PREDICTION ENGINE (v2.1)
# ==============================================================================
# REFERENCES:
# [1] Planck Collaboration 2018: A&A 641, A6 (2020) - Early Universe Constraints
# [2] Madau & Dickinson 2014: ARA&A 52, 415 - Cosmic Star Formation History
# [3] Riess et al. 2022: ApJ 934, L7 - Late Universe Constraints (SHOES)
# ==============================================================================

# --- 1. FIXED CONSTANTS (Planck 2018 Baseline) ---
c = 299792.458          # Speed of light in km/s
H0_PLANCK = 67.4        # Planck 2018 H0 estimate (km/s/Mpc)
OM0 = 0.315             # Planck 2018 Matter Density Parameter
OL0_STD = 1.0 - OM0     # Standard Model Dark Energy
Z_CMB = 1100.0          # Redshift of last scattering surface

# --- 2. THE GEOMETRIC HYPOTHESIS ---
# The Osacra Constant is derived from the projection efficiency of a 
# Rhombic Dodecahedral lattice (4D shadow) onto 3D space.
ALPHA_GEOMETRIC = 1.0 / np.sqrt(3)  # ≈ 0.57735

# --- 3. PHYSICS KERNEL ---

def madau_dickinson_sfr(z):
    """
    Returns the normalized Cosmic Star Formation Rate (SFR).
    Source: Eq 15 in Madau & Dickinson (2014).
    """
    numerator = 0.015 * (1 + z)**2.7
    denominator = 1 + ((1 + z) / 2.9)**5.6
    return numerator / denominator

def generate_entropy_profile(z_max=20.0, steps=2000):
    """
    Calculates the Entropy Flux Profile F(z) based on BH accretion history.
    """
    z_vals = np.linspace(0, z_max, steps)
    
    # Time dilation factor approximation (using Planck cosmology as a baseline)
    Hz_approx = H0_PLANCK * np.sqrt(OM0*(1+z_vals)**3 + OL0_STD)
    dt_dz = 1.0 / ((1+z_vals) * Hz_approx)
    
    sfr = madau_dickinson_sfr(z_vals)

    # 1. Integrate Mass Accumulation (M)
    # cumtrapz (scipy 1.6+) or cumulative_trapezoid (scipy 1.10+)
    integrand_mass = sfr * dt_dz
    cum_integral = cumulative_trapezoid(integrand_mass, z_vals, initial=0)
    
    # Mass accumulates from z_max -> 0
    total_mass = cum_integral[-1]
    mass_profile = np.maximum(total_mass - cum_integral, 0)
    mass_norm = mass_profile / mass_profile[0] if mass_profile[0] > 0 else mass_profile

    # 2. Calculate Entropy Flux (Force) ~ Mass * Accretion Rate
    entropy_rate = mass_norm * sfr
    
    # 3. Calculate Vacuum Potential (Integral of Flux)
    integrand_vac = entropy_rate * dt_dz
    cum_vac_integral = cumulative_trapezoid(integrand_vac, z_vals, initial=0)
    
    # Vacuum accumulates from z_max -> 0
    total_vac = cum_vac_integral[-1]
    vacuum_profile = np.maximum(total_vac - cum_vac_integral, 0)
    
    # Normalize to 1 at z=0
    if vacuum_profile[0] > 0:
        normalized_vacuum = vacuum_profile / vacuum_profile[0]
    else:
        normalized_vacuum = vacuum_profile
        
    return z_vals, normalized_vacuum

# Initialize Lookup Table
print("--- Initializing Physics Kernel ---")
Z_LOOKUP, VACUUM_LOOKUP = generate_entropy_profile()
get_entropy_factor = interp1d(Z_LOOKUP, VACUUM_LOOKUP, kind='cubic', fill_value="extrapolate")

# --- 4. THE OSACRA METRIC ---

def hubble_osacra(z, h0_candidate):
    """
    The Modified Friedmann Equation.
    """
    # Rescale Matter Density to preserve Omega_m*h^2 (Physical Matter Density)
    # If H0 increases, Omega_m must decrease to keep physical density constant.
    Om_local = OM0 * (H0_PLANCK / h0_candidate)**2
    
    # Flatness constraint: Remainder is Vacuum
    Ovac_total = 1.0 - Om_local
    
    # Partition Vacuum into Geometric (Static) and Entropic (Dynamic)
    term_geometric = (1.0 - ALPHA_GEOMETRIC)
    term_entropic = ALPHA_GEOMETRIC * get_entropy_factor(z)
    
    Ovac_dynamic = Ovac_total * (term_geometric + term_entropic)
    
    E_squared = Om_local * (1+z)**3 + Ovac_dynamic
    return h0_candidate * np.sqrt(np.maximum(E_squared, 1e-10))

# --- 5. SOLVER & VISUALIZATION ---

def solve_and_plot():
    print(f"\n--- OSACRA GEOMETRIC PREDICTION ENGINE ---")
    print(f"    Alpha: {ALPHA_GEOMETRIC:.6f}")
    
    # A. Calculate Target Distance (Planck CMB Anchor)
    # Distance to z=1100 in standard LCDM
    integrand_std = lambda z: c / (H0_PLANCK * np.sqrt(OM0*(1+z)**3 + OL0_STD))
    target_dist, _ = quad(integrand_std, 0, Z_CMB)
    print(f"    Target Horizon (Planck): {target_dist:.2f} Mpc")
    
    # B. Solve for Osacra H0
    def objective(h_guess):
        integrand_osacra = lambda z: c / hubble_osacra(z, h_guess)
        d_osacra, _ = quad(integrand_osacra, 0, Z_CMB, limit=100)
        return d_osacra - target_dist
    
    try:
        predicted_h0 = brentq(objective, 65.0, 80.0, xtol=1e-4)
        print(f"    PREDICTED H0: {predicted_h0:.4f} km/s/Mpc")
        
        # C. Generate Proof Plot
        z_plot = np.linspace(0, 2.5, 500)
        h_std = H0_PLANCK * np.sqrt(OM0*(1+z_plot)**3 + OL0_STD)
        h_osa = hubble_osacra(z_plot, predicted_h0)
        
        plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(z_plot, h_std, '--', color='gray', label=f'Standard Model ($H_0={H0_PLANCK}$)')
        plt.plot(z_plot, h_osa, color='#00CC66', linewidth=2.5, label=f'Osacra Geometric ($H_0={predicted_h0:.2f}$)')
        
        plt.title(f"Geometric Solution ($\\alpha=1/\\sqrt{{3}}$): $H_0 = {predicted_h0:.2f}$")
        plt.xlabel("Redshift z")
        plt.ylabel("H(z) [km/s/Mpc]")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 2.5)
        plt.gca().invert_xaxis()
        
        plt.savefig('osacra_geometric_proof.png', bbox_inches='tight')
        print("    Proof plot saved.")
        
    except Exception as e:
        print(f"    Solver Failed: {e}")

if __name__ == "__main__":
    solve_and_plot()