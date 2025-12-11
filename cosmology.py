import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, cumulative_trapezoid
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# ==============================================================================
# OSACRA-HOLOGRAPHIC COSMOLOGY: RIGOROUS SIMULATION (v4.0)
# ==============================================================================
# "SCIENTIFIC HONESTY" STATEMENT:
# This model proposes a geometric ANSATZ for vacuum energy partitioning. 
# It is NOT a derived proof from QFT. It tests whether a specific geometric 
# coupling (Rhombic Dodecahedron) + Entropy Production (Star Formation) 
# yields a cosmologically consistent history.
#
# INPUTS (FIXED):
# 1. Early Universe: Planck 2018 (Omega_m, z_CMB)
# 2. Astrophysics: Madau-Dickinson Star Formation Rate (2014)
# 3. Geometry: Rhombic Dodecahedron (Alpha = 1/sqrt(3)) - THE HYPOTHESIS
#
# OUTPUTS (PREDICTED):
# 1. H0 (Local Hubble Constant)
# 2. q(z) (Deceleration History)
# 3. mu(z) (Distance Modulus vs SN1a)
# 4. t0 (Age of Universe)
# ==============================================================================

# --- 1. FIXED CONSTANTS (Planck 2018 Baseline) ---
c = 299792.458          # Speed of light in km/s
H0_PLANCK = 67.4        # Planck 2018 H0 estimate
OM0 = 0.315             # Planck 2018 Matter Density
OL0_STD = 1.0 - OM0     # Standard Model Dark Energy
Z_CMB = 1100.0          # Redshift of last scattering surface

# --- 2. THE GEOMETRIC ANSATZ ---
# Hypothesis: Vacuum coupling is driven by the projection efficiency of
# a Rhombic Dodecahedron (4D hypercube shadow).
ALPHA_ANSATZ = 1.0 / np.sqrt(3)  # ≈ 0.57735

# --- 3. PHYSICS KERNEL (ENTROPY INTEGRATION) ---
def madau_dickinson_sfr(z):
    """
    Standard fit for Cosmic Star Formation Rate (Madau & Dickinson 2014).
    This drives the information content of the vacuum.
    """
    numerator = 0.015 * (1 + z)**2.7
    denominator = 1 + ((1 + z) / 2.9)**5.6
    return numerator / denominator

def generate_entropy_profile(z_max=20.0, steps=2000):
    """
    Calculates the accumulated information mass M(z) from SFR.
    Assumes entropy accumulates as the universe evolves (z_max -> 0).
    """
    z_vals = np.linspace(0, z_max, steps)
    
    # We use a standard expansion history for the integration kernel to avoid 
    # circular dependency during initialization. This is a valid approximation 
    # because H(z) variations at high-z are dominated by Matter, which is fixed.
    Hz_approx = H0_PLANCK * np.sqrt(OM0*(1+z_vals)**3 + OL0_STD)
    dt_dz = 1.0 / ((1+z_vals) * Hz_approx)
    
    sfr = madau_dickinson_sfr(z_vals)
    
    # Integrate Entropy Flux (SFR * Time Dilation)
    integrand_mass = sfr * dt_dz
    cum_integral = cumulative_trapezoid(integrand_mass, z_vals, initial=0)
    
    # Mass accumulates from past (high z) to present (low z)
    # The total accumulated mass at z=0 is cum_integral[-1]
    # The mass at redshift z is (Total - Accumulated_up_to_z)
    # Note: cumulative_trapezoid integrates z=0 -> z_max. 
    # So cum_integral[z] is mass from 0 to z.
    # Total mass available is cum_integral[-1].
    # Mass existing at redshift z (accumulated from z_max down to z):
    mass_profile = np.maximum(cum_integral[-1] - cum_integral, 0)
    
    # Normalize: We posit the Vacuum Potential is proportional to this accumulated mass
    if mass_profile[0] > 0:
        normalized_vacuum = mass_profile / mass_profile[0]
    else:
        normalized_vacuum = mass_profile
        
    return z_vals, normalized_vacuum

# Initialize Physics Lookup Table
Z_LOOKUP, VACUUM_LOOKUP = generate_entropy_profile()
get_entropy_factor = interp1d(Z_LOOKUP, VACUUM_LOOKUP, kind='cubic', fill_value="extrapolate")

# --- 4. THE OSACRA METRIC ---
def hubble_osacra(z, h0_candidate):
    """
    The Modified Friedmann Equation.
    rho_vac(z) = rho_vac_0 * [ (1-alpha) + alpha * Entropy(z) ]
    """
    # 1. Determine local densities to match H0_candidate
    # We must preserve physical matter density (Omega_m * h^2) or 
    # fixed Omega_m? Standard approach for H0 tension solvers:
    # Keep Omega_m fixed, implies physical density rho_m scales with h^2.
    # OR: Keep physical density fixed (from CMB), implies Omega_m scales.
    # PLANCK Constraint: Physical density Omega_m*h^2 is very tight.
    # Let's respect the physical matter density of Planck.
    
    h_planck = H0_PLANCK / 100.0
    h_cand = h0_candidate / 100.0
    
    # Omega_m at z=0 for the candidate (scaling to preserve physical density)
    Om_candidate = OM0 * (h_planck / h_cand)**2
    
    # Flatness assumption (standard inflation)
    Ovac_total_candidate = 1.0 - Om_candidate
    
    # 2. Calculate Dynamic Vacuum Term
    # The vacuum density is NOT constant. It evolves.
    # Evolution factor E_vac(z):
    # At z=0, Entropy(z)=1, so E_vac=1. Matches H0.
    # At z=high, Entropy(z)->0, so E_vac -> (1-alpha).
    
    term_geometric = (1.0 - ALPHA_ANSATZ)
    term_entropic = ALPHA_ANSATZ * get_entropy_factor(z)
    
    # Effective Omega_Lambda(z) * (H0/H(z))^2... 
    # Easier to write E^2(z)
    
    E_squared = Om_candidate * (1+z)**3 + Ovac_total_candidate * (term_geometric + term_entropic)
    
    return h0_candidate * np.sqrt(np.maximum(E_squared, 1e-10))

# --- 5. VALIDATION TOOLS ---

def get_age_universe(h0):
    """Calculates age t0 in Gyr."""
    integrand = lambda z: 1.0 / ((1+z) * hubble_osacra(z, h0))
    age_mpc_seconds, _ = quad(integrand, 0, np.inf)
    
    # Conversion: Mpc/km * s -> Gyr
    # 1 Mpc = 3.086e19 km
    # 1 Gyr = 3.154e16 s
    # Unit conversion factor is approx 977.8 / H0 (if H0 in km/s/Mpc)
    # Precise:
    km_per_mpc = 3.08567758e19
    seconds_per_gyr = 3.15576e16
    
    # The integral gives result in units of [1 / (km/s/Mpc)] = [Mpc * s / km]
    # Value * (km/Mpc) = seconds. 
    # But hubble_osacra returns km/s/Mpc.
    # Integral = sum( dz / [(1+z) * H] )
    # Units: 1 / (km/s/Mpc) = Mpc / (km/s) = (Mpc * s) / km
    
    age_seconds = age_mpc_seconds * km_per_mpc
    return age_seconds / seconds_per_gyr

def get_distance_modulus_delta(h0_candidate):
    """
    Compares the predicted distance modulus at z=1.0 vs Standard Model (Planck).
    This serves as a proxy for the 'Pantheon Test'.
    """
    # Standard Model dL
    integrand_std = lambda z: 1.0 / np.sqrt(OM0*(1+z)**3 + OL0_STD)
    dist_std, _ = quad(integrand_std, 0, 1.0) # Comoving distance in Hubble units
    dl_std = (c / H0_PLANCK) * (1+1.0) * dist_std
    mu_std = 5 * np.log10(dl_std) + 25
    
    # Osacra Model dL
    integrand_osa = lambda z: c / hubble_osacra(z, h0_candidate)
    dc_osa, _ = quad(integrand_osa, 0, 1.0)
    dl_osa = (1+1.0) * dc_osa
    mu_osa = 5 * np.log10(dl_osa) + 25
    
    return mu_osa - mu_std

def get_deceleration_transition(h0_candidate):
    """Finds the redshift z_t where the universe switches from deceleration to acceleration."""
    z_scan = np.linspace(0, 1.5, 1000)
    h_vals = hubble_osacra(z_scan, h0_candidate)
    
    # q(z) = (1+z)/H * dH/dz - 1
    dh_dz = np.gradient(h_vals, z_scan)
    q = (1+z_scan)/h_vals * dh_dz - 1
    
    # Find zero crossing
    for i in range(len(z_scan)-1):
        if q[i] < 0 and q[i+1] > 0:
            return z_scan[i]
    return 0.0 # Should not happen

# --- 6. MAIN SIMULATION ---

def run_simulation():
    print(f"--- OSACRA-HOLOGRAPHIC SIMULATION (Scientific Rigor Mode) ---")
    print(f"    Ansatz Alpha: {ALPHA_ANSATZ:.6f} (1/sqrt(3))")
    
    # A. CMB Anchor
    # Calculate the Acoustic Horizon distance (approx) defined by Planck parameters
    integrand_std = lambda z: c / (H0_PLANCK * np.sqrt(OM0*(1+z)**3 + OL0_STD))
    target_horizon, _ = quad(integrand_std, 0, Z_CMB)
    print(f"    Target Acoustic Horizon (Planck): {target_horizon:.2f} Mpc")
    
    # B. Solver
    def objective(h_guess):
        integrand_osa = lambda z: c / hubble_osacra(z, h_guess)
        d_osa, _ = quad(integrand_osa, 0, Z_CMB, limit=100)
        return d_osa - target_horizon
    
    pred_h0 = brentq(objective, 65.0, 75.0)
    
    # C. Validation Calculations
    age = get_age_universe(pred_h0)
    zt = get_deceleration_transition(pred_h0)
    d_mu = get_distance_modulus_delta(pred_h0)
    
    print(f"\n--- PREDICTION RESULTS ---")
    print(f"    H0 (Predicted): {pred_h0:.2f} km/s/Mpc")
    print(f"    Age of Universe: {age:.2f} Gyr")
    print(f"    Deceleration Transition z_t: {zt:.2f} (LambdaCDM ~0.6)")
    print(f"    SN1a Mag Delta (z=1): {d_mu:.4f} mag")
    
    # D. Interpretation
    print(f"\n--- SCIENTIFIC HONESTY CHECK ---")
    if 0.5 < zt < 0.9:
        print("    [PASS] Transition z_t is physically plausible.")
    else:
        print("    [FAIL] Transition z_t contradicts structure formation history.")
        
    if abs(d_mu) < 0.1:
        print("    [PASS] Indistinguishable from Standard Candles (< 0.1 mag).")
    else:
        print("    [FAIL] Tension with Pantheon dataset.")
        
    return pred_h0, zt, age

# --- 7. PLOTTING SUITE ---

def generate_plots(pred_h0, zt):
    # Plot 1: Physics (History)
    z = np.linspace(0, 2.5, 200)
    h_std = H0_PLANCK * np.sqrt(OM0*(1+z)**3 + OL0_STD)
    h_osa = hubble_osacra(z, pred_h0)
    
    plt.figure(figsize=(12, 5))
    
    # Subplot A: H(z)
    plt.subplot(1, 2, 1)
    plt.plot(z, h_std, '--', color='gray', alpha=0.7, label=f'Standard Model ($H_0={H0_PLANCK}$)')
    plt.plot(z, h_osa, color='#00CC66', linewidth=2.5, label=f'Osacra Ansatz ($H_0={pred_h0:.2f}$)')
    plt.xlabel('Redshift z')
    plt.ylabel('H(z) [km/s/Mpc]')
    plt.title('Expansion History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()
    
    # Subplot B: Context (Comparison Plot)
    # Forest Plot of H0 values
    plt.subplot(1, 2, 2)
    
    # Data
    methods = ['Planck 2018\n(Early Universe)', 'TRGB (Freedman)\n(Mid-Range)', 'Osacra Model\n(Geometric Prediction)', 'SH0ES (Riess)\n(Late Universe)']
    values = [67.4, 69.8, pred_h0, 73.04]
    errors = [0.5, 0.8, 0.0, 1.04] # Osacra has no error bar as it is a theoretical point prediction here
    colors = ['gray', 'blue', '#00CC66', 'red']
    
    y_pos = np.arange(len(methods))
    
    plt.errorbar(values, y_pos, xerr=errors, fmt='o', capsize=5, color='black', alpha=0.5)
    for i in range(len(methods)):
        plt.plot(values[i], y_pos[i], 'o', markersize=12, color=colors[i])
        plt.text(values[i], y_pos[i]+0.2, f"{values[i]:.2f}", ha='center', fontsize=10, fontweight='bold')

    plt.yticks(y_pos, methods)
    plt.xlabel('H0 [km/s/Mpc]')
    plt.title('The Hubble Tension Landscape')
    plt.grid(True, axis='x', alpha=0.3)
    plt.xlim(65, 76)
    
    # Add vertical line for Osacra
    plt.axvline(pred_h0, color='#00CC66', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('osacra_rigorous_analysis.png')
    print("\n    Plots saved to 'osacra_rigorous_analysis.png'")

if __name__ == "__main__":
    h0_val, zt_val, age_val = run_simulation()
    generate_plots(h0_val, zt_val)