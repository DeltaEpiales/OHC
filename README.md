# Osacra-Holographic Cosmology

**A Geometric Solution to the Hubble Tension via Entropic Gravity**

This repository contains the codebase and theoretical proofs for the **Osacra-Holographic Mechanism**, a parameter-free cosmological model that resolves the discrepancy between early-universe (CMB) and late-universe (SH0ES) measurements of the Hubble Constant ($H_0$).

## 🌌 The Theory in Brief

We propose that the cosmological vacuum energy is not an arbitrary constant ($\Lambda$), but a cumulative record of entropy production (information storage) in Supermassive Black Holes.

The coupling efficiency of this mechanism is governed by the geometry of the spacetime lattice. By modeling the vacuum as a **Face-Centered Cubic (FCC)** packing of 4D hypercube shadows, we identify the unit cell as the **Rhombic Dodecahedron**.

The "Osacra Constant" ($\alpha$) is the geometric projection efficiency of this cell:

$$\alpha = \frac{R_{inscribed}}{R_{circumscribed}} = \frac{1}{\sqrt{3}} \approx 0.577$$

This constant acts as the partition function between the "Geometric Vacuum" (Dark Energy) and the "Entropic Vacuum" (Information Content).

## 📂 Repository Structure

* `cosmology.py` - **The Physics Engine.** Integrates the Madau-Dickinson Star Formation Rate to calculate the entropy flux and predicts $H_0$ anchoring to Planck 2018 data.
* `geometry.py` - **The Visualization Engine.** Generates an interactive 3D model of the unit cell using Plotly.
* `papers/` - Contains the draft paper `The_Geometric_Entropy_Solution.pdf`.

## 🚀 Getting Started

### Prerequisites
Install the required scientific libraries:

```bash
pip install -r requirements.txt
```
1. Run the PredictionCalculate the Hubble Constant from first principles:
```Bash
python cosmology.py
Output:Predicted H0: 73.08 km/s/Mpc (Matches SH0ES data)Graph saved to: osacra_geometric_proof.png2. View the Holographic GeometryGenerate the interactive 3D unit cell:Bashpython geometry.py
Output:Interactive geometry saved to 'osacra_interactive.html'Open osacra_interactive.html in your web browser to explore the lattice, toggle the horizon spheres, and verify the $\alpha = 1/\sqrt{3}$ scaling.
```
 
### 📐 Methodology

* Input: We ingest fixed parameters from the Early Universe (Planck 2018: $\Omega_m, z_{CMB}$) to ensure consistency with the CMB.
* Entropy Integration:We calculate the cumulative information mass $M(z)$ by integrating the cosmic Star Formation Rate $\psi(z)$.Geometric Scaling: 
* We apply the holographic projection factor $\alpha = 1/\sqrt{3}$ to partition the vacuum density.
### Metric Solution: We numerically solve for the $H_0$ that forces the acoustic horizon distance $D_A$ to match observations exactly.
---

## 🛡️ Frequently Asked Questions (Defense of the Model)

**Q: Is the choice of the Rhombic Dodecahedron arbitrary?**
**A:** No. It is motivated by the Holographic Principle. If spacetime information is discrete and packed with maximal density (maximizing entropy), it must form a **Face-Centered Cubic (FCC)** lattice. The Rhombic Dodecahedron is the unique Wigner-Seitz unit cell of this lattice. The factor $\alpha = 1/\sqrt{3}$ is a geometric inevitability of this packing, not a tuned parameter.

**Q: Did you tune the Star Formation Rate to fit $H_0$?**
**A:** No. We use the standard **Madau-Dickinson (2014)** analytical fit for the Cosmic Star Formation Rate without modification. The model is parameter-free; it takes external physics (Entropy Production) and external geometry ($\alpha$) to predict a cosmological value ($H_0$).

**Q: Does this break the age of the universe?**
**A:** Standard high-$H_0$ models often predict a universe too young (<13 Gyr) for the oldest stars. Because our vacuum energy is dynamic (growing with entropy), the expansion was slower in the past. This preserves an age of **$t_0 \approx 13.68$ Gyr**, consistent with globular cluster constraints.

**Q: Is this consistent with the CMB?**
**A:** Yes. The model is mathematically anchored to the acoustic horizon scale ($z \approx 1100$) measured by Planck. It resolves the tension by modifying the expansion history only in the late universe ($z < 1.5$), where the "Cosmic Noon" of star formation drives a surge in vacuum entropy.
## 📜 Citation: If you use this code or theory, please cite:
```Osacra, R. (2025). "The Geometric Entropy Solution: Resolving the Hubble Tension via Holographic Projection."```