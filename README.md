# Osacra-Holographic Cosmology

**The Geometric Entropy Ansatz: Investigating the Hubble Tension via Holographic Scaling**

This repository contains the codebase and theoretical framework for the **Osacra-Holographic Ansatz**, a parameter-free extension to $\Lambda$CDM that investigates whether the Hubble Tension ($H_0$) can be resolved by a dynamic vacuum energy coupled to cosmic entropy.

## The Theory in Brief

The "Hubble Tension" is a $5\sigma$ discrepancy between early-universe ($67.4$) and late-universe ($73.0$) measurements of the expansion rate. We propose that this tension arises because Vacuum Energy ($\Lambda$) is not constant, but a cumulative record of information storage.

### The Geometric Ansatz

We postulate that the vacuum microstructure follows a **Face-Centered Cubic (FCC)** packing efficiency—the densest possible packing of information in 3D space. The unit cell of this lattice is the **Rhombic Dodecahedron**.

The "Osacra Constant" ($\alpha$) is the geometric projection efficiency of this cell, defined by its underlying cubic scaffold:

$$\alpha = \frac{R_{inscribed}}{R_{circumscribed}} = \frac{1}{\sqrt{3}} \approx 0.577$$

This constant acts as the partition function between the "Geometric Vacuum" (static Dark Energy) and the "Entropic Vacuum" (dynamic information content).

## Repository Structure

  * `cosmology.py` - **The Physics Engine.** Integrates the Madau-Dickinson Star Formation Rate to calculate entropy flux, solves the modified Friedmann equation, and runs falsification tests (Age, $q(z)$, Supernovae).
  * `geometry.py` - **The Visualization Engine.** A dual-mode script that generates:
      * `osacra_interactive.html`: An interactive 3D model (Plotly) to explore the lattice and scaffold.
      * `osacra_geometry.png`: A high-resolution static figure for the paper.
  * `paper.tex` - **The Manuscript.** A complete LaTeX source file incorporating the latest results and theoretical derivations.

## Getting Started

### Prerequisites

Install the required scientific libraries:

```bash
pip install -r requirements.txt
```

### 1\. Run the Physics Engine

Calculate the Hubble Constant and perform falsification tests:

```bash
python cosmology.py
```

**Output:**

```text
--- OSACRA-HOLOGRAPHIC SIMULATION (Scientific Rigor Mode) ---
Ansatz Alpha: 0.577350 (1/sqrt(3))
Target Acoustic Horizon (Planck): 13934.52 Mpc

--- PREDICTION RESULTS ---
H0 (Predicted): 70.80 km/s/Mpc
Age of Universe: 13.71 Gyr
Deceleration Transition z_t: 0.74
SN1a Mag Delta (z=1): -0.0312 mag

--- SCIENTIFIC HONESTY CHECK ---
[PASS] Transition z_t is physically plausible.
[PASS] Indistinguishable from Standard Candles (< 0.1 mag).
```

*Graph saved to: `osacra_rigorous_analysis.png`*

### 2\. View the Holographic Geometry

Generate the interactive 3D unit cell and paper figure:

```bash
python geometry.py
```

**Output:**

  * `osacra_interactive.html`: Open in browser to rotate the lattice and verify the $\alpha$ ratio.
  * `osacra_geometry.png`: Static figure for publication.

-----

## Methodology & Results

### 1\. The Prediction ($H_0$)

The model assumes the vacuum energy scales with the entropy produced by star formation (Madau-Dickinson SFR). Anchored to the Planck CMB horizon, the model uniquely predicts:
$$H_0 = 70.80 \text{ km/s/Mpc}$$
This value sits strictly between the Planck ($67.4$) and SH0ES ($73.0$) values, aligning with **TRGB (Tip of the Red Giant Branch)** measurements.

### 2\. Validation (Falsification Tests)

To ensure scientific rigor, we test against intermediate-redshift constraints:

  * **Age of Universe:** $t_0 = 13.71$ Gyr (Consistent with Globular Clusters).
  * **Deceleration Transition:** $z_t \approx 0.74$. The universe accelerates slightly earlier than $\Lambda$CDM ($z_t \approx 0.6$) due to the entropy surge at "Cosmic Noon."
  * **Pantheon Test:** The distance modulus deviation at $z=1$ is $\Delta \mu \approx -0.03$ mag, making the model observationally indistinguishable from standard candles.

-----

## Frequently Asked Questions

**Q: Is $\alpha = 1/\sqrt{3}$ a tuned parameter?**
**A:** No, it is a geometric ansatz. We hypothesize that vacuum packing follows the Rhombic Dodecahedron geometry. If this hypothesis is true, $\alpha$ is fixed mathematically by the cell's radius ratio. We do not adjust it to fit the data; the result ($70.80$) is a consequence of the geometry.

**Q: Why isn't the prediction 73 km/s/Mpc?**
**A:** Previous iterations claimed 73, but rigorous integration of the entropy history reveals the natural solution is $70.80$. This suggests the "true" $H_0$ lies in the middle, supporting the TRGB dataset over the extremes.

**Q: Does this break the Supernova data?**
**A:** No. Our falsification test shows a distance modulus delta of only $-0.03$ mag, which is well within the error bars of the Pantheon+ dataset.

## Citation

If you use this code or theory, please cite:

```text
Osacra, R. (2025). "The Geometric Entropy Solution: Resolving the Hubble Tension via Holographic Scaling."
```