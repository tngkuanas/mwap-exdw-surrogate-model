# Theoretical Foundations of Multiphase Flow and Reservoir Mechanics
**Project:** ExxonMobil DataWorks Challenge 2026  
**Workspace:** `/Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp`  
**Document:** `RESERVOIR_PHYSICS_FOUNDATIONS.md`  
**Author:** Pair Programming Research Assistant  
**Date:** 2026-10-11  

---

## 1. Governing Conservation Laws and Darcy Flow

The transport of fluids through porous subsurface media is governed by mass conservation coupled with Darcy's empirical law for multiphase fluid filtration.

### 1.1 Multiphase Mass Conservation Equations
For a reservoir system containing three fluid phases—oil ($o$), water ($w$), and gas ($g$)—with gas dissolved in the oil phase according to the solution gas-oil ratio $R_s(p)$, the volumetric mass conservation differential equations in three-dimensional space are:

$$\nabla \cdot \left[ \frac{k \, k_{ro}(S_w, S_g)}{\mu_o(p) B_o(p)} \nabla \Phi_o \right] + \tilde{q}_o = \frac{\partial}{\partial t}\left( \frac{\phi \, S_o}{B_o(p)} \right)$$

$$\nabla \cdot \left[ \frac{k \, k_{rw}(S_w)}{\mu_w(p) B_w(p)} \nabla \Phi_w \right] + \tilde{q}_w = \frac{\partial}{\partial t}\left( \frac{\phi \, S_w}{B_w(p)} \right)$$

$$\nabla \cdot \left[ \frac{k \, k_{rg}(S_g)}{\mu_g(p) B_g(p)} \nabla \Phi_g + R_s(p) \frac{k \, k_{ro}(S_w, S_g)}{\mu_o(p) B_o(p)} \nabla \Phi_o \right] + \tilde{q}_g = \frac{\partial}{\partial t}\left( \frac{\phi \, S_g}{B_g(p)} + \frac{\phi \, R_s(p) S_o}{B_o(p)} \right)$$

where:
- $k$: Absolute permeability tensor ($\text{m}^2$ or mD).
- $k_{r\alpha}$: Relative permeability to phase $\alpha \in \{o, w, g\}$.
- $\mu_\alpha$: Dynamic viscosity of phase $\alpha$ ($\text{Pa}\cdot\text{s}$ or cP).
- $B_\alpha$: Formation volume factor of phase $\alpha$ ($\text{res bbl} / \text{STB}$ or $\text{res bbl} / \text{MSCF}$).
- $\Phi_\alpha = p_\alpha - \rho_\alpha g z$: Flow potential for phase $\alpha$.
- $\phi$: Porosity of the rock matrix.
- $S_\alpha$: Phase saturation, subject to the closure relation $\sum_\alpha S_\alpha = S_o + S_w + S_g = 1.0$.
- $\tilde{q}_\alpha$: Volumetric source/sink term (well production/injection).

---

## 2. Saturation Functions and Relative Permeability

Multiphase flow interference is modeled through relative permeability curves $k_{r\alpha}(S_\alpha)$ and capillary pressure $p_c(S)$. In the benchmark model, water-oil relative permeabilities follow the Corey (Brooks-Corey) power-law formulation.

### 2.1 Corey Relative Permeability Formulations
Defining the normalized (effective) water saturation $S_{en}$:

$$S_{en} = \frac{S_w - S_{wc}}{1 - S_{wc} - S_{orw}}$$

The phase relative permeabilities are expressed as:

$$k_{rw}(S_w) = k_{rw}^o \left( S_{en} \right)^{n_w}$$

$$k_{ro}(S_w) = k_{ro}^o \left( 1 - S_{en} \right)^{n_o}$$

where:
- $S_{wc}$ (or $S_{wl}$): Connate / irreducible water saturation ($0.12 - 0.20$).
- $S_{orw}$ (or $S_{owcr}$): Residual oil saturation to water displacement ($0.15 - 0.25$).
- $k_{rw}^o$: End-point water relative permeability at $(1 - S_{orw})$.
- $k_{ro}^o$: End-point oil relative permeability at $S_{wc}$ (typically $1.0$).
- $n_w, n_o$: Corey curvature exponents (typically $n_w \approx 2.0 - 3.0$, $n_o \approx 2.0 - 4.0$).

### 2.2 Inspection of Field Grid Endpoints
Direct inspection of the 3D cell data across Cases 1–85 reveals:
- Initial oil saturation in pay zones: $S_{o, \text{init}} \approx 0.75 - 0.82$.
- Residual oil saturation parameter: $S_{owcr} \approx 0.20$.
- Mobile oil saturation range: $\Delta S_o = S_{o, \text{init}} - S_{owcr} \approx 0.55 - 0.62$.
- Initial reservoir pressure: $p_{\text{init}} \approx 4,670\text{ psia}$.
- Bubble point pressure: $p_b \approx 3,200\text{ psia}$.
Because $p(t) > p_b$ throughout the waterflood life, free gas is negligible ($S_g \approx 0$), and all produced gas remains in solution ($R_s = \text{const} = 0.3633\text{ MSCF/STB}$).

---

## 3. Buckley-Leverett Fractional Flow and Front Advance

For 1D immiscible displacement of oil by water without capillary pressure and gravity effects, Buckley and Leverett (1942) established that the fractional flow of water $f_w(S_w)$ is:

$$f_w(S_w) = \frac{q_w}{q_t} = \frac{1}{1 + \frac{k_{ro}(S_w)}{k_{rw}(S_w)} \frac{\mu_w}{\mu_o}}$$

where $q_t = q_o + q_w$ is the total liquid production rate.

### 3.1 Water-Oil Ratio (WOR) Relationship
The surface water-oil ratio ($\text{WOR}$) is directly coupled to fractional flow:

$$\text{WOR} = \frac{q_w}{q_o} = \frac{f_w}{1 - f_w} \cdot \frac{B_o}{B_w} = \frac{k_{rw}(S_w) \mu_o B_o}{k_{ro}(S_w) \mu_w B_w}$$

Substituting Corey relative permeabilities yields the logit-linear relationship:

$$\ln(\text{WOR}) = \ln\left( \frac{k_{rw}^o \mu_o B_o}{k_{ro}^o \mu_w B_w} \right) + n_w \ln(S_{en}) - n_o \ln(1 - S_{en})$$

As cumulative water injection advances the saturation profile according to Welge's tangent construction:

$$\left( \frac{dx}{dt} \right)_{S_w} = \frac{v_t}{\phi} \left( \frac{df_w}{dS_w} \right)_{S_w}$$

Water breakthrough occurs at time $t_{bt} = \frac{L \phi}{v_t (df_w/dS_w)_{S_{wf}}}$. Post-breakthrough, $f_w$ rises rapidly, causing $\text{WOR}$ to grow exponentially with cumulative oil produced ($Q_o$), validating the empirical Ershaghi & Omoregie (1978) waterflood performance model:

$$\ln(\text{WOR}) \approx a + b \cdot Q_o$$

---

## 4. Waterflood Voidage Replacement and Decline Mechanics

In active waterflood operations with peripheral or pattern water injection, reservoir pressure is stabilized by balancing produced fluid volumes with injected water volumes.

### 4.1 Voidage Replacement Ratio (VRR)
$$\text{VRR}(t) = \frac{i_w(t) B_w}{q_o(t) B_o + q_w(t) B_w + [q_g(t) - q_o(t) R_s] B_g}$$

When $\text{VRR} \approx 1.0$:
1. Reservoir average pressure $p(t)$ stabilizes ($dp/dt \approx 0$).
2. Total liquid production rate $q_L(t) = q_o(t) + q_w(t)$ becomes approximately constant or limited by artificial lift / surface separator capacity:
   $$q_L(t) \approx q_{L, \text{max}} = \text{const}$$
3. Oil rate decline is driven strictly by water displacement (rising water cut $f_w(t)$):
   $$q_o(t) = q_L(t) \cdot [1 - f_w(t)]$$
   $$\frac{dq_o}{dt} = -q_L \frac{df_w}{dt} = -q_o(t) \cdot \frac{d \ln(1 - f_w)}{dt}$$

### 4.2 Arps Decline Law Asymptotics
The general Arps (1945) decline equation expresses rate as:

$$q_o(t) = \frac{q_{o,0}}{(1 + b D_i t)^{1/b}}$$

The loss-ratio parameter $b$ represents the rate of change of reciprocal decline:

$$b = \frac{d}{dt}\left( \frac{1}{D(t)} \right) = -\frac{d}{dt}\left( \frac{q_o}{dq_o/dt} \right)$$

- Under primary gas-depletion or compressible solution-gas drive, $b \in [0.3, 0.7]$.
- Under constant-liquid waterflooding with rising water cut, $q_o(t) \propto e^{-D t}$ locally, which corresponds mathematically to $b \to 0$ (exponential decline).
- When unconstrained hyperbolic decline ($b > 0$) is fitted to a waterflood tail, $(1 + b D t)^{-1/b}$ predicts an unphysically flat, heavy tail, leading to severe EUR overestimation.
- This fundamental reservoir physics explains why the locked ensemble assigned **61.8% weight to Exponential DCA** and **0.0% weight to Hyperbolic DCA** for oil forecasting.
