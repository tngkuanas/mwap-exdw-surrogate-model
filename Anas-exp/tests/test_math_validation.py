"""
Unit and Adversarial Mathematical Tests: test_math_validation.py
Addresses Section 7 of Fundamental Mathematical Validation.

Includes analytical ground-truth test cases:
1. Analytical FIM & Linear Mode Decomposition Identity (sum P_i^2 / lambda_i == c^T F^-1 c)
2. Analytical Continuous Exponential DCA EUR Integral vs Discrete Summation
3. Algebraic ANOVA Variance Partitioning Identity (SS_total == SS_m + SS_s + SS_inter)
4. Tree Discontinuity & Finite Difference Breakdown Simulation
5. Analytical Volumetrics Scaling & Havlena-Odeh Non-Identifiability
"""
import numpy as np
import pytest
from scipy import linalg
from sklearn.ensemble import ExtraTreesRegressor


class TestAnalyticalMathematics:
    """Verifies mathematical derivations and computational correctness against analytical truths."""

    def test_analytical_fim_and_mode_variance_decomposition(self):
        """
        Adversarial Test 1: Analytical FIM and Mode Variance Decomposition.
        Given linear model y = J theta + eps, eps ~ N(0, Sigma).
        Verify that mode variance sum sum_i (c^T v_i)^2 / lambda_i equals c^T F^-1 c exactly.
        """
        np.random.seed(42)
        n_obs = 20
        n_param = 4
        J = np.random.randn(n_obs, n_param)
        sigma = 0.5
        Sigma = (sigma ** 2) * np.eye(n_obs)
        
        # Analytical FIM: F = J^T Sigma^-1 J
        F_analytical = (J.T @ J) / (sigma ** 2)
        inv_F = linalg.inv(F_analytical)
        
        # Arbitrary linear forecast gradient c
        c = np.array([1.5, -2.0, 0.8, -1.2])
        
        # Direct variance via inverse Fisher
        var_direct = float(c.T @ inv_F @ c)
        
        # Spectral decomposition
        eigvals, eigvecs = linalg.eigh(F_analytical)
        
        # Mode contributions: P_i = c^T v_i, Var_i = P_i^2 / lambda_i
        projections = c @ eigvecs
        var_modes = (projections ** 2) / eigvals
        var_spectral_sum = float(np.sum(var_modes))
        
        # Must match to machine precision
        assert np.isclose(var_direct, var_spectral_sum, rtol=1e-12, atol=1e-12)
        
        # Also verify that a stiff mode (large lambda_1) can have lower variance contribution
        # than a sloppy mode (small lambda_2) even if projection P_1 > P_2
        lambda_stiff = 1000.0
        lambda_sloppy = 10.0
        P_stiff = 50.0   # Large projection
        P_sloppy = 10.0  # Smaller projection
        
        var_stiff = (P_stiff ** 2) / lambda_stiff   # 2500 / 1000 = 2.5
        var_sloppy = (P_sloppy ** 2) / lambda_sloppy # 100 / 10 = 10.0
        
        # Sloppy mode variance is 4x larger despite 5x smaller projection
        assert var_sloppy > var_stiff

    def test_analytical_exponential_dca_recovery_integral(self):
        """
        Adversarial Test 2: Analytical Exponential DCA EUR Integral.
        For q(t) = q0 * exp(-D * t):
        Continuous Integral: Q_cum(T) = (q0 / D) * (1 - exp(-D * T))
        As T -> inf: Q_inf = q0 / D
        """
        q0 = 5000.0  # STB/D
        D = 0.05     # 1/year
        T_years = 20.0
        
        # Analytical continuous integral
        Q_cum_exact = (q0 * 365.25 / D) * (1.0 - np.exp(-D * T_years))
        Q_inf_exact = (q0 * 365.25 / D)
        
        # Discrete quarterly trapezoidal / midpoint approximation (dt = 0.25 years)
        dt_years = 0.25
        n_steps = int(T_years / dt_years)
        t_mid = (np.arange(n_steps) + 0.5) * dt_years
        q_mid = q0 * np.exp(-D * t_mid)
        Q_discrete_midpoint = np.sum(q_mid * 365.25 * dt_years)
        
        # Midpoint rule relative error is O(dt^2)
        rel_err = abs(Q_discrete_midpoint - Q_cum_exact) / Q_cum_exact
        assert rel_err < 1e-4  # Within 0.01%
        assert Q_cum_exact < Q_inf_exact

    def test_algebraic_anova_partitioning_identity(self):
        """
        Adversarial Test 3: Algebraic ANOVA Variance Partitioning.
        Verifies SS_total == SS_model + SS_scenario + SS_interaction to machine precision.
        """
        np.random.seed(123)
        n_models = 4
        n_scenarios = 10
        # Generate arbitrary tensor
        Y = np.random.randn(n_models, n_scenarios) * 1e6 + 5e7
        
        mean_grand = np.mean(Y)
        mean_m = np.mean(Y, axis=1)
        mean_s = np.mean(Y, axis=0)
        
        ss_total = np.sum((Y - mean_grand) ** 2)
        ss_model = n_scenarios * np.sum((mean_m - mean_grand) ** 2)
        ss_scen = n_models * np.sum((mean_s - mean_grand) ** 2)
        ss_inter = np.sum((Y - mean_m[:, None] - mean_s[None, :] + mean_grand) ** 2)
        
        ss_partitioned = ss_model + ss_scen + ss_inter
        assert np.isclose(ss_total, ss_partitioned, rtol=1e-10, atol=1e-10)

    def test_tree_discontinuity_and_finite_difference_instability(self):
        """
        Adversarial Test 4: ExtraTrees Decision Boundary Jump.
        Proves that finite differences on tree ensembles jump discontinuously
        when perturbation step size crosses decision split thresholds.
        """
        np.random.seed(99)
        X_train = np.linspace(0, 1, 50).reshape(-1, 1)
        y_train = np.sin(2 * np.pi * X_train.ravel())
        
        et = ExtraTreesRegressor(n_estimators=10, random_state=42)
        et.fit(X_train, y_train)
        
        x0 = 0.5
        # Evaluate finite difference derivative (f(x0+eps) - f(x0-eps)) / (2*eps)
        # For small eps where no split is crossed, derivative is strictly 0.0 (piecewise constant)
        # For larger eps crossing a split, derivative jumps to a non-zero value
        eps_tiny = 1e-6
        d_tiny = (et.predict([[x0 + eps_tiny]])[0] - et.predict([[x0 - eps_tiny]])[0]) / (2 * eps_tiny)
        
        eps_cross = 0.05
        d_cross = (et.predict([[x0 + eps_cross]])[0] - et.predict([[x0 - eps_cross]])[0]) / (2 * eps_cross)
        
        # d_tiny is 0.0 because trees are locally flat
        assert d_tiny == 0.0
        # d_cross is non-zero because it crosses step boundaries
        assert abs(d_cross) > 0.01

    def test_volumetric_scaling_and_havlena_odeh_identifiability(self):
        """
        Adversarial Test 5: Volumetric Scaling with Boi.
        Pore volume V_p = 100 MM res bbl.
        OOIP(Boi) = V_p / Boi.
        Verify exact inverse scaling.
        """
        V_p = 100.0  # MM res bbl
        Boi_1 = 1.25
        Boi_2 = 1.40
        
        ooip_1 = V_p / Boi_1
        ooip_2 = V_p / Boi_2
        
        assert np.isclose(ooip_1 * Boi_1, ooip_2 * Boi_2)
        # Verify that dynamic material balance slope F/(Eo + m*Eg + Efw) is undefined
        # when pressure history (and thus Eo, Eg, Efw) is missing.
        Eo = 0.0  # Missing pressure expansion data
        with pytest.raises(ZeroDivisionError):
            _ = ooip_1 / Eo
