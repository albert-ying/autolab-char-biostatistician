---
name: bayesian-modeling
description: Bayesian statistical modeling with PyMC. Hierarchical models, MCMC diagnostics, posterior predictive checks, model comparison.
metadata:
    skill-author: Albert Ying
---

# Bayesian modeling

## When to use

- Hierarchical or multilevel data
- Small sample sizes where priors matter
- Uncertainty quantification beyond p-values
- Model comparison with LOO/WAIC

## PyMC workflow

```python
import pymc as pm
import arviz as az

with pm.Model() as model:
    # Priors
    mu = pm.Normal("mu", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=5)

    # Likelihood
    obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=data)

    # Sample
    trace = pm.sample(2000, tune=1000, cores=4, random_seed=42)

# Diagnostics
az.plot_trace(trace)
az.summary(trace, hdi_prob=0.94)  # 94% HDI
az.plot_posterior(trace)
```

## Hierarchical model

```python
with pm.Model() as hierarchical:
    # Group-level
    mu_group = pm.Normal("mu_group", mu=0, sigma=10)
    sigma_group = pm.HalfNormal("sigma_group", sigma=5)

    # Subject-level (partial pooling)
    mu_subject = pm.Normal("mu_subject", mu=mu_group, sigma=sigma_group,
                           shape=n_subjects)
    sigma = pm.HalfNormal("sigma", sigma=5)

    obs = pm.Normal("obs", mu=mu_subject[subject_idx], sigma=sigma,
                    observed=y)
    trace = pm.sample(2000, tune=1000)
```

## Diagnostics checklist

1. R-hat < 1.01 for all parameters
2. ESS > 400 (effective sample size)
3. No divergences (or < 0.1% of samples)
4. Posterior predictive checks: simulated data resembles observed
5. Prior predictive checks: priors produce plausible data ranges
