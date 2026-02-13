---
name: causal-inference
description: Causal inference methods. DAGs, do-calculus intuition, instrumental variables, difference-in-differences, regression discontinuity.
metadata:
    skill-author: Albert Ying
---

# Causal inference

## When to use

- Estimating causal effects from observational data
- Drawing DAGs to identify confounders
- Choosing between IV, DiD, RDD, or matching
- Interpreting causal vs associational claims

## DAG-based reasoning

```python
# Use DAGs to identify adjustment sets
# Minimal: condition on confounders, never on colliders or mediators

# Example DAG: Treatment -> Outcome, Confounder -> Treatment, Confounder -> Outcome
# Adjustment set: {Confounder}
# Do NOT condition on: Mediator (blocks causal path), Collider (creates bias)
```

## Difference-in-differences

```python
import statsmodels.formula.api as smf

# Requires: treated/control groups, pre/post periods
# Assumption: parallel trends in absence of treatment
model = smf.ols("outcome ~ treated * post + C(unit) + C(time)", data=df).fit()
# The interaction coefficient (treated:post) is the DiD estimate
print(model.summary().tables[1])
```

## Instrumental variables

```python
from linearmodels.iv import IV2SLS

# Instrument Z: affects Treatment but not Outcome directly
model = IV2SLS.from_formula("outcome ~ 1 + covariates [treatment ~ instrument]", data=df)
result = model.fit()

# Check first-stage F-statistic > 10 (weak instrument test)
```

## Decision guide

| Method | When to use | Key assumption |
|--------|-------------|----------------|
| Matching/IPW | Rich covariates, no unobserved confounding | Conditional ignorability |
| IV | Have a valid instrument | Exclusion restriction |
| DiD | Panel data, treatment at known time | Parallel trends |
| RDD | Treatment assigned by threshold | Continuity at cutoff |
