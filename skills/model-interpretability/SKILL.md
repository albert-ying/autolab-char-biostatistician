---
name: model-interpretability
description: ML model interpretability with SHAP. Feature importance, interaction effects, partial dependence, model debugging.
metadata:
    skill-author: Albert Ying
---

# Model interpretability

## When to use

- Explaining black-box model predictions
- Feature importance ranking
- Detecting feature interactions
- Debugging model behavior and bias

## SHAP workflow

```python
import shap

# For tree models (XGBoost, LightGBM, Random Forest)
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# Global importance (beeswarm)
shap.plots.beeswarm(shap_values)

# Single prediction (waterfall)
shap.plots.waterfall(shap_values[0])

# Feature importance (bar)
shap.plots.bar(shap_values)

# Dependence (interaction)
shap.plots.scatter(shap_values[:, "feature_name"], color=shap_values)
```

## For non-tree models

```python
# KernelSHAP (model-agnostic, slower)
explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 100))
shap_values = explainer.shap_values(X_test[:50])

# DeepSHAP (for neural networks)
explainer = shap.DeepExplainer(model, X_train[:100])
shap_values = explainer.shap_values(X_test[:50])
```

## Interpretation guidelines

- SHAP values are additive: sum of all feature contributions = prediction - base value.
- Positive SHAP = pushes prediction higher; negative = pushes lower.
- Global importance: mean absolute SHAP value across samples.
- Always pair SHAP with domain knowledge. Statistical importance is not causal.
- For high-stakes decisions: report confidence intervals on SHAP values using bootstrap.
