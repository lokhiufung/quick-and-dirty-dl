# Contrastive Predictive Coding

## Notes
- ...n time series and high-dimensional modeling,approaches that use next step prediction exploit the local smoothness of the signal. When predicting further in the future, the amount of shared information becomes much lower, and the model needsto infer more global structure.
- ...First, a non-linear encoder g_enc maps the input sequence of observations x_t to a sequence of latent representations z_t=g_enc(x_t), potentially with a lower temporal resolution.