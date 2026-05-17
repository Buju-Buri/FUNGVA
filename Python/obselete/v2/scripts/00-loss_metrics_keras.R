##############################################################
################### Keras VAE Loss & Metrics ################
##############################################################

# ---------------- Setup Python & Keras ----------------------
library(reticulate)

# Use a stable Python (adjust path if needed)
# This should point to a Python with tensorflow/keras installed
use_python("/usr/bin/python3", required = TRUE)  # <-- change to your Python path
# Alternatively, use a conda environment:
# use_condaenv("r-tensorflow", required = TRUE)

# Load keras3 and tensorflow
library(keras3)
library(tensorflow)

# ---------------- Cross-Covariance Penalty -----------------
cross_cov_penalty_keras <- function(mu, cov) {
  mu_c  <- mu - k_mean(mu, axis = 1, keepdims = TRUE)
  cov_c <- cov - k_mean(cov, axis = 1, keepdims = TRUE)
  
  cov_mat <- k_dot(k_transpose(mu_c), cov_c) / (k_cast(k_shape(mu)[1] - 1, "float32") + 1e-6)
  k_mean(k_square(cov_mat))
}

# ---------------- VAE Loss Function ------------------------
make_vae_loss_keras <- function(lambda_pen = 0) {
  function(y_true, y_pred) {
    recon  <- y_pred$recon      # predicted FC (batch, 2278)
    mu     <- y_pred$mu         # latent mean (batch, latent_dim)
    logvar <- y_pred$logvar     # latent logvar (batch, latent_dim)
    cov    <- y_pred$cov        # covariates (batch, cov_dim)
    
    B <- k_cast(k_shape(recon)[1], "float32")
    
    # Reconstruction loss
    recon_loss <- 0.5 * k_sum(k_square(recon - y_true)) / B
    
    # KL divergence
    kl_loss <- -0.5 * k_sum(1 + logvar - k_square(mu) - k_exp(logvar)) / B
    
    # Cross-covariance penalty
    pen_loss <- cross_cov_penalty_keras(mu, cov)
    
    recon_loss + kl_loss + lambda_pen * pen_loss
  }
}

# ---------------- Metrics -------------------------
recon_metric_keras <- custom_metric("recon", function(y_true, y_pred) {
  B <- k_cast(k_shape(y_true)[1], "float32")
  k_sum(k_square(y_pred$recon - y_true)) * 0.5 / B
})

kl_metric_keras <- custom_metric("kl", function(y_true, y_pred) {
  B <- k_cast(k_shape(y_true)[1], "float32")
  -0.5 * k_sum(1 + y_pred$logvar - k_square(y_pred$mu) - k_exp(y_pred$logvar)) / B
})

pen_metric_keras <- custom_metric("penalty", function(y_true, y_pred) {
  cross_cov_penalty_keras(y_pred$mu, y_pred$cov)
})

active_dims_metric_keras <- custom_metric("active_dims", function(y_true, y_pred) {
  threshold <- 0.01
  kl_mat <- -0.5 * (1 + y_pred$logvar - k_square(y_pred$mu) - k_exp(y_pred$logvar))
  kl_dim_sum <- k_sum(kl_mat, axis = 0L)
  kl_mean <- kl_dim_sum / k_cast(k_shape(y_true)[1], "float32")
  k_sum(k_cast(kl_mean > threshold, "float32"))
})