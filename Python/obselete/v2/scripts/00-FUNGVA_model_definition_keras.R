##############################################################
############## Keras Version of FUNGVA ######################
##############################################################

library(keras)
library(tensorflow)

# ------------------- Graph CNN layer ------------------------
graph_cnn_layer <- function(units, mask = NULL, name = NULL) {
  keras::layer_dense(
    units = units,
    activation = "linear",
    use_bias = TRUE,
    name = name,
    kernel_initializer = "glorot_uniform"
  ) %>% 
    list(mask = mask)
}

# ------------------- FiLM modulation function ----------------
film_modulate <- function(h, cov_input, film_net, out_dim) {
  film_params <- film_net(cov_input)
  gamma <- film_params[, 1:out_dim]
  beta  <- film_params[, (out_dim+1):(2*out_dim)]
  (1 + gamma) * h + beta
}

# ------------------- Build FUNGVA model ---------------------
build_fungva_keras <- function(latent_dim = 12, cov_dim = 7, masks = NULL) {
  # Input layers
  p <- 68 * (68 - 1) / 2
  fc_input <- layer_input(shape = c(p), name = "fc_input")
  cov_input <- layer_input(shape = c(cov_dim), name = "cov_input")
  
  # ---------------- Encoder ----------------
  en_mu <- fc_input %>%
    layer_dense(1024, activation = "relu") %>%
    layer_dense(128, activation = "relu") %>%
    layer_dense(latent_dim, name = "mu")
  
  en_logvar <- fc_input %>%
    layer_dense(1024, activation = "relu") %>%
    layer_dense(128, activation = "relu") %>%
    layer_dense(latent_dim, name = "logvar")
  
  # Reparameterization
  z <- layer_lambda(function(inputs) {
    mu <- inputs[[1]]
    logvar <- inputs[[2]]
    eps <- k_random_normal(shape = k_shape(mu))
    mu + eps * k_exp(0.5 * logvar)
  })(list(en_mu, en_logvar))
  
  # ---------------- FiLM nets ----------------
  film_hidden <- 32
  create_film_net <- function(name) {
    cov_input %>%
      layer_dense(film_hidden, activation = "relu") %>%
      layer_dense(2 * 68, activation = "linear", name = name)
  }
  
  film1 <- create_film_net("film1")
  film2 <- create_film_net("film2")
  film3 <- create_film_net("film3")
  film4 <- create_film_net("film4")
  film5 <- create_film_net("film5")
  
  # ---------------- Decoder heads ----------------
  decoder_head <- function(z, film_net, mask = NULL) {
    h <- z %>%
      layer_dense(68, activation = "relu")
    h <- film_modulate(h, cov_input, film_net, out_dim = 68)
    # Apply graph mask if provided
    if (!is.null(mask)) {
      h <- layer_lambda(function(x) x * mask)(h)
    }
    # Outer product to reconstruct 68x68
    h_out <- layer_lambda(function(x) {
      x_exp <- k_expand_dims(x, axis = 2L)
      k_batch_dot(x_exp, k_permute_dimensions(x_exp, c(1L,3L,2L)))
    })(h)
    # Flatten lower triangle
    # Here you may need to implement indexing outside Keras if necessary
    h_vec <- layer_lambda(function(x) {
      k.reshape(x, shape = c(-1, p))
    })(h_out)
    h_vec
  }
  
  dcl1_vec <- decoder_head(z, film1, mask = masks[[1]])
  dcl2_vec <- decoder_head(z, film2, mask = masks[[2]])
  dcl3_vec <- decoder_head(z, film3, mask = masks[[3]])
  dcl4_vec <- decoder_head(z, film4, mask = masks[[4]])
  dcl5_vec <- decoder_head(z, film5, mask = masks[[5]])
  
  # Sum all heads + intercept
  dcl_sum <- layer_add(list(dcl1_vec, dcl2_vec, dcl3_vec, dcl4_vec, dcl5_vec))
  if (!is.null(masks[[6]])) {
    dcl_sum <- layer_lambda(function(x) x * masks[[6]])(dcl_sum)
  }
  
  fc_pred <- dcl_sum %>% layer_activation("tanh")
  
  # ---------------- Model ----------------
  model <- keras_model(
    inputs = list(fc_input, cov_input),
    outputs = list(
      recon = dcl_sum,
      fc_pred = fc_pred,
      mu = en_mu,
      logvar = en_logvar,
      z = z,
      cov = cov_input
    )
  )
  
  model
}