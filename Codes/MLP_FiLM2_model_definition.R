##############################################################
################ MLP Decoder Baseline Module #################
##############################################################

# Assumes:
# - p = 2278 lower-triangular FC edges for 68 ROIs
# - same encoder structure as graph model
# - same covariate FiLM idea, but decoder is plain MLP instead of GraphCNN heads

FUNGVA_MLP_FiLM2 <- nn_module(
  classname = "FUNGVA_MLP_FiLM2",
  
  initialize = function(latent_dim, cov_dim) {
    self$latent_dim <- latent_dim
    self$cov_dim <- cov_dim
    
    p <- (68 * 67) / 2  # 2278
    
    # ---------------- Encoder (same as graph model) ----------------
    self$en_mu_la1  <- nn_linear(p, 1024, bias = FALSE)
    self$en_var_la1 <- nn_linear(p, 1024, bias = FALSE)
    self$en_mu_la2  <- nn_linear(1024, 128, bias = FALSE)
    self$en_var_la2 <- nn_linear(1024, 128, bias = FALSE)
    self$en_mu_la3  <- nn_linear(128, latent_dim)
    self$en_var_la3 <- nn_linear(128, latent_dim)
    
    # ---------------- MLP decoder trunk ----------------
    # latent + covariate-conditioned FiLM on hidden layers
    dec_hidden1 <- 128
    dec_hidden2 <- 1024
    
    self$dc_la1 <- nn_linear(latent_dim, dec_hidden1)
    self$dc_la2 <- nn_linear(dec_hidden1, dec_hidden2)
    self$dc_out <- nn_linear(dec_hidden2, p)
    
    # ---------------- FiLM generators for decoder hidden layers ----------------
    film_hidden <- 32
    
    self$film1 <- nn_sequential(
      nn_linear(cov_dim, film_hidden),
      nn_relu(),
      nn_linear(film_hidden, 2 * dec_hidden1)
    )
    
    self$film2 <- nn_sequential(
      nn_linear(cov_dim, film_hidden),
      nn_relu(),
      nn_linear(film_hidden, 2 * dec_hidden2)
    )
    
    self$tanh_act <- nn_tanh()
  },
  
  encode = function(x) {
    mu <- x %>%
      self$en_mu_la1() %>%
      nnf_relu() %>%
      self$en_mu_la2() %>%
      nnf_relu() %>%
      self$en_mu_la3()
    
    logvar <- x %>%
      self$en_var_la1() %>%
      nnf_relu() %>%
      self$en_var_la2() %>%
      nnf_relu() %>%
      self$en_var_la3()
    
    list(mu = mu, logvar = logvar)
  },
  
  reparameterize = function(mu, logvar) {
    std <- torch_exp(0.5 * logvar)
    eps <- torch_randn_like(std)
    mu + eps * std
  },
  
  film_modulate = function(h, cov, film_net, out_dim) {
    film_params <- film_net(cov)
    gamma <- film_params$narrow(2, 1, out_dim)
    beta  <- film_params$narrow(2, out_dim + 1, out_dim)
    (1 + gamma) * h + beta
  },
  
  decode = function(z, cov) {
    h1 <- z %>%
      self$dc_la1() %>%
      nnf_relu()
    h1 <- self$film_modulate(h1, cov, self$film1, out_dim = h1$size(2))
    
    h2 <- h1 %>%
      self$dc_la2() %>%
      nnf_relu()
    h2 <- self$film_modulate(h2, cov, self$film2, out_dim = h2$size(2))
    
    dc_output <- self$dc_out(h2)
    fc_pred <- self$tanh_act(dc_output)
    
    list(
      recon = dc_output,  # Fisher-z scale
      fc_pred = fc_pred   # correlation scale
    )
  },
  
  forward = function(x) {
    fc  <- x$fc
    cov <- x$cov
    
    enc <- self$encode(fc)
    mu <- enc$mu
    logvar <- enc$logvar
    
    z <- self$reparameterize(mu, logvar)
    dec <- self$decode(z, cov)
    
    list(
      recon = dec$recon,
      mu = mu,
      logvar = logvar,
      fc_pred = dec$fc_pred,
      z = z,
      cov = cov
    )
  }
)

FUNGVA_MLP_FiLM2_Trainer <- nn_module(
  classname = "FUNGVA_MLP_FiLM2_Trainer",
  
  initialize = function(latent_dim, cov_dim) {
    self$model <- FUNGVA_MLP_FiLM2(latent_dim = latent_dim, cov_dim = cov_dim)
  },
  
  forward = function(x) {
    self$model(x)
  }
)











