##############################################################
############## Building the Modules for the model ############
##############################################################

GraphCNN <- nn_module(
  classname = "GraphCNN",
  
  initialize = function(in_features, out_features, bias = TRUE) {
    self$linear <- nn_linear(in_features, out_features, bias = bias)
    self$mask <- NULL
  },
  
  set_mask = function(mask) {
    self$mask <- mask$detach()$to(
      dtype  = self$linear$weight$dtype,
      device = self$linear$weight$device
    )
    self$mask$requires_grad_(FALSE)
  },
  
  forward = function(x) {
    if (!is.null(self$mask)) {
      w <- self$linear$weight * self$mask
      nnf_linear(x, w, self$linear$bias)
    } else {
      self$linear(x)
    }
  }
)

FUNGVA <- nn_module(
  classname = "FUNGVA",
  
  initialize = function(latent_dim, cov_dim) {
    self$latent_dim <- latent_dim
    self$cov_dim <- cov_dim
    
    p <- (68 * 67) / 2
    
    idx <- torch_tril(torch_ones(68, 68, dtype = torch_bool()), diagonal = -1)$nonzero()
    lt_flat_idx <- ((idx[, 1] - 1) * 68 + idx[, 2])$to(dtype = torch_long())
    self$register_buffer("lt_flat_idx", lt_flat_idx)
    
    # Encoder
    self$en_mu_la1  <- nn_linear(p, 1024, bias = FALSE)
    self$en_var_la1 <- nn_linear(p, 1024, bias = FALSE)
    self$en_mu_la2  <- nn_linear(1024, 128, bias = FALSE)
    self$en_var_la2 <- nn_linear(1024, 128, bias = FALSE)
    self$en_mu_la3  <- nn_linear(128, latent_dim)
    self$en_var_la3 <- nn_linear(128, latent_dim)
    
    # Decoder heads
    self$dc1_la1 <- nn_linear(latent_dim, 68)
    self$dc2_la1 <- nn_linear(latent_dim, 68)
    self$dc3_la1 <- nn_linear(latent_dim, 68)
    self$dc4_la1 <- nn_linear(latent_dim, 68)
    self$dc5_la1 <- nn_linear(latent_dim, 68)
    
    self$dc1_la2 <- GraphCNN(68, 68)
    self$dc2_la2 <- GraphCNN(68, 68)
    self$dc3_la2 <- GraphCNN(68, 68)
    self$dc4_la2 <- GraphCNN(68, 68)
    self$dc5_la2 <- GraphCNN(68, 68)
    
    self$dcintercept <- GraphCNN(p, p)
    
    # FiLM nets
    film_hidden <- 32
    
    self$film1 <- nn_sequential(
      nn_linear(cov_dim, film_hidden),
      nn_relu(),
      nn_linear(film_hidden, 2 * 68)
    )
    self$film2 <- nn_sequential(
      nn_linear(cov_dim, film_hidden),
      nn_relu(),
      nn_linear(film_hidden, 2 * 68)
    )
    self$film3 <- nn_sequential(
      nn_linear(cov_dim, film_hidden),
      nn_relu(),
      nn_linear(film_hidden, 2 * 68)
    )
    self$film4 <- nn_sequential(
      nn_linear(cov_dim, film_hidden),
      nn_relu(),
      nn_linear(film_hidden, 2 * 68)
    )
    self$film5 <- nn_sequential(
      nn_linear(cov_dim, film_hidden),
      nn_relu(),
      nn_linear(film_hidden, 2 * 68)
    )
    
    self$tanh_act <- nn_tanh()
  },
  
  set_mask = function(masks) {
    self$dc1_la2$set_mask(masks[[1]])
    self$dc2_la2$set_mask(masks[[2]])
    self$dc3_la2$set_mask(masks[[3]])
    self$dc4_la2$set_mask(masks[[4]])
    self$dc5_la2$set_mask(masks[[5]])
    self$dcintercept$set_mask(masks[[6]])
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
  
  film_modulate = function(h, cov, film_net) {
    film_params <- film_net(cov)
    gamma <- film_params$narrow(2, 1, 68)
    beta  <- film_params$narrow(2, 69, 68)
    (1 + gamma) * h + beta
  },
  
  decode = function(z, cov) {
    lt_idx <- self$lt_flat_idx
    
    dcl1_pre <- z %>% self$dc1_la1() %>% nnf_relu()
    dcl1_mod <- self$film_modulate(dcl1_pre, cov, self$film1)
    dcl1 <- self$dc1_la2(dcl1_mod)
    dcl1_out <- torch_bmm(dcl1$unsqueeze(3), dcl1$unsqueeze(2))
    dcl1_vec <- dcl1_out$reshape(c(dcl1_out$size(1), 68 * 68))$index_select(dim = 2, index = lt_idx)
    
    dcl2_pre <- z %>% self$dc2_la1() %>% nnf_relu()
    dcl2_mod <- self$film_modulate(dcl2_pre, cov, self$film2)
    dcl2 <- self$dc2_la2(dcl2_mod)
    dcl2_out <- torch_bmm(dcl2$unsqueeze(3), dcl2$unsqueeze(2))
    dcl2_vec <- dcl2_out$reshape(c(dcl2_out$size(1), 68 * 68))$index_select(dim = 2, index = lt_idx)
    
    dcl3_pre <- z %>% self$dc3_la1() %>% nnf_relu()
    dcl3_mod <- self$film_modulate(dcl3_pre, cov, self$film3)
    dcl3 <- self$dc3_la2(dcl3_mod)
    dcl3_out <- torch_bmm(dcl3$unsqueeze(3), dcl3$unsqueeze(2))
    dcl3_vec <- dcl3_out$reshape(c(dcl3_out$size(1), 68 * 68))$index_select(dim = 2, index = lt_idx)
    
    dcl4_pre <- z %>% self$dc4_la1() %>% nnf_relu()
    dcl4_mod <- self$film_modulate(dcl4_pre, cov, self$film4)
    dcl4 <- self$dc4_la2(dcl4_mod)
    dcl4_out <- torch_bmm(dcl4$unsqueeze(3), dcl4$unsqueeze(2))
    dcl4_vec <- dcl4_out$reshape(c(dcl4_out$size(1), 68 * 68))$index_select(dim = 2, index = lt_idx)
    
    dcl5_pre <- z %>% self$dc5_la1() %>% nnf_relu()
    dcl5_mod <- self$film_modulate(dcl5_pre, cov, self$film5)
    dcl5 <- self$dc5_la2(dcl5_mod)
    dcl5_out <- torch_bmm(dcl5$unsqueeze(3), dcl5$unsqueeze(2))
    dcl5_vec <- dcl5_out$reshape(c(dcl5_out$size(1), 68 * 68))$index_select(dim = 2, index = lt_idx)
    
    dcl_out <- dcl1_vec + dcl2_vec + dcl3_vec + dcl4_vec + dcl5_vec
    
    dc_output <- self$dcintercept(dcl_out)
    fc_pred <- self$tanh_act(dc_output)
    
    list(
      recon = dc_output,
      fc_pred = fc_pred
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

FUNGVA_Trainer <- nn_module(
  classname = "FUNGVA_Trainer",
  
  initialize = function(latent_dim, cov_dim, masks = NULL) {
    self$model <- FUNGVA(latent_dim = latent_dim, cov_dim = cov_dim)
    if (!is.null(masks)) self$model$set_mask(masks)
  },
  
  forward = function(x) {
    self$model(x)
  }
)













