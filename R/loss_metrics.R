#######################################################################################
################################# Loss Function & Metrics #############################
#######################################################################################

cross_cov_penalty <- function(mu, cov) {
  B <- mu$size(1)
  
  mu_c  <- mu  - mu$mean(dim = 1, keepdim = TRUE)
  cov_c <- cov - cov$mean(dim = 1, keepdim = TRUE)
  
  cov_mat <- torch_matmul(mu_c$t(), cov_c) / max(B - 1, 1)
  
  torch_mean(cov_mat$pow(2))
}

make_vae_loss <- function(lambda_pen = 0) {
  function(pred, target) {
    recon  <- pred$recon
    mu     <- pred$mu
    logvar <- pred$logvar
    cov    <- pred$cov
    
    B <- target$size(1)
    
    recon_loss <- 0.5 * nnf_mse_loss(recon, target, reduction = "sum") / B
    
    kl <- -0.5 * torch_sum(
      1 + logvar - mu$pow(2) - torch_exp(logvar)
    ) / B
    
    pen <- cross_cov_penalty(mu, cov)
    
    recon_loss + kl + lambda_pen * pen
  }
}

recon_metric_gen <- luz::luz_metric(
  name = "recon",
  abbrev = "Recon",
  initialize = function() {
    self$sum <- 0
    self$n <- 0
  },
  update = function(preds, target) {
    B <- target$size(1)
    val <- (0.5 * nnf_mse_loss(preds$recon, target, reduction = "sum") / B)$item()
    self$sum <- self$sum + val * B
    self$n <- self$n + B
  },
  compute = function() self$sum / self$n
)
recon_metric <- recon_metric_gen()

kl_metric_gen <- luz::luz_metric(
  name = "kl",
  abbrev = "KL",
  initialize = function() {
    self$sum <- 0
    self$n <- 0
  },
  update = function(preds, target) {
    B <- target$size(1)
    val <- (-0.5 * torch_sum(
      1 + preds$logvar - preds$mu$pow(2) - torch_exp(preds$logvar)
    ) / B)$item()
    self$sum <- self$sum + val * B
    self$n <- self$n + B
  },
  compute = function() self$sum / self$n
)
kl_metric <- kl_metric_gen()

pen_metric_gen <- luz::luz_metric(
  name = "penalty",
  abbrev = "Pen",
  initialize = function() {
    self$sum <- 0
    self$n <- 0
  },
  update = function(preds, target) {
    B <- target$size(1)
    val <- cross_cov_penalty(preds$mu, preds$cov)$item()
    self$sum <- self$sum + val * B
    self$n <- self$n + B
  },
  compute = function() self$sum / self$n
)
pen_metric <- pen_metric_gen()

active_dims_metric_gen <- function(threshold = 0.01) {
  luz::luz_metric(
    name = "active_dims",
    abbrev = "ActDim",
    initialize = function() {
      self$kl_sum <- NULL
      self$n <- 0
      self$threshold <- threshold
    },
    update = function(preds, target) {
      B <- target$size(1)
      kl_mat <- -0.5 * (1 + preds$logvar - preds$mu$pow(2) - torch_exp(preds$logvar))
      kl_dim_sum <- kl_mat$sum(dim = 1)
      kl_dim_sum_r <- as.numeric(kl_dim_sum$to(device = "cpu"))
      
      if (is.null(self$kl_sum)) {
        self$kl_sum <- kl_dim_sum_r
      } else {
        self$kl_sum <- self$kl_sum + kl_dim_sum_r
      }
      self$n <- self$n + B
    },
    compute = function() {
      kl_mean <- self$kl_sum / self$n
      sum(kl_mean > self$threshold)
    }
  )
}
active_dims_metric <- active_dims_metric_gen(threshold = 0.01)()

