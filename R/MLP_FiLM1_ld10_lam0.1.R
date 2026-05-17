source("libraries.R")
source("MLP_FiLM1_model_definition.R")
source("datasets.R")
source("loss_metrics.R")


##############################################################
#################### Training Function #######################
##############################################################

fit_one_seed_MLP_FiLM1 <- function(
    seed_id,
    cov_dim,
    train_dl,
    val_dl,
    latent_dim = 10,
    lambda_pen = 0.1,
    max_lr = 0.00538324276491867,
    epochs = 800,
    patience = 30,
    model_dir = NULL
) {
  cat("\n---------------------------------\n")
  cat("MLP_FiLM1 run:",
      "latent_dim =", latent_dim,
      "lambda_pen =", lambda_pen,
      "seed =", seed_id,
      "max_lr =", max_lr, "\n")
  cat("---------------------------------\n")
  
  loss_fn <- make_vae_loss(lambda_pen)
  
  model <- FUNGVA_MLP_FiLM1_Trainer %>%
    setup(
      loss = loss_fn,
      optimizer = optim_adam,
      metrics = list(recon_metric, kl_metric, pen_metric, active_dims_metric)
    )
  
  set.seed(seed_id)
  torch_manual_seed(seed_id)
  
  fitted <- model %>%
    set_hparams(latent_dim = latent_dim, cov_dim = cov_dim) %>%
    fit(
      train_dl,
      epochs = epochs,
      valid_data = val_dl,
      callbacks = list(
        luz_callback_early_stopping(patience = patience),
        luz_callback_keep_best_model(),
        luz_callback_lr_scheduler(
          lr_one_cycle,
          max_lr = max_lr,
          epochs = epochs,
          steps_per_epoch = length(train_dl),
          call_on = "on_batch_end"
        )
      ),
      verbose = TRUE
    )
  
  if (!is.null(model_dir)) {
    model_file <- file.path(
      model_dir,
      paste0("MLP_FiLM1_ld10_lam0.1_seed_", seed_id, ".pt")
    )
    luz_save(fitted, model_file)
  }
  
  fitted
}

seed_vec <- c(101, 202, 303)

model_dir <- "MLP_FiLM1_ld10_lam0.1"
dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)

seed_runs <- vector("list", length(seed_vec))

for (i in seq_along(seed_vec)) {
  sd <- seed_vec[i]
  
  run_i <- fit_one_seed_MLP_FiLM1(
    seed_id = sd,
    cov_dim = cov_dim,
    train_dl = train_dl,
    val_dl = val_dl,
    latent_dim = 10,
    lambda_pen = 0.1,
    max_lr = 0.00538324276491867,
    epochs = 800,
    patience = 30,
    model_dir = model_dir
  )
  
}




