source("libraries.R")
source("FUNGVA_model_definition.R")
source("datasets.R")
source("loss_metrics.R")


##############################################################
#################### Training Function #######################
##############################################################

fit_one_seed_graph <- function(
    seed_id,
    train_dl,
    val_dl,
    masks,
    cov_dim,
    latent_dim = 12,
    lambda_pen = 0.1,
    max_lr = 0.025270402,
    epochs = 800,
    patience = 30,
    model_dir = NULL
) {
  cat("\n---------------------------------\n")
  cat("FUNGVA model run:",
      "latent_dim =", latent_dim,
      "lambda_pen =", lambda_pen,
      "seed =", seed_id,
      "max_lr =", max_lr, "\n")
  cat("---------------------------------\n")
  
  set.seed(seed_id)
  torch_manual_seed(seed_id)
  
  loss_fn <- make_vae_loss(lambda_pen)
  
  model <- FUNGVA_Trainer %>%
    setup(
      loss = loss_fn,
      optimizer = optim_adam,
      metrics = list(recon_metric, kl_metric, pen_metric, active_dims_metric)
    )
  
  fitted <- model %>%
    set_hparams(latent_dim = latent_dim, cov_dim = cov_dim, masks = masks) %>%
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
    model_file <- file.path(model_dir, paste0("FUNGVA_ld12_lam0.1_seed_", seed_id, ".pt"))
    luz_save(fitted, model_file)
  }
  
  fitted
}


##############################################################
######################## Run Training ########################
##############################################################

seed_vec <- c(101, 202, 303)

model_dir <- "FUNGVA_ld12_lam0.1"

dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)


for (i in seq_along(seed_vec)) {
  sd <- seed_vec[i]
  
  run_i <- fit_one_seed_graph(
    seed_id = sd,
    train_dl = train_dl,
    val_dl = val_dl,
    masks = masks,
    cov_dim = cov_dim,
    latent_dim = 12,
    lambda_pen = 0.1,
    max_lr = 0.025270402,
    epochs = 800,
    patience = 30,
    model_dir = model_dir
  )
  
}

