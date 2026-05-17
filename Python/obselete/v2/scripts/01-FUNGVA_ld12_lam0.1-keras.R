##############################################################
################### Keras FUNGVA Training ####################
##############################################################

# Keras versions of the original files are needed:
# - libraries_keras.R
# - FUNGVA_model_definition_keras.R
# - datasets_keras.R
# - loss_metrics_keras.R

setwd("~/fungva/working-dir/scripts/")

source("00-libraries_keras.R")
source("00-FUNGVA_model_definition_keras.R")
source("00-datasets_keras.R")
source("00-loss_metrics_keras.R")


##############################################################
################ One-Cycle Learning Rate #####################
##############################################################

make_one_cycle_schedule = function(
    max_lr,
    epochs,
    steps_per_epoch,
    pct_start = 0.3,
    div_factor = 25,
    final_div_factor = 1e4
) {
  total_steps = epochs * steps_per_epoch
  warmup_steps = floor(pct_start * total_steps)
  
  initial_lr = max_lr / div_factor
  final_lr = max_lr / final_div_factor
  
  function(step) {
    step = as.numeric(step)
    
    if (step <= warmup_steps) {
      lr = initial_lr + (max_lr - initial_lr) * (step / warmup_steps)
    } else {
      decay_step = step - warmup_steps
      decay_steps = total_steps - warmup_steps
      lr = max_lr - (max_lr - final_lr) * (decay_step / decay_steps)
    }
    
    max(lr, final_lr)
  }
}


##############################################################
#################### Training Function #######################
##############################################################

fit_one_seed_graph_keras = function(
    seed_id,
    train_x,
    train_y,
    val_x,
    val_y,
    masks,
    cov_dim,
    latent_dim = 12,
    lambda_pen = 0.1,
    max_lr = 0.025270402,
    epochs = 800,
    patience = 30,
    batch_size = 32,
    model_dir = NULL
) {
  cat("\n---------------------------------\n")
  cat(
    "FUNGVA Keras model run:",
    "latent_dim =", latent_dim,
    "lambda_pen =", lambda_pen,
    "seed =", seed_id,
    "max_lr =", max_lr, "\n"
  )
  cat("---------------------------------\n")
  
  set.seed(seed_id)
  tensorflow::tf$random$set_seed(seed_id)
  
  steps_per_epoch = ceiling(nrow(train_y) / batch_size)
  
  lr_schedule = make_one_cycle_schedule(
    max_lr = max_lr,
    epochs = epochs,
    steps_per_epoch = steps_per_epoch
  )
  
  step_counter = 0L
  
  one_cycle_callback = keras::callback_lambda(
    on_train_batch_begin = function(batch, logs = NULL) {
      step_counter <<- step_counter + 1L
      lr = lr_schedule(step_counter)
      keras::k_set_value(self$model$optimizer$learning_rate, lr)
    }
  )
  
  early_stop_callback = keras::callback_early_stopping(
    monitor = "val_loss",
    patience = patience,
    restore_best_weights = TRUE
  )
  
  model_file = NULL
  
  if (!is.null(model_dir)) {
    dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)
    
    model_file = file.path(
      model_dir,
      paste0("FUNGVA_ld12_lam0.1_seed_", seed_id, ".keras")
    )
  }
  
  checkpoint_callback = keras::callback_model_checkpoint(
    filepath = model_file,
    monitor = "val_loss",
    save_best_only = TRUE,
    save_weights_only = FALSE
  )
  
  model = build_fungva_keras(
    latent_dim = latent_dim,
    cov_dim = cov_dim,
    masks = masks
  )
  
  loss_fn = make_vae_loss_keras(lambda_pen = lambda_pen)
  
  model %>% keras::compile(
    optimizer = keras::optimizer_adam(),
    loss = loss_fn,
    metrics = list(
      recon_metric_keras,
      kl_metric_keras,
      pen_metric_keras,
      active_dims_metric_keras
    )
  )
  
  callbacks_list = list(
    early_stop_callback,
    one_cycle_callback
  )
  
  if (!is.null(model_file)) {
    callbacks_list = c(callbacks_list, list(checkpoint_callback))
  }
  
  fitted = model %>% keras::fit(
    x = train_x,
    y = train_y,
    validation_data = list(val_x, val_y),
    epochs = epochs,
    batch_size = batch_size,
    callbacks = callbacks_list,
    verbose = 1
  )
  
  if (!is.null(model_file)) {
    model %>% keras::save_model(model_file)
  }
  
  list(
    model = model,
    history = fitted
  )
}


##############################################################
######################## Run Training ########################
##############################################################

seed_vec = c(101, 202, 303)

model_dir = "output_FUNGVA_ld12_lam0.1_keras"

dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)

seed_runs = vector("list", length(seed_vec))

for (i in seq_along(seed_vec)) {
  sd = seed_vec[i]
  
  run_i = fit_one_seed_graph_keras(
    seed_id = sd,
    train_x = train_x,
    train_y = train_y,
    val_x = val_x,
    val_y = val_y,
    masks = masks,
    cov_dim = cov_dim,
    latent_dim = 12,
    lambda_pen = 0.1,
    max_lr = 0.025270402,
    epochs = 800,
    patience = 30,
    batch_size = 32,
    model_dir = model_dir
  )
  
  seed_runs[[i]] = run_i
}

names(seed_runs) = paste0("seed_", seed_vec)