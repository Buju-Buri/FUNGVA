source("libraries.R")
source("FUNGVA_model_definition.R")
source("datasets.R")
source("loss_metrics.R")


##############################################################
################### Helper Functions #########################
##############################################################

extract_lr_stats <- function(rates_and_losses) {
  df <- as.data.frame(rates_and_losses)
  
  lr_col   <- grep("lr", names(df), value = TRUE)[1]
  loss_col <- grep("loss", names(df), value = TRUE)[1]
  
  lr   <- df[[lr_col]]
  loss <- df[[loss_col]]
  
  idx_min <- which.min(loss)
  lr_min_loss <- lr[idx_min]
  
  idx_div <- which(loss > 4 * min(loss))[1]
  if (!is.na(idx_div)) {
    lr_diverge <- lr[idx_div]
  } else {
    lr_diverge <- max(lr)
  }
  
  data.frame(
    lr_min_loss = lr_min_loss,
    lr_diverge = lr_diverge
  )
}

choose_max_lr_auto <- function(
    lr_results,
    k = 10,
    cv_threshold = 0.5,
    p_safe = 0.25,
    p_stable = 0.50
) {
  div <- lr_results$lr_diverge
  min_loss <- lr_results$lr_min_loss
  
  div <- div[is.finite(div) & !is.na(div) & div > 0]
  min_loss <- min_loss[is.finite(min_loss) & !is.na(min_loss) & min_loss > 0]
  
  if (length(div) < 3 || length(min_loss) < 3) {
    stop("Need at least 3 LR finder runs.")
  }
  
  cv_div <- stats::sd(div) / mean(div)
  cv_min <- stats::sd(min_loss) / mean(min_loss)
  
  if (cv_div >= 1) {
    div_stat <- min(div)
    div_method <- "min"
  } else if (cv_div >= cv_threshold) {
    div_stat <- as.numeric(stats::quantile(div, p_safe))
    div_method <- paste0("quantile_", p_safe)
  } else {
    div_stat <- as.numeric(stats::quantile(div, p_stable))
    div_method <- paste0("quantile_", p_stable)
  }
  
  if (cv_min >= 1) {
    min_stat <- min(min_loss)
    min_method <- "min"
  } else if (cv_min >= cv_threshold) {
    min_stat <- as.numeric(stats::quantile(min_loss, p_safe))
    min_method <- paste0("quantile_", p_safe)
  } else {
    min_stat <- as.numeric(stats::quantile(min_loss, p_stable))
    min_method <- paste0("quantile_", p_stable)
  }
  
  lr_from_div <- div_stat / k
  lr_from_min <- min_stat
  max_lr <- min(lr_from_min, lr_from_div)
  
  list(
    max_lr = max_lr,
    cv_diverge = cv_div,
    cv_min_loss = cv_min,
    div_method = div_method,
    min_method = min_method,
    lr_from_div = lr_from_div,
    lr_from_min = lr_from_min
  )
}

run_lr_search_for_config <- function(
    latent_dim,
    lambda_pen,
    cov_dim,
    train_dl,
    masks,
    n_runs = 5,
    start_lr = 1e-5,
    end_lr = 3e-1,
    plot_dir = NULL
) {
  lr_runs <- vector("list", n_runs)
  lr_stats_list <- vector("list", n_runs)
  
  loss_fn <- make_vae_loss(lambda_pen)
  
  for (i in seq_len(n_runs)) {
    set.seed(1000 + latent_dim * 100 + round(lambda_pen * 1000) + i)
    torch_manual_seed(1000 + latent_dim * 100 + round(lambda_pen * 1000) + i)
    
    model_tmp <- FUNGVA_Trainer %>%
      setup(loss = loss_fn, optimizer = optim_adam)
    
    lr_fit <- model_tmp %>%
      set_hparams(latent_dim = latent_dim, cov_dim = cov_dim, masks = masks) %>%
      lr_finder(train_dl, start_lr = start_lr, end_lr = end_lr)
    
    lr_runs[[i]] <- lr_fit
    lr_stats_list[[i]] <- extract_lr_stats(lr_fit)
  }
  
  lr_results <- dplyr::bind_rows(lr_stats_list)
  lr_choice <- choose_max_lr_auto(lr_results)
  
  if (!is.null(plot_dir)) {
    p <- plot(lr_runs[[1]])
    ggsave(
      filename = file.path(
        plot_dir,
        paste0("lr_finder_latent_", latent_dim, "_lambda_", lambda_pen, ".png")
      ),
      plot = p,
      width = 7,
      height = 5,
      dpi = 300
    )
  }
  
  list(
    lr_runs = lr_runs,
    lr_results = lr_results,
    lr_choice = lr_choice
  )
}

get_collapse_flag <- function(total_kl, active_dims, latent_dim,
                              kl_tol_per_dim = 0.01,
                              min_active_ratio = 0.1) {
  kl_flag <- total_kl < (kl_tol_per_dim * latent_dim)
  active_flag <- active_dims < ceiling(min_active_ratio * latent_dim)
  kl_flag || active_flag
}

extract_best_metrics <- function(fitted_obj) {
  metrics_train <- as.data.frame(do.call(rbind, fitted_obj$records$metrics$train))
  metrics_valid <- as.data.frame(do.call(rbind, fitted_obj$records$metrics$valid))
  
  best_idx <- which.min(metrics_valid$loss)
  
  tibble::tibble(
    best_epoch       = best_idx,
    train_total_loss = unlist(metrics_train$loss[best_idx]),
    val_total_loss   = unlist(metrics_valid$loss[best_idx]),
    train_recon      = unlist(metrics_train$recon[best_idx]),
    val_recon        = unlist(metrics_valid$recon[best_idx]),
    train_kl         = unlist(metrics_train$kl[best_idx]),
    val_kl           = unlist(metrics_valid$kl[best_idx]),
    train_pen        = unlist(metrics_train$pen[best_idx]),
    val_pen          = unlist(metrics_valid$pen[best_idx]),
    train_act_dim    = unlist(metrics_train$actdim[best_idx]),
    val_act_dim      = unlist(metrics_valid$actdim[best_idx])
  )
}

fit_one_config <- function(
    latent_dim,
    lambda_pen,
    cov_dim,
    train_dl,
    val_dl,
    masks,
    epochs = 800,
    patience = 30,
    lr_plot_dir = NULL,
    csv_log_dir = NULL,
    fitted_plot_dir = NULL
) {
  cat("\n=============================\n")
  cat("Running latent_dim =", latent_dim, " lambda =", lambda_pen, "\n")
  cat("=============================\n")
  
  lr_info <- run_lr_search_for_config(
    latent_dim = latent_dim,
    lambda_pen = lambda_pen,
    cov_dim = cov_dim,
    train_dl = train_dl,
    masks = masks,
    n_runs = 5,
    plot_dir = lr_plot_dir
  )
  
  max_lr <- lr_info$lr_choice$max_lr
  cat("Chosen max_lr =", max_lr, "\n")
  
  loss_fn <- make_vae_loss(lambda_pen)
  
  model <- FUNGVA_Trainer %>%
    setup(
      loss = loss_fn,
      optimizer = optim_adam,
      metrics = list(recon_metric, kl_metric, pen_metric, active_dims_metric)
    )
  
  set.seed(500 + latent_dim * 100 + round(lambda_pen * 1000))
  torch_manual_seed(500 + latent_dim * 100 + round(lambda_pen * 1000))
  
  callbacks_list <- list(
    luz_callback_early_stopping(patience = patience),
    luz_callback_keep_best_model(),
    luz_callback_lr_scheduler(
      lr_one_cycle,
      max_lr = max_lr,
      epochs = epochs,
      steps_per_epoch = length(train_dl),
      call_on = "on_batch_end"
    )
  )
  
  fitted <- model %>%
    set_hparams(latent_dim = latent_dim, cov_dim = cov_dim, masks = masks) %>%
    fit(
      train_dl,
      epochs = epochs,
      valid_data = val_dl,
      callbacks = callbacks_list,
      verbose = TRUE
    )
  
  if (!is.null(fitted_plot_dir)) {
    p <- plot(fitted)
    ggsave(
      filename = file.path(
        fitted_plot_dir,
        paste0("metrics_latent_", latent_dim, "_lambda_", lambda_pen, ".png")
      ),
      plot = p,
      width = 8,
      height = 6,
      dpi = 300
    )
  }
  
  metric_row <- extract_best_metrics(fitted)
  
  collapse_flag <- get_collapse_flag(
    total_kl = metric_row$val_kl,
    active_dims = metric_row$val_act_dim,
    latent_dim = latent_dim,
    kl_tol_per_dim = 0.01,
    min_active_ratio = 0.1
  )
  
  result_row <- metric_row %>%
    dplyr::mutate(
      latent_dim = latent_dim,
      lambda_pen = lambda_pen,
      max_lr = max_lr,
      collapse_flag = collapse_flag,
      cv_diverge = lr_info$lr_choice$cv_diverge,
      cv_min_loss = lr_info$lr_choice$cv_min_loss,
      lr_div_method = lr_info$lr_choice$div_method,
      lr_min_method = lr_info$lr_choice$min_method
    ) %>%
    dplyr::select(
      latent_dim, lambda_pen, best_epoch, max_lr,
      train_total_loss, val_total_loss,
      train_recon, val_recon,
      train_kl, val_kl,
      train_pen, val_pen,
      train_act_dim, val_act_dim,
      collapse_flag,
      cv_diverge, cv_min_loss,
      lr_div_method, lr_min_method
    )
  
  list(
    fitted = fitted,
    lr_info = lr_info,
    result_row = result_row
  )
}

##############################################################
################### Latent x Lambda Experiment ###############
##############################################################

latent_grid <- c(8, 10, 12, 14, 16, 18, 20, 22, 24)
lambda_grid <- c(0.01, 0.1, 1, 10)

lr_plot_dir <- "lr_plots"
fitted_plot_dir <- "fitted_plots"
summary_csv <- "latent_lambda_summary.csv"

dir.create(lr_plot_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(csv_log_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(fitted_plot_dir, recursive = TRUE, showWarnings = FALSE)

all_runs <- list()
results_table <- tibble::tibble()

k <- 1
for (ld in latent_grid) {
  for (lam in lambda_grid) {
    
    run_k <- fit_one_config(
      latent_dim = ld,
      lambda_pen = lam,
      cov_dim = cov_dim,
      train_dl = train_dl,
      val_dl = val_dl,
      masks = masks,
      epochs = 800,
      patience = 30,
      lr_plot_dir = lr_plot_dir,
      csv_log_dir = csv_log_dir,
      fitted_plot_dir = fitted_plot_dir
    )
    
    all_runs[[k]] <- run_k
    names(all_runs)[k] <- paste0("ld_", ld, "_lam_", lam)
    
    results_table <- dplyr::bind_rows(results_table, run_k$result_row)
    print(results_table)
    
    k <- k + 1
  }
}

results_table <- results_table %>%
  arrange(val_total_loss)

print(results_table)
write.csv(results_table, summary_csv, row.names = FALSE)

##############################################################
##################### Compact Post-processing ################
##############################################################

# Best lambda for each latent dimension
best_by_latent <- results_table %>%
  filter(!collapse_flag) %>%
  group_by(latent_dim) %>%
  arrange(val_total_loss, .by_group = TRUE) %>%
  slice(1) %>%
  ungroup() %>%
  arrange(val_total_loss)

print(best_by_latent)

best_by_latent_csv <- "best_lambda_by_latent.csv"
write.csv(best_by_latent, best_by_latent_csv, row.names = FALSE)


