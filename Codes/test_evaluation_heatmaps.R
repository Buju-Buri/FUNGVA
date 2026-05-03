##############################################################
######## COMBINED TEST-SET HEATMAPS FOR ALL MODELS ##########
##############################################################

##############################################################
######################## Setup ###############################
##############################################################

source("libraries.R")
source("datasets.R")
source("loss_metrics.R")
source("FUNGVA_model_definition.R")
source("MLP_FiLM1_model_definition.R")
source("MLP_FiLM2_model_definition.R")

##############################################################
######################## User settings #######################
##############################################################

seed_vec <- c(101, 202, 303)

model_specs <- tibble::tribble(
  ~model,          ~dir_path, ~file_prefix,
  "Graph_FUNGVA",  "FUNGVA_ld12_lam0.1",   "graph_ld12_lam0.1_seed_",
  "MLP_FiLM1",     "MLP_FiLM1_ld10_lam0.1", "MLP_FiLM1_ld10_lam0.1_seed_",
  "MLP_FiLM2",     "MLP_FiLM2_ld10_lam1",   "MLP_FiLM2_ld10_lam1_seed_"
)

output_dir <- "test_evaluation"
save_dir <- file.path(output_dir, "heatmaps_combined")
dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)

##############################################################
############### Evaluation dataloader (NO shuffle) ###########
##############################################################

eval_test_dl <- dataloader(test_ds, batch_size = 15, shuffle = FALSE)

##############################################################
################## VECTOR -> MATRIX HELPER ###################
##############################################################

vec_to_mat <- function(v, p = 68) {
  M <- matrix(0, p, p)
  idx <- 1L
  for (i in 2:p) {
    for (j in 1:(i - 1)) {
      M[i, j] <- v[idx]
      M[j, i] <- v[idx]
      idx <- idx + 1L
    }
  }
  diag(M) <- 0
  M
}

make_df <- function(delta_vec, panel_label) {
  M <- vec_to_mat(delta_vec, 68)
  
  expand.grid(row = 1:68, col = 1:68) %>%
    mutate(
      value = as.vector(M),
      panel = panel_label
    )
}

##############################################################
################ GROUP DIFFERENCE FUNCTION ###################
##############################################################

compute_delta <- function(mat, group_vec) {
  idx0 <- which(group_vec == 0)
  idx1 <- which(group_vec == 1)
  
  colMeans(mat[idx1, , drop = FALSE]) -
    colMeans(mat[idx0, , drop = FALSE])
}

##############################################################
################## OBSERVED DELTAS (ONCE) ####################
##############################################################

# correlation scale
obs_mat <- tanh(fc_test_atanh)

# APOE4: 1 - 0 = carrier - non-carrier
# SEX:   1 - 0 = male - female
delta_obs_apoe <- compute_delta(obs_mat, cov_test$APOE4)
delta_obs_sex  <- compute_delta(obs_mat, cov_test$SEX)

##############################################################
################## PREDICTION COLLECTION #####################
##############################################################

collect_pred_mat <- function(fitted_obj, dl) {
  model <- fitted_obj$model
  model$eval()
  
  pred_list <- list()
  i <- 1L
  
  coro::loop(for (batch in dl) {
    x <- batch[[1]]
    
    with_no_grad({
      out <- model(x)
    })
    
    pred_list[[i]] <- as.matrix(out$fc_pred$to(device = "cpu"))
    i <- i + 1L
  })
  
  do.call(rbind, pred_list)
}

##############################################################
############### AVERAGE PREDICTED DELTAS #####################
##############################################################

pred_delta_apoe_all <- list()
pred_delta_sex_all  <- list()

for (m in seq_len(nrow(model_specs))) {
  
  model_name   <- model_specs$model[m]
  model_dir    <- model_specs$dir_path[m]
  file_prefix  <- model_specs$file_prefix[m]
  display_name <- model_specs$display_name[m]
  
  cat("\n====================================\n")
  cat("Processing model:", display_name, "\n")
  cat("====================================\n")
  
  delta_pred_apoe_list <- list()
  delta_pred_sex_list  <- list()
  
  for (sd in seed_vec) {
    cat("  Seed:", sd, "\n")
    
    model_path <- file.path(model_dir, paste0(file_prefix, sd, ".pt"))
    fitted <- luz_load(model_path)
    
    if (model_name == "Graph_FUNGVA") {
      fitted$model$model$set_mask(masks)
    }
    
    pred_mat <- collect_pred_mat(fitted, eval_test_dl)
    
    delta_pred_apoe_list[[as.character(sd)]] <-
      compute_delta(pred_mat, cov_test$APOE4)
    
    delta_pred_sex_list[[as.character(sd)]] <-
      compute_delta(pred_mat, cov_test$SEX)
  }
  
  pred_delta_apoe_all[[display_name]] <-
    Reduce("+", delta_pred_apoe_list) / length(seed_vec)
  
  pred_delta_sex_all[[display_name]] <-
    Reduce("+", delta_pred_sex_list) / length(seed_vec)
}

##############################################################
#################### COMBINED APOE4 FIGURE ###################
##############################################################

df_apoe_combined <- bind_rows(
  make_df(delta_obs_apoe, "Observed"),
  make_df(pred_delta_apoe_all[["Graph-FUNGVA"]], "Graph-FUNGVA"),
  make_df(pred_delta_apoe_all[["MLP-FiLM1"]],    "MLP-FiLM1"),
  make_df(pred_delta_apoe_all[["MLP-FiLM2"]],    "MLP-FiLM2")
)

df_apoe_combined$panel <- factor(
  df_apoe_combined$panel,
  levels = c("Observed", "Graph-FUNGVA", "MLP-FiLM1", "MLP-FiLM2")
)

lim_apoe <- max(abs(df_apoe_combined$value), na.rm = TRUE)

p_apoe_combined <- ggplot(df_apoe_combined, aes(x = col, y = row, fill = value)) +
  geom_tile() +
  facet_wrap(~ panel, nrow = 1) +
  scale_fill_gradient2(
    low = "blue",
    mid = "white",
    high = "red",
    limits = c(-lim_apoe, lim_apoe)
  ) +
  scale_x_continuous(breaks = seq(1, 68, by = 5)) +
  scale_y_reverse(breaks = seq(1, 68, by = 5)) +
  labs(
    title = "APOE4 Group Difference on Test Set (Carrier - Non-carrier)",
    x = NULL,
    y = NULL,
    fill = expression(Delta~FC)
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    strip.text = element_text(face = "bold"),
    axis.text = element_text(size = 7),
    axis.ticks = element_line(),
    panel.grid = element_blank()
  )

ggsave(
  filename = file.path(save_dir, "Combined_Heatmap_APOE4.png"),
  plot = p_apoe_combined,
  width = 14,
  height = 4.2,
  dpi = 300
)

print(p_apoe_combined)

##############################################################
##################### COMBINED SEX FIGURE ####################
##############################################################

df_sex_combined <- bind_rows(
  make_df(delta_obs_sex, "Observed"),
  make_df(pred_delta_sex_all[["Graph-FUNGVA"]], "Graph-FUNGVA"),
  make_df(pred_delta_sex_all[["MLP-FiLM1"]],    "MLP-FiLM1"),
  make_df(pred_delta_sex_all[["MLP-FiLM2"]],    "MLP-FiLM2")
)

df_sex_combined$panel <- factor(
  df_sex_combined$panel,
  levels = c("Observed", "Graph-FUNGVA", "MLP-FiLM1", "MLP-FiLM2")
)

lim_sex <- max(abs(df_sex_combined$value), na.rm = TRUE)

p_sex_combined <- ggplot(df_sex_combined, aes(x = col, y = row, fill = value)) +
  geom_tile() +
  facet_wrap(~ panel, nrow = 1) +
  scale_fill_gradient2(
    low = "blue",
    mid = "white",
    high = "red",
    limits = c(-lim_sex, lim_sex)
  ) +
  scale_x_continuous(breaks = seq(1, 68, by = 5)) +
  scale_y_reverse(breaks = seq(1, 68, by = 5)) +
  labs(
    title = "SEX Group Difference on Test Set (Male - Female)",
    x = NULL,
    y = NULL,
    fill = expression(Delta~FC)
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    strip.text = element_text(face = "bold"),
    axis.text = element_text(size = 7),
    axis.ticks = element_line(),
    panel.grid = element_blank()
  )

ggsave(
  filename = file.path(save_dir, "Combined_Heatmap_SEX.png"),
  plot = p_sex_combined,
  width = 14,
  height = 4.2,
  dpi = 300
)

print(p_sex_combined)

##############################################################
######################## DONE ###############################
##############################################################

cat("\nSaved combined heatmaps to:\n", save_dir, "\n")