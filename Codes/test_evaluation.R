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
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

##############################################################
############### Evaluation dataloaders (NO shuffle) ##########
##############################################################

eval_train_dl <- dataloader(train_ds, batch_size = 16, shuffle = FALSE)
eval_val_dl   <- dataloader(val_ds,   batch_size = 15, shuffle = FALSE)
eval_test_dl  <- dataloader(test_ds,  batch_size = 15, shuffle = FALSE)

split_objects <- list(
  train = list(dl = eval_train_dl, obs_z = fc_train_atanh, cov_raw = cov_train),
  val   = list(dl = eval_val_dl,   obs_z = fc_val_atanh,   cov_raw = cov_val),
  test  = list(dl = eval_test_dl,  obs_z = fc_test_atanh,  cov_raw = cov_test)
)

##############################################################
###################### Helper functions ######################
##############################################################

safe_cor <- function(x, y) {
  ok <- is.finite(x) & is.finite(y)
  x <- x[ok]; y <- y[ok]
  if (length(x) < 3) return(NA_real_)
  if (sd(x) == 0 || sd(y) == 0) return(NA_real_)
  cor(x, y)
}

safe_rmse <- function(x, y) {
  ok <- is.finite(x) & is.finite(y)
  x <- x[ok]; y <- y[ok]
  if (length(x) == 0) return(NA_real_)
  sqrt(mean((x - y)^2))
}

normalize_binary <- function(x) {
  if (is.factor(x)) x <- as.character(x)
  as.numeric(x)
}

collect_preds <- function(fitted, dl) {
  model <- fitted$model
  model$eval()
  
  pred_z <- list()
  pred_r <- list()
  obs_z  <- list()
  
  i <- 1L
  coro::loop(for (batch in dl) {
    x <- batch[[1]]
    y <- batch[[2]]
    
    with_no_grad({
      out <- model(x)
    })
    
    pred_z[[i]] <- as.matrix(out$recon$to(device = "cpu"))
    pred_r[[i]] <- as.matrix(out$fc_pred$to(device = "cpu"))
    obs_z[[i]]  <- as.matrix(y$to(device = "cpu"))
    
    i <- i + 1L
  })
  
  list(
    pred_z = do.call(rbind, pred_z),
    pred_r = do.call(rbind, pred_r),
    obs_z  = do.call(rbind, obs_z)
  )
}

compute_subject_metrics <- function(obs_z, pred_z, ids) {
  obs_r  <- tanh(obs_z)
  pred_r <- tanh(pred_z)
  
  tibble(
    subject_id = ids,
    subject_corr = vapply(
      seq_len(nrow(obs_z)),
      function(i) safe_cor(obs_r[i, ], pred_r[i, ]),
      numeric(1)
    ),
    subject_rmse = vapply(
      seq_len(nrow(obs_z)),
      function(i) safe_rmse(obs_z[i, ], pred_z[i, ]),
      numeric(1)
    )
  )
}

compute_group_metrics <- function(obs_z, pred_z, cov_raw, var) {
  g <- normalize_binary(cov_raw[[var]])
  
  idx0 <- which(g == 0)
  idx1 <- which(g == 1)
  
  if (length(idx0) < 2 || length(idx1) < 2) {
    return(tibble(
      covariate = var,
      n_group0 = length(idx0),
      n_group1 = length(idx1),
      delta_corr = NA_real_,
      delta_rmse = NA_real_
    ))
  }
  
  obs_r  <- tanh(obs_z)
  pred_r <- tanh(pred_z)
  
  delta_obs  <- colMeans(obs_r[idx1, , drop = FALSE]) - colMeans(obs_r[idx0, , drop = FALSE])
  delta_pred <- colMeans(pred_r[idx1, , drop = FALSE]) - colMeans(pred_r[idx0, , drop = FALSE])
  
  tibble(
    covariate = var,
    n_group0 = length(idx0),
    n_group1 = length(idx1),
    delta_corr = safe_cor(delta_obs, delta_pred),
    delta_rmse = safe_rmse(delta_obs, delta_pred)
  )
}

##############################################################
######################## Main loop ###########################
##############################################################

all_subject <- list()
all_group   <- list()

k1 <- 1L
k2 <- 1L

for (m in seq_len(nrow(model_specs))) {
  
  model_name  <- model_specs$model[m]
  model_dir   <- model_specs$dir_path[m]
  file_prefix <- model_specs$file_prefix[m]
  
  for (sd in seed_vec) {
    
    model_path <- file.path(
      model_dir,
      paste0(file_prefix, sd, ".pt")
    )
    
    cat("\n====================================\n")
    cat("Loading:", model_name, "Seed:", sd, "\n")
    cat("Path:", model_path, "\n")
    cat("====================================\n")
    
    fitted <- luz_load(model_path)
    
    # only needed for graph model
    if (model_name == "Graph_FUNGVA") {
      fitted$model$model$set_mask(masks)
    }
    
    for (sp in names(split_objects)) {
      
      cat("Evaluating:", sp, "\n")
      
      res <- collect_preds(fitted, split_objects[[sp]]$dl)
      
      ids <- rownames(split_objects[[sp]]$obs_z)
      rownames(res$pred_z) <- ids
      
      subj <- compute_subject_metrics(
        split_objects[[sp]]$obs_z,
        res$pred_z,
        ids
      ) %>%
        mutate(
          model = model_name,
          seed = sd,
          split = sp
        )
      
      grp <- bind_rows(
        compute_group_metrics(
          split_objects[[sp]]$obs_z,
          res$pred_z,
          split_objects[[sp]]$cov_raw,
          "SEX"
        ),
        compute_group_metrics(
          split_objects[[sp]]$obs_z,
          res$pred_z,
          split_objects[[sp]]$cov_raw,
          "APOE4"
        )
      ) %>%
        mutate(
          model = model_name,
          seed = sd,
          split = sp
        )
      
      all_subject[[k1]] <- subj
      all_group[[k2]]   <- grp
      
      k1 <- k1 + 1L
      k2 <- k2 + 1L
    }
  }
}

subject_df <- bind_rows(all_subject)
group_df   <- bind_rows(all_group)

##############################################################
######################## Summaries ###########################
##############################################################

subject_summary_by_seed <- subject_df %>%
  group_by(model, seed, split) %>%
  summarise(
    mean_subject_corr = mean(subject_corr, na.rm = TRUE),
    sd_subject_corr   = sd(subject_corr, na.rm = TRUE),
    mean_subject_rmse = mean(subject_rmse, na.rm = TRUE),
    sd_subject_rmse   = sd(subject_rmse, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(model, seed, split)

subject_summary_final <- subject_summary_by_seed %>%
  group_by(model, split) %>%
  summarise(
    mean_all_subject_corr = mean(mean_subject_corr, na.rm = TRUE),
    sd_subject_corr   = sd(mean_subject_corr, na.rm = TRUE),
    mean_all_subject_rmse = mean(mean_subject_rmse, na.rm = TRUE),
    sd_subject_rmse   = sd(mean_subject_rmse, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(split, model)

group_summary_by_seed <- group_df %>%
  arrange(model, seed, split, covariate)

group_summary_final <- group_summary_by_seed %>%
  group_by(model, split, covariate) %>%
  summarise(
    mean_delta_corr = mean(delta_corr, na.rm = TRUE),
    sd_delta_corr   = sd(delta_corr, na.rm = TRUE),
    mean_delta_rmse = mean(delta_rmse, na.rm = TRUE),
    sd_delta_rmse   = sd(delta_rmse, na.rm = TRUE),
    mean_n_group0   = mean(n_group0, na.rm = TRUE),
    mean_n_group1   = mean(n_group1, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(split, covariate, model)

##############################################################
######################## Save ###############################
##############################################################

write.csv(subject_df, file.path(output_dir, "subject_metrics.csv"), row.names = FALSE)
write.csv(group_df,   file.path(output_dir, "group_metrics.csv"), row.names = FALSE)

write.csv(subject_summary_by_seed, file.path(output_dir, "subject_summary_by_seed.csv"), row.names = FALSE)
write.csv(subject_summary_final,   file.path(output_dir, "subject_summary_final.csv"), row.names = FALSE)

write.csv(group_summary_by_seed, file.path(output_dir, "group_summary_by_seed.csv"), row.names = FALSE)
write.csv(group_summary_final,   file.path(output_dir, "group_summary_final.csv"), row.names = FALSE)

cat("\nDONE. Results saved.\n")

##############################################################
################ FINAL TABLES + MINIMAL FIGURES ##############
##############################################################


final_dir <- file.path(output_dir, "final_tables_figures")
dir.create(final_dir, recursive = TRUE, showWarnings = FALSE)

##############################################################
######################## Clean labels ########################
##############################################################

model_label_map <- c(
  "Graph_FUNGVA" = "Graph-FUNGVA",
  "MLP_FiLM1"    = "MLP-FiLM1",
  "MLP_FiLM2"    = "MLP-FiLM2"
)

split_label_map <- c(
  "train" = "Train",
  "val"   = "Validation",
  "test"  = "Test"
)

subject_df_plot <- subject_df %>%
  mutate(
    model = recode(model, !!!model_label_map),
    split = recode(split, !!!split_label_map)
  )

subject_summary_final_tbl <- subject_summary_final %>%
  mutate(
    model = recode(model, !!!model_label_map),
    split = recode(split, !!!split_label_map)
  )

group_summary_final_tbl <- group_summary_final %>%
  mutate(
    model = recode(model, !!!model_label_map),
    split = recode(split, !!!split_label_map),
    covariate = recode(covariate,
                       "APOE4" = "APOE4",
                       "SEX"   = "Sex")
  )

##############################################################
######################## Table 1 #############################
############ Reconstruction performance (Test only) ##########
##############################################################

table_recon_test <- subject_summary_final_tbl %>%
  filter(split == "Test") %>%
  transmute(
    Model = model,
    `Correlation (mean ± SD)` = sprintf("%.3f ± %.3f", mean_all_subject_corr, sd_subject_corr),
    `RMSE (mean ± SD)`        = sprintf("%.3f ± %.3f", mean_all_subject_rmse, sd_subject_rmse)
  )

write.csv(
  table_recon_test,
  file = file.path(final_dir, "Table_Reconstruction_Test.csv"),
  row.names = FALSE
)

print(table_recon_test)

##############################################################
######################## Table 2 #############################
######## Group-difference recovery performance (Test) ########
##############################################################

table_group_test <- group_summary_final_tbl %>%
  filter(split == "Test") %>%
  transmute(
    Covariate = covariate,
    Model = model,
    `Delta Correlation (mean ± SD)` = sprintf("%.3f ± %.3f", mean_delta_corr, sd_delta_corr),
    `Delta RMSE (mean ± SD)`        = sprintf("%.4f ± %.4f", mean_delta_rmse, sd_delta_rmse)
  ) %>%
  arrange(Covariate, Model)

write.csv(
  table_group_test,
  file = file.path(final_dir, "Table_GroupDiff_Test.csv"),
  row.names = FALSE
)

print(table_group_test)

##############################################################
######################## Figure 1 ############################
######## Subject-wise correlation distribution (Test) ########
##############################################################

p_recon_test <- ggplot(
  subject_df_plot %>% filter(split == "Test"),
  aes(x = model, y = subject_corr)
) +
  geom_boxplot(width = 0.6, outlier.shape = NA) +
  geom_jitter(width = 0.12, alpha = 0.45, size = 1.8) +
  labs(
    title = "Subject-wise Correlation on Test Set",
    x = NULL,
    y = "Observed vs Predicted Correlation"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.text.x = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

ggsave(
  filename = file.path(final_dir, "Figure_Reconstruction_Test.png"),
  plot = p_recon_test,
  width = 8,
  height = 5.5,
  dpi = 300
)

print(p_recon_test)

##############################################################
######################## Figure 2 ############################
############## Train / Val / Test generalization #############
##############################################################

p_generalization <- ggplot(
  subject_summary_final_tbl,
  aes(x = split, y = mean_all_subject_corr, group = model, shape = model)
) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 3) +
  labs(
    title = "Generalization Across Data Splits",
    x = NULL,
    y = "Mean Subject-wise Correlation",
    shape = "Model"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    panel.grid.minor = element_blank()
  )

ggsave(
  filename = file.path(final_dir, "Figure_Generalization_Gap.png"),
  plot = p_generalization,
  width = 8,
  height = 5.5,
  dpi = 300
)

print(p_generalization)


##############################################################
######################## Figure 3 ############################
######## Group-difference generalization across splits #######
##############################################################

group_summary_final_plot <- group_summary_final %>%
  mutate(
    model = recode(model, !!!model_label_map),
    split = factor(recode(split, !!!split_label_map),
                   levels = c("Train", "Validation", "Test")),
    covariate = recode(covariate,
                       "APOE4" = "APOE4",
                       "SEX"   = "Sex")
  )

p_group_generalization <- ggplot(
  group_summary_final_plot,
  aes(x = split, y = mean_delta_corr, group = model, shape = model)
) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 3) +
  facet_wrap(~ covariate, scales = "free_y") +
  labs(
    title = "Generalization of Group-Difference Recovery",
    x = NULL,
    y = "Mean Delta Correlation",
    shape = "Model"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    panel.grid.minor = element_blank()
  )

ggsave(
  filename = file.path(final_dir, "Figure_Group_Generalization.png"),
  plot = p_group_generalization,
  width = 9,
  height = 5.8,
  dpi = 300
)

print(p_group_generalization)

##############################################################
######################## Optional note #######################
##############################################################

cat("\nSaved final tables and figures to:\n", final_dir, "\n")












