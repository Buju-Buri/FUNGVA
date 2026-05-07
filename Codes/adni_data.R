library(torch)
library(luz)
library(dplyr)
library(ggplot2)
library(tidyr)
library(BlandAltmanLeh)


id_all <- readRDS("ADNI\\ADNI_functional_with_PET\\IDs_6grps.rds")
sub_dem <- read.csv("ADNI\\tau_PET_longitudinal\\subject_demographics.csv")
fmri <- readRDS("ADNI\\ADNI_functional_with_PET\\X_6grps.rds")

########Retrieve IDs for all FMRIs SEX-wise#############

id_all_df <- data.frame(RID=as.integer(unlist(id_all)), Sex=ifelse(grepl("female",names(unlist(id_all))),"F","M"))

#########Select Covariates and choose IDs with no missing values###########

sub_dem <- sub_dem[sub_dem$RID %in% as.integer(names(fmri)),c(1,6,7,9,10,11,14,15)]
sub_dem <- sub_dem[complete.cases(sub_dem),]

sub_dem$PTMARRY <- ifelse(sub_dem$PTMARRY=="Married",1,0)
sub_dem$PTGENDER <- ifelse(sub_dem$PTGENDER=="Female",0,1)
sub_dem$PTEDUCAT <- as.numeric(sub_dem$PTEDUCAT)
sub_dem$APOE4 <- ifelse(sub_dem$APOE4==0,0,1)
sub_dem <- sub_dem[,c(1,3:8)]

id_all_df <- id_all_df[id_all_df$RID %in% sub_dem$RID,]
all_ids <- id_all_df$RID

###############blood_biomarker_FIN################

fnih <- read.csv("ADNI\\biospecimen\\blood\\blood_biomarker_FIN\\FNIHBC_BLOOD_BIOMARKER_TRAJECTORIES_23Jun2025.csv")
fnih <- fnih[,c(4,5,13,14)]

fnih_ptau217 <- fnih[fnih$PLASMA_BIOMARKER=="Fuji_plasma_ptau217",c(1,2,4)]
colnames(fnih_ptau217)[3] <- "pTau217"
fnih_ptau217 <- fnih_ptau217[complete.cases(fnih_ptau217),]
sum(all_ids %in% unique(fnih_ptau217$RID))

fnih_abeta42 <- fnih[fnih$PLASMA_BIOMARKER=="Fuji_plasma_Ab42",c(1,2,4)]
colnames(fnih_abeta42)[3] <- "abeta42"
fnih_abeta42 <- fnih_abeta42[complete.cases(fnih_abeta42),]
sum(all_ids %in% unique(fnih_abeta42$RID))

fnih_abeta40 <- fnih[fnih$PLASMA_BIOMARKER=="Fuji_plasma_Ab40",c(1,2,4)]
colnames(fnih_abeta40)[3] <- "abeta40"
fnih_abeta40 <- fnih_abeta40[complete.cases(fnih_abeta40),]
sum(all_ids %in% unique(fnih_abeta40$RID))

fnih_fuji <- inner_join(fnih_ptau217, fnih_abeta42, 
                        by = c("RID", "EXAMDATE"))

fnih_fuji <- inner_join(fnih_fuji, fnih_abeta40, 
                        by = c("RID", "EXAMDATE"))

fnih_fuji <- fnih_fuji %>%
  mutate(across(3:5, as.numeric))

fnih_fuji$ab42_ab40 <- fnih_fuji$abeta42/fnih_fuji$abeta40
fnih_fuji$ptau_ab42 <- fnih_fuji$pTau217/fnih_fuji$abeta42


sum(all_ids %in% unique(fnih_fuji$RID))

###############penn_plasma_biomarker################

upenn <- read.csv("ADNI\\biospecimen\\blood\\penn_plasma_biomarker\\UPENN_PLASMA_FUJIREBIO_QUANTERIX_01Apr2026.csv")
upenn <- upenn[,c(3,6,9:13)]
upenn <- upenn[
  upenn$pT217_F != -4 &
    upenn$AB42_F != -4 &
    upenn$AB40_F != -4 &
    upenn$AB42_AB40_F != -4 &
    upenn$pT217_AB42_F != -4,
]
upenn <- upenn[complete.cases(upenn),]
sum(all_ids %in% unique(upenn$RID))

sum(all_ids %in% unique(c(upenn$RID,fnih_fuji$RID)))

common_visits <- inner_join(upenn, fnih_fuji, 
                            by = c("RID", "EXAMDATE"))


###########Exploratory Checks###########

hist(upenn$AB42_AB40_F)
hist(fnih_fuji$ab42_ab40)
hist(upenn$pT217_AB42_F)
hist(fnih_fuji$ptau_ab42)
plot(common_visits$AB42_AB40_F,common_visits$ab42_ab40)
plot(log10(common_visits$AB42_AB40_F), log10(common_visits$ab42_ab40))
plot(common_visits$pT217_AB42_F, common_visits$ptau_ab42)
plot(log10(common_visits$pT217_AB42_F), log10(common_visits$ptau_ab42))
library(BlandAltmanLeh)

# ---------- reusable Bland-Altman function on log10 scale ----------#########
run_ba_log10 <- function(upenn, fnih, ratio_name = "Ratio") {
  
  # keep only valid paired observations
  keep <- is.finite(upenn) & is.finite(fnih) & upenn > 0 & fnih > 0
  
  upenn_log <- log10(upenn[keep])
  fnih_log  <- log10(fnih[keep])
  
  if (length(upenn_log) < 3) {
    stop("Not enough valid paired observations for Bland-Altman analysis.")
  }
  
  # Bland-Altman stats
  ba_stats <- bland.altman.stats(upenn_log, fnih_log)
  
  # proportional bias check
  ba_df <- data.frame(
    mean_log = ba_stats$means,
    diff_log = ba_stats$diffs
  )
  
  prop_bias_fit <- lm(diff_log ~ mean_log, data = ba_df)
  
  # plotting range with a little padding
  y_all <- c(ba_stats$diffs, ba_stats$mean.diffs,
             ba_stats$upper.limit, ba_stats$lower.limit)
  y_pad <- 0.08 * diff(range(y_all))
  
  plot(
    ba_stats$means, ba_stats$diffs,
    main = paste("Bland-Altman Plot:", ratio_name, "(log10 scale)"),
    xlab = "Mean of log10(UPenn, FNIH)",
    ylab = "Difference in log10 ratio (UPenn - FNIH)",
    pch = 16
  )
  
  abline(h = ba_stats$mean.diffs, lwd = 2)
  abline(h = ba_stats$upper.limit, lty = 2)
  abline(h = ba_stats$lower.limit, lty = 2)
  
  # optional regression line for proportional bias
  abline(prop_bias_fit, lty = 3)
  
  # annotate lines
  usr <- par("usr")
  x_text <- usr[2]
  
  text(x_text, ba_stats$mean.diffs,
       labels = paste0(" Bias = ", round(ba_stats$mean.diffs, 4)),
       pos = 3, cex = 0.9)
  
  text(x_text, ba_stats$upper.limit,
       labels = paste0(" Upper LoA = ", round(ba_stats$upper.limit, 4)),
       pos = 3, cex = 0.9)
  
  text(x_text, ba_stats$lower.limit,
       labels = paste0(" Lower LoA = ", round(ba_stats$lower.limit, 4)),
       pos = 1, cex = 0.9)
  
  cat("\n-----------------------------\n")
  cat("Bland-Altman analysis:", ratio_name, "\n")
  cat("Valid paired observations:", length(upenn_log), "\n")
  cat("Mean bias (UPenn - FNIH) on log10 scale:", round(ba_stats$mean.diffs, 6), "\n")
  cat("Upper LoA:", round(ba_stats$upper.limit, 6), "\n")
  cat("Lower LoA:", round(ba_stats$lower.limit, 6), "\n")
  cat("\nProportional bias test:\n")
  print(summary(prop_bias_fit))
  
  invisible(list(
    ba_stats = ba_stats,
    ba_df = ba_df,
    prop_bias_fit = prop_bias_fit
  ))
}

# pTau217 / Aβ42
ba_ptau_ab42 <- run_ba_log10(
  upenn = common_visits$pT217_AB42_F,
  fnih  = common_visits$ptau_ab42,
  ratio_name = "pTau217 / Aβ42"
)

# Aβ42 / Aβ40
ba_ab42_ab40 <- run_ba_log10(
  upenn = common_visits$AB42_AB40_F,
  fnih  = common_visits$ab42_ab40,
  ratio_name = "Aβ42 / Aβ40"
)

set.seed(123)

############################################################
# 1. Make sure ratio columns are numeric and positive
############################################################

common_visits <- common_visits %>%
  mutate(
    pT217_AB42_F = as.numeric(pT217_AB42_F),
    ptau_ab42    = as.numeric(ptau_ab42),
    AB42_AB40_F  = as.numeric(AB42_AB40_F),
    ab42_ab40    = as.numeric(ab42_ab40)
  )

fnih_fuji <- fnih_fuji %>%
  mutate(
    ptau_ab42   = as.numeric(ptau_ab42),
    ab42_ab40   = as.numeric(ab42_ab40)
  )

upenn <- upenn %>%
  mutate(
    pT217_AB42_F = as.numeric(pT217_AB42_F),
    AB42_AB40_F  = as.numeric(AB42_AB40_F)
  )

############################################################
# 2. Fit calibration models on log10 scale
#    UPenn is the reference, so outcome = UPenn
############################################################

set.seed(123)

# pTau217 / Aβ42
fit_ptau <- lm(
  log10(pT217_AB42_F) ~ log10(ptau_ab42),
  data = common_visits
)

summary(fit_ptau)

set.seed(123)

# Aβ42 / Aβ40
fit_ab42 <- lm(
  log10(AB42_AB40_F) ~ log10(ab42_ab40),
  data = common_visits
)

summary(fit_ab42)

############################################################
# 3. Apply calibration to ALL FNIH rows
#    This converts FNIH ratios onto the UPenn scale
############################################################

fnih_fuji <- fnih_fuji %>%
  mutate(
    ptau_ab42_upennscale = 10^(predict(
      fit_ptau,
      newdata = data.frame(ptau_ab42 = ptau_ab42)
    )),
    ab42_ab40_upennscale = 10^(predict(
      fit_ab42,
      newdata = data.frame(ab42_ab40 = ab42_ab40)
    ))
  )

############################################################
# 4. Identify overlapping visits
#    UPenn takes precedence, so overlapping FNIH rows are removed
############################################################

overlap_keys <- common_visits %>%
  distinct(RID, EXAMDATE)

fnih_nonoverlap <- anti_join(
  fnih_fuji,
  overlap_keys,
  by = c("RID", "EXAMDATE")
)

############################################################
# 5. Create harmonized versions with common column names
############################################################

# Keep UPenn values unchanged
upenn_harmonized <- upenn %>%
  transmute(
    RID,
    EXAMDATE,
    ptau217_abeta42_ratio = pT217_AB42_F,
    abeta42_abeta40_ratio = AB42_AB40_F,
    source = "UPenn"
  )

# Use calibrated FNIH values
fnih_harmonized <- fnih_nonoverlap %>%
  transmute(
    RID,
    EXAMDATE,
    ptau217_abeta42_ratio = ptau_ab42_upennscale,
    abeta42_abeta40_ratio = ab42_ab40_upennscale,
    source = "FNIH_calibrated"
  )

############################################################
# 6. Merge final dataset
############################################################

merged_ratios <- bind_rows(
  upenn_harmonized,
  fnih_harmonized
) %>%
  arrange(RID, EXAMDATE)

############################################################
# 7. Quick checks
############################################################

# counts
nrow(upenn_harmonized)
nrow(fnih_harmonized)
nrow(merged_ratios)

# no duplicate visits after merge
merged_ratios %>%
  count(RID, EXAMDATE) %>%
  filter(n > 1)

# summary of final ratios
summary(merged_ratios$ptau217_abeta42_ratio)
summary(merged_ratios$abeta42_abeta40_ratio)

############################################################
# 8. Optional: compare overlap before/after calibration
#    This is useful for checking how well calibration worked
############################################################

common_visits_check <- common_visits %>%
  mutate(
    ptau_pred_upennscale = 10^(predict(
      fit_ptau,
      newdata = data.frame(ptau_ab42 = ptau_ab42)
    )),
    ab42_pred_upennscale = 10^(predict(
      fit_ab42,
      newdata = data.frame(ab42_ab40 = ab42_ab40)
    ))
  )

# Raw vs calibrated scatterplots
par(mfrow = c(2, 2))

plot(
  common_visits$ptau_ab42, common_visits$pT217_AB42_F,
  main = "Raw: pTau217/Aβ42",
  xlab = "FNIH",
  ylab = "UPenn",
  pch = 16
)
abline(0, 1, lty = 2)

plot(
  common_visits_check$ptau_pred_upennscale, common_visits_check$pT217_AB42_F,
  main = "Calibrated: pTau217/Aβ42",
  xlab = "FNIH calibrated to UPenn scale",
  ylab = "UPenn",
  pch = 16
)
abline(0, 1, lty = 2)

plot(
  common_visits$ab42_ab40, common_visits$AB42_AB40_F,
  main = "Raw: Aβ42/Aβ40",
  xlab = "FNIH",
  ylab = "UPenn",
  pch = 16
)
abline(0, 1, lty = 2)

plot(
  common_visits_check$ab42_pred_upennscale, common_visits_check$AB42_AB40_F,
  main = "Calibrated: Aβ42/Aβ40",
  xlab = "FNIH calibrated to UPenn scale",
  ylab = "UPenn",
  pch = 16
)
abline(0, 1, lty = 2)

par(mfrow = c(1, 1))


common_visits_post <- common_visits %>%
  mutate(
    ptau_pred = 10^(predict(fit_ptau,
                            newdata = data.frame(ptau_ab42 = ptau_ab42)
    )),
    ab42_pred = 10^(predict(fit_ab42,
                            newdata = data.frame(ab42_ab40 = ab42_ab40)
    ))
  )

# run Bland–Altman again
run_ba_log10(
  upenn = common_visits_post$pT217_AB42_F,
  fnih  = common_visits_post$ptau_pred,
  ratio_name = "pTau217/Aβ42 (post-calibration)"
)

run_ba_log10(
  upenn = common_visits_post$AB42_AB40_F,
  fnih  = common_visits_post$ab42_pred,
  ratio_name = "Aβ42/Aβ40 (post-calibration)"
)

boxplot(
  ptau217_abeta42_ratio ~ source,
  data = merged_ratios,
  main = "Post-merge distribution: pTau217/Aβ42"
)

boxplot(
  abeta42_abeta40_ratio ~ source,
  data = merged_ratios,
  main = "Post-merge distribution: Aβ42/Aβ40"
)

sum(all_ids %in% unique(merged_ratios$RID))

biomarkers <- merged_ratios[merged_ratios$RID %in% all_ids,]

############################################################
# 1. Convert EXAMDATE to Date format
############################################################

biomarkers$EXAMDATE <- as.Date(biomarkers$EXAMDATE)
sub_dem$EXAMDATE   <- as.Date(sub_dem$EXAMDATE)

############################################################
# 2. Ensure one demographic row per RID
# (choose earliest visit; change to slice_tail(1) if needed)
############################################################

sub_dem_unique <- sub_dem %>%
  arrange(RID, EXAMDATE) %>%
  group_by(RID) %>%
  slice(1) %>%
  ungroup()

############################################################
# 3. Keep only subjects present in biomarkers
############################################################

sub_dem_unique <- sub_dem_unique %>%
  filter(RID %in% biomarkers$RID)

############################################################
# 4. Join biomarker and demographic data by RID
############################################################

merged_all <- biomarkers %>%
  inner_join(sub_dem_unique, by = "RID", suffix = c("_bio", "_dem"))

############################################################
# 5. Compute absolute time difference
############################################################

merged_all <- merged_all %>%
  mutate(
    date_diff = abs(as.numeric(EXAMDATE_bio - EXAMDATE_dem))
  )

############################################################
# 6. Select closest biomarker visit per RID
############################################################

cov_df <- merged_all %>%
  filter(date_diff <= 365*6) %>%
  group_by(RID) %>%
  slice_min(order_by = date_diff, n = 1, with_ties = FALSE) %>%
  ungroup()

############################################################
# 7. Final clean dataset
############################################################

cov_df <- cov_df %>%
  transmute(
    RID,
    EXAMDATE = EXAMDATE_dem,
    AGE,
    SEX = PTGENDER,
    PARTNERED = PTMARRY,
    EDUCATION = PTEDUCAT,
    APOE4,
    pTau_ratio_log = log10(ptau217_abeta42_ratio),
    abeta_ratio_log = log10(abeta42_abeta40_ratio)
  )

############################################################
# 8. Sanity checks
############################################################

# number of subjects
cat("Number of subjects:", length(unique(cov_df$RID)), "\n")

# check missing values
print(colSums(is.na(cov_df)))

# check time differences
cat("\nSummary of time differences (days):\n")
print(summary(merged_all$date_diff))

########Final dataset###############

fungva_fmri <- fmri[as.character(cov_df$RID)]
fungva_fmri <- lapply(fungva_fmri, function(M) {
  diag(M) <- 1
  M
})

saveRDS(fungva_fmri, file = "fungva_fmri.rds" )

cov_df <- cov_df %>%
  mutate(RID = as.character(RID))

saveRDS(cov_df, file = "covariates.rds" )



############################################################
# 1. Set seeds (for reproducibility)
############################################################

set.seed(123)
torch_manual_seed(123)

############################################################
# 2. Create stratification variable (sex × APOE4)
############################################################

cov_df <- cov_df %>%
  mutate(
    strat_group = paste0(SEX, "_", APOE4)  # e.g., "0_1"
  )

############################################################
# 3. Helper function:
# allocate exact total counts across strata
############################################################

allocate_stratified_counts <- function(n_vec, total_target) {
  raw_alloc  <- n_vec / sum(n_vec) * total_target
  base_alloc <- floor(raw_alloc)
  remainder  <- total_target - sum(base_alloc)
  
  if (remainder > 0) {
    add_idx <- order(raw_alloc - base_alloc, decreasing = TRUE)[1:remainder]
    base_alloc[add_idx] <- base_alloc[add_idx] + 1
  }
  
  base_alloc
}

############################################################
# 4. Compute exact train/val counts per stratum
############################################################

strata_counts <- cov_df %>%
  count(strat_group, name = "n") %>%
  arrange(strat_group)

strata_counts <- strata_counts %>%
  mutate(
    n_train = allocate_stratified_counts(n, 240),
    n_val   = allocate_stratified_counts(n, 60)
  ) %>%
  mutate(
    n_test = n - n_train - n_val
  )

print(strata_counts)

############################################################
# 5. Perform exact stratified split
############################################################

split_df <- lapply(seq_len(nrow(strata_counts)), function(i) {
  
  g <- strata_counts$strat_group[i]
  n_train <- strata_counts$n_train[i]
  n_val   <- strata_counts$n_val[i]
  n_test  <- strata_counts$n_test[i]
  
  dat_g <- cov_df %>%
    filter(strat_group == g)
  
  idx <- sample(nrow(dat_g))
  
  train_idx <- idx[1:n_train]
  val_idx   <- idx[(n_train + 1):(n_train + n_val)]
  test_idx  <- idx[(n_train + n_val + 1):(n_train + n_val + n_test)]
  
  dat_g %>%
    mutate(
      split = case_when(
        row_number() %in% train_idx ~ "train",
        row_number() %in% val_idx   ~ "val",
        row_number() %in% test_idx  ~ "test"
      )
    )
}) %>%
  bind_rows()

############################################################
# 6. Extract datasets
############################################################

train_df <- split_df %>% filter(split == "train")
val_df   <- split_df %>% filter(split == "val")
test_df  <- split_df %>% filter(split == "test")

############################################################
# 7. Check sizes
############################################################

cat("Train:", nrow(train_df), "\n")
cat("Val:", nrow(val_df), "\n")
cat("Test:", nrow(test_df), "\n")

############################################################
# 8. Check stratification balance
############################################################

cat("\nSex distribution:\n")
print(prop.table(table(split_df$split, split_df$SEX), 1))

cat("\nAPOE4 distribution:\n")
print(prop.table(table(split_df$split, split_df$APOE4), 1))

cat("\nStrat group distribution:\n")
print(prop.table(table(split_df$split, split_df$strat_group), 1))


summary(train_df$pTau_ratio_log)
summary(val_df$pTau_ratio_log)
summary(test_df$pTau_ratio_log)

summary(train_df$abeta_ratio_log)
summary(val_df$abeta_ratio_log)
summary(test_df$abeta_ratio_log)

############################################################
# 9. Final Datasets
############################################################

cov_train <- train_df[,1:9]
fc_train <- fungva_fmri[cov_train$RID]
cov_val <- val_df[,1:9]
fc_val <- fungva_fmri[cov_val$RID]
cov_test <- test_df[,1:9]
fc_test <- fungva_fmri[cov_test$RID]

saveRDS(fc_train, file = "fc_train.rds" )
saveRDS(fc_val, file = "fc_val.rds" )
saveRDS(fc_test, file = "fc_test.rds" )

############################################################
# 10. Checking for outliers in covariates
############################################################


detect_outliers <- function(x) {
  x <- x[is.finite(x)]
  
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  
  lower <- q1 - 1.5 * iqr
  upper <- q3 + 1.5 * iqr
  
  which(x < lower | x > upper)
}

num_vars <- c("AGE", "EDUCATION", "pTau_ratio_log", "abeta_ratio_log")

####Training data#######

outlier_indices <- lapply(num_vars, function(v) {
  idx <- detect_outliers(cov_train[[v]])
  
  if (length(idx) == 0) {
    return(NULL)
  }
  
  data.frame(
    variable = v,
    index = idx,
    value = cov_train[[v]][idx]
  )
})

outlier_df <- do.call(rbind, outlier_indices)

outlier_df

cov_long <- cov_train %>%
  select(all_of(num_vars)) %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value")

ggplot(cov_long, aes(x = variable, y = value)) +
  geom_boxplot(outlier.colour = "red", outlier.size = 2) +
  theme_minimal(base_size = 14) +
  labs(title = "Covariate Outlier Check", x = "", y = "Value")

########Validation Data###########

outlier_indices <- lapply(num_vars, function(v) {
  idx <- detect_outliers(cov_val[[v]])
  
  if (length(idx) == 0) {
    return(NULL)
  }
  
  data.frame(
    variable = v,
    index = idx,
    value = cov_val[[v]][idx]
  )
})

outlier_df <- do.call(rbind, outlier_indices)

outlier_df

cov_long <- cov_val %>%
  select(all_of(num_vars)) %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value")

ggplot(cov_long, aes(x = variable, y = value)) +
  geom_boxplot(outlier.colour = "red", outlier.size = 2) +
  theme_minimal(base_size = 14) +
  labs(title = "Covariate Outlier Check", x = "", y = "Value")

######Test Data#####

outlier_indices <- lapply(num_vars, function(v) {
  idx <- detect_outliers(cov_test[[v]])
  
  if (length(idx) == 0) {
    return(NULL)
  }
  
  data.frame(
    variable = v,
    index = idx,
    value = cov_test[[v]][idx]
  )
})

outlier_df <- do.call(rbind, outlier_indices)

outlier_df

cov_long <- cov_test %>%
  select(all_of(num_vars)) %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value")

ggplot(cov_long, aes(x = variable, y = value)) +
  geom_boxplot(outlier.colour = "red", outlier.size = 2) +
  theme_minimal(base_size = 14) +
  labs(title = "Covariate Outlier Check", x = "", y = "Value")
