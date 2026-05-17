##############################################################
##################### Data and Masking ######################
##############################################################

# Read preprocessed RDS files
fc_train = readRDS("~/fungva/data/fc_train.rds")
fc_val   = readRDS("~/fungva/data/fc_val.rds")
fc_test  = readRDS("~/fungva/data/fc_test.rds")
cov_df   = readRDS("~/fungva/data/covariates.rds")

# Convert FC matrices to lower-triangle vectors
lt_rowwise <- function(M) {
  p = nrow(M)
  out = numeric((p * (p - 1)) / 2)
  k = 1L
  for (i in 2:p) {
    len = i - 1L
    out[k:(k+len-1)] = M[i, 1:(i-1)]
    k = k + len
  }
  out
}

fc_train_matrix <- do.call(rbind, lapply(fc_train, lt_rowwise))
fc_val_matrix   <- do.call(rbind, lapply(fc_val, lt_rowwise))
fc_test_matrix  <- do.call(rbind, lapply(fc_test, lt_rowwise))

eps = 1e-6
fc_train_atanh <- atanh(pmin(pmax(fc_train_matrix, -1 + eps), 1 - eps))
fc_val_atanh   <- atanh(pmin(pmax(fc_val_matrix, -1 + eps), 1 - eps))
fc_test_atanh  <- atanh(pmin(pmax(fc_test_matrix, -1 + eps), 1 - eps))

# Covariates
num_vars   <- c("AGE", "EDUCATION", "pTau_ratio_log", "abeta_ratio_log")
other_vars <- c("SEX", "PARTNERED", "APOE4")

train_ids <- rownames(fc_train_atanh)
val_ids   <- rownames(fc_val_atanh)
test_ids  <- rownames(fc_test_atanh)

cov_train <- cov_df %>% filter(RID %in% train_ids) %>% slice(match(train_ids, RID))
cov_val   <- cov_df %>% filter(RID %in% val_ids)   %>% slice(match(val_ids, RID))
cov_test  <- cov_df %>% filter(RID %in% test_ids)  %>% slice(match(test_ids, RID))

train_means <- sapply(cov_train[num_vars], mean, na.rm = TRUE)
train_sds   <- sapply(cov_train[num_vars], sd, na.rm = TRUE)

scale_with_train <- function(df, vars, means, sds) {
  x <- as.matrix(df[, vars, drop = FALSE])
  x <- sweep(x, 2, means, "-")
  x <- sweep(x, 2, sds, "/")
  x
}

cov_num_train <- scale_with_train(cov_train, num_vars, train_means, train_sds)
cov_num_val   <- scale_with_train(cov_val,   num_vars, train_means, train_sds)
cov_num_test  <- scale_with_train(cov_test,  num_vars, train_means, train_sds)

cov_other_train <- as.matrix(cov_train[, other_vars, drop = FALSE])
cov_other_val   <- as.matrix(cov_val[,   other_vars, drop = FALSE])
cov_other_test  <- as.matrix(cov_test[,  other_vars, drop = FALSE])

cov_train_mat <- cbind(cov_num_train, cov_other_train)
cov_val_mat   <- cbind(cov_num_val,   cov_other_val)
cov_test_mat  <- cbind(cov_num_test,  cov_other_test)

cov_dim <- ncol(cov_train_mat)

##############################################################
######################## Dataset #############################
##############################################################

# Simple dataset object for Keras
fungva_dataset <- function(fc_mat, cov_mat) {
  stopifnot(nrow(fc_mat) == nrow(cov_mat))
  list(
    fc = fc_mat,
    cov = cov_mat,
    n = nrow(fc_mat)
  )
}

# Batch generator for keras::fit()
fungva_generator <- function(dataset, batch_size = 32, shuffle = TRUE) {
  function() {
    n <- dataset$n
    idx <- 1:n
    if (shuffle) idx <- sample(idx)
    batches <- split(idx, ceiling(seq_along(idx)/batch_size))
    
    lapply(batches, function(batch_idx) {
      list(
        x = list(
          fc  = dataset$fc[batch_idx, , drop = FALSE],
          cov = dataset$cov[batch_idx, , drop = FALSE]
        ),
        y = dataset$fc[batch_idx, , drop = FALSE]
      )
    })
  }
}

# Create datasets and generators
train_ds <- fungva_dataset(fc_train_atanh, cov_train_mat)
val_ds   <- fungva_dataset(fc_val_atanh,   cov_val_mat)
test_ds  <- fungva_dataset(fc_test_atanh,  cov_test_mat)

batch_size <- 16
train_gen <- fungva_generator(train_ds, batch_size = batch_size, shuffle = TRUE)
val_gen   <- fungva_generator(val_ds,   batch_size = 16, shuffle = FALSE)
test_gen  <- fungva_generator(test_ds,  batch_size = 16, shuffle = FALSE)