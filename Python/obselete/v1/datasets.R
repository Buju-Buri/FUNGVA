##############################################################
##################### Data and Masking #######################
##############################################################

setwd("~/fungva/working-dir/")

library(dplyr)

fc_train = readRDS("fc_train.rds")
fc_val   = readRDS("fc_val.rds")
fc_test  = readRDS("fc_test.rds")
cov_df   = readRDS("covariates.rds")

fc_train_array = 1 - aperm(simplify2array(fc_train), c(3, 1, 2))
A_mat = apply(fc_train_array, c(2, 3), mean)

masks = list()
ranks = t(apply(A_mat, 1, function(row) rank(row, ties.method = "first")))

n_size = 2

masks[[1]] = matrix(as.numeric(ranks < (n_size^1 + 1)), nrow = 68)
masks[[2]] = matrix(as.numeric(ranks < (n_size^2 + 1)), nrow = 68)
masks[[3]] = matrix(as.numeric(ranks < (n_size^3 + 1)), nrow = 68)
masks[[4]] = matrix(as.numeric(ranks < (n_size^4 + 1)), nrow = 68)
masks[[5]] = matrix(as.numeric(ranks < (n_size^5 + 1)), nrow = 68)
masks[[6]] = diag(2278)

storage.mode(masks[[1]]) = "double"
storage.mode(masks[[2]]) = "double"
storage.mode(masks[[3]]) = "double"
storage.mode(masks[[4]]) = "double"
storage.mode(masks[[5]]) = "double"
storage.mode(masks[[6]]) = "double"

rm(fc_train_array, A_mat, ranks, n_size)

lt_rowwise = function(M) {
  p = nrow(M)
  out = vector("numeric", (p * (p - 1)) / 2)
  k = 1L
  
  for (i in 2:p) {
    len = i - 1L
    out[k:(k + len - 1L)] = M[i, 1:(i - 1L)]
    k = k + len
  }
  
  out
}

fc_train_matrix = do.call(rbind, lapply(fc_train, lt_rowwise))
fc_val_matrix   = do.call(rbind, lapply(fc_val, lt_rowwise))
fc_test_matrix  = do.call(rbind, lapply(fc_test, lt_rowwise))

eps = 1e-6

fc_train_atanh = atanh(pmin(pmax(fc_train_matrix, -1 + eps), 1 - eps))
fc_val_atanh   = atanh(pmin(pmax(fc_val_matrix,   -1 + eps), 1 - eps))
fc_test_atanh  = atanh(pmin(pmax(fc_test_matrix,  -1 + eps), 1 - eps))

train_ids = rownames(fc_train_atanh)
val_ids   = rownames(fc_val_atanh)
test_ids  = rownames(fc_test_atanh)

num_vars   = c("AGE", "EDUCATION", "pTau_ratio_log", "abeta_ratio_log")
other_vars = c("SEX", "PARTNERED", "APOE4")

cov_train = cov_df %>%
  filter(RID %in% train_ids) %>%
  slice(match(train_ids, RID))

cov_val = cov_df %>%
  filter(RID %in% val_ids) %>%
  slice(match(val_ids, RID))

cov_test = cov_df %>%
  filter(RID %in% test_ids) %>%
  slice(match(test_ids, RID))

train_means = sapply(cov_train[num_vars], mean, na.rm = TRUE)
train_sds   = sapply(cov_train[num_vars], sd, na.rm = TRUE)

scale_with_train = function(df, vars, means, sds) {
  x = as.matrix(df[, vars, drop = FALSE])
  x = sweep(x, 2, means, "-")
  x = sweep(x, 2, sds, "/")
  x
}

cov_num_train = scale_with_train(cov_train, num_vars, train_means, train_sds)
cov_num_val   = scale_with_train(cov_val,   num_vars, train_means, train_sds)
cov_num_test  = scale_with_train(cov_test,  num_vars, train_means, train_sds)

cov_other_train = as.matrix(cov_train[, other_vars, drop = FALSE])
cov_other_val   = as.matrix(cov_val[,   other_vars, drop = FALSE])
cov_other_test  = as.matrix(cov_test[,  other_vars, drop = FALSE])

cov_train_mat = cbind(cov_num_train, cov_other_train)
cov_val_mat   = cbind(cov_num_val,   cov_other_val)
cov_test_mat  = cbind(cov_num_test,  cov_other_test)

cov_dim = ncol(cov_train_mat)

storage.mode(fc_train_atanh) = "double"
storage.mode(fc_val_atanh)   = "double"
storage.mode(fc_test_atanh)  = "double"
storage.mode(cov_train_mat)  = "double"
storage.mode(cov_val_mat)    = "double"
storage.mode(cov_test_mat)   = "double"

##############################################################
######################## Dataset #############################
##############################################################

fungva_dataset = function(fc_mat, cov_mat) {
  stopifnot(nrow(fc_mat) == nrow(cov_mat))
  
  dataset_obj = list(
    fc_mat = fc_mat,
    cov_mat = cov_mat,
    n = nrow(fc_mat)
  )
  
  class(dataset_obj) = "fungva_dataset"
  
  dataset_obj
}

fungva_getitem = function(ds, i) {
  list(
    x = list(
      fc  = as.numeric(ds$fc_mat[i, ]),
      cov = as.numeric(ds$cov_mat[i, ])
    ),
    y = as.numeric(ds$fc_mat[i, ])
  )
}

fungva_length = function(ds) {
  ds$n
}

fungva_dataloader = function(ds, batch_size, shuffle = FALSE) {
  n = fungva_length(ds)
  indices = seq_len(n)
  
  if (shuffle) {
    indices = sample(indices)
  }
  
  batches = split(indices, ceiling(seq_along(indices) / batch_size))
  
  dl = lapply(batches, function(batch_idx) {
    list(
      x = list(
        fc  = ds$fc_mat[batch_idx, , drop = FALSE],
        cov = ds$cov_mat[batch_idx, , drop = FALSE]
      ),
      y = ds$fc_mat[batch_idx, , drop = FALSE],
      indices = batch_idx
    )
  })
  
  class(dl) = "fungva_dataloader"
  
  dl
}

train_ds = fungva_dataset(fc_train_atanh, cov_train_mat)
val_ds   = fungva_dataset(fc_val_atanh,   cov_val_mat)
test_ds  = fungva_dataset(fc_test_atanh,  cov_test_mat)

set.seed(1)

train_dl = fungva_dataloader(train_ds, batch_size = 16, shuffle = TRUE)
val_dl   = fungva_dataloader(val_ds,   batch_size = 15, shuffle = FALSE)
test_dl  = fungva_dataloader(test_ds,  batch_size = 15, shuffle = FALSE)