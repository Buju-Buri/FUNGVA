##############################################################
##################### Data and Masking #######################
##############################################################

fc_train <- readRDS("fc_train.rds")
fc_val   <- readRDS("fc_val.rds")
fc_test  <- readRDS("fc_test.rds")
cov_df   <- readRDS("covariates.rds")

fc_train_array <- 1 - aperm(simplify2array(fc_train), c(3, 1, 2))
A_mat <- apply(fc_train_array, c(2, 3), mean)

masks <- list()
ranks <- t(apply(A_mat, 1, function(row) rank(row, ties.method = "first")))

n_size <- 2
masks[[1]] <- torch_tensor(matrix(as.numeric(ranks < (n_size^1 + 1)), nrow = 68), dtype = torch_float())
masks[[2]] <- torch_tensor(matrix(as.numeric(ranks < (n_size^2 + 1)), nrow = 68), dtype = torch_float())
masks[[3]] <- torch_tensor(matrix(as.numeric(ranks < (n_size^3 + 1)), nrow = 68), dtype = torch_float())
masks[[4]] <- torch_tensor(matrix(as.numeric(ranks < (n_size^4 + 1)), nrow = 68), dtype = torch_float())
masks[[5]] <- torch_tensor(matrix(as.numeric(ranks < (n_size^5 + 1)), nrow = 68), dtype = torch_float())
masks[[6]] <- torch_tensor(diag(2278), dtype = torch_float())

rm(fc_train_array, A_mat, ranks, n_size)

lt_rowwise <- function(M) {
  p <- nrow(M)
  out <- vector("numeric", (p * (p - 1)) / 2)
  k <- 1L
  for (i in 2:p) {
    len <- i - 1L
    out[k:(k + len - 1L)] <- M[i, 1:(i - 1L)]
    k <- k + len
  }
  out
}

fc_train_matrix <- do.call(rbind, lapply(fc_train, lt_rowwise))
fc_val_matrix   <- do.call(rbind, lapply(fc_val, lt_rowwise))
fc_test_matrix  <- do.call(rbind, lapply(fc_test, lt_rowwise))

eps <- 1e-6
fc_train_atanh <- atanh(pmin(pmax(fc_train_matrix, -1 + eps), 1 - eps))
fc_val_atanh   <- atanh(pmin(pmax(fc_val_matrix,   -1 + eps), 1 - eps))
fc_test_atanh  <- atanh(pmin(pmax(fc_test_matrix,  -1 + eps), 1 - eps))

train_ids <- rownames(fc_train_atanh)
val_ids   <- rownames(fc_val_atanh)
test_ids  <- rownames(fc_test_atanh)

num_vars   <- c("AGE", "EDUCATION", "pTau_ratio_log", "abeta_ratio_log")
other_vars <- c("SEX", "PARTNERED", "APOE4")

cov_train <- cov_df %>%
  filter(RID %in% train_ids) %>%
  slice(match(train_ids, RID))

cov_val <- cov_df %>%
  filter(RID %in% val_ids) %>%
  slice(match(val_ids, RID))

cov_test <- cov_df %>%
  filter(RID %in% test_ids) %>%
  slice(match(test_ids, RID))

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

storage.mode(fc_train_atanh) <- "double"
storage.mode(fc_val_atanh)   <- "double"
storage.mode(fc_test_atanh)  <- "double"
storage.mode(cov_train_mat)  <- "double"
storage.mode(cov_val_mat)    <- "double"
storage.mode(cov_test_mat)   <- "double"

##############################################################
######################## Dataset #############################
##############################################################

fungva_dataset <- dataset(
  name = "fungva_dataset",
  
  initialize = function(fc_mat, cov_mat) {
    stopifnot(nrow(fc_mat) == nrow(cov_mat))
    self$fc_mat <- fc_mat
    self$cov_mat <- cov_mat
  },
  
  .getitem = function(i) {
    list(
      x = list(
        fc  = torch_tensor(self$fc_mat[i, ], dtype = torch_float()),
        cov = torch_tensor(self$cov_mat[i, ], dtype = torch_float())
      ),
      y = torch_tensor(self$fc_mat[i, ], dtype = torch_float())
    )
  },
  
  .length = function() {
    nrow(self$fc_mat)
  }
)

train_ds <- fungva_dataset(fc_train_atanh, cov_train_mat)
val_ds   <- fungva_dataset(fc_val_atanh,   cov_val_mat)
test_ds  <- fungva_dataset(fc_test_atanh,  cov_test_mat)

train_dl <- dataloader(train_ds, batch_size = 16, shuffle = TRUE)
val_dl   <- dataloader(val_ds, batch_size = 15, shuffle = FALSE)
test_dl  <- dataloader(test_ds, batch_size = 15, shuffle = FALSE)



