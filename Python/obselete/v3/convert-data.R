# install.packages("hdf5r")  # run once if needed
# install.packages("arrow")  # run once if needed

library(hdf5r)
library(arrow)

fc_to_array = function(fc_list) {
  x = array(
    unlist(lapply(fc_list, as.matrix)),
    dim = c(68, 68, length(fc_list))
  )
  
  # Python-friendly shape: subjects x nodes x nodes
  aperm(x, c(3, 1, 2))
}

dir.create("python_data", showWarnings = FALSE)

h5 = H5File$new("python_data/fc_data.h5", mode = "w")

h5[["X_train"]] = fc_to_array(fc_train)
h5[["X_val"]] = fc_to_array(fc_val)
h5[["X_test"]] = fc_to_array(fc_test)

h5[["train_ids"]] = names(fc_train)
h5[["val_ids"]] = names(fc_val)
h5[["test_ids"]] = names(fc_test)

h5$close_all()

write_parquet(covariates, "python_data/covariates.parquet")