library(xgboost)

xgboost_cross_val <- function(X_train, y_train, 
                                     max_depth_range = NULL, 
                                     learning_rate_range = NULL, 
                                     n_estimators_range = NULL,  
                                     gamma_range = NULL,
                                     reg_lambda_range = NULL,    
                                     cv = 5, 
                                     random_state = 123) {
  
  if (is.null(max_depth_range))      max_depth_range <- c(3, 4, 5)
  if (is.null(learning_rate_range))  learning_rate_range <- c(0.01, 0.05, 0.1)
  if (is.null(n_estimators_range))   n_estimators_range <- c(300, 600)
  if (is.null(gamma_range))          gamma_range <- c(0, 0.25)
  if (is.null(reg_lambda_range))     reg_lambda_range <- c(1, 10)
  
  grid <- expand.grid(
    max_depth = max_depth_range,
    eta       = learning_rate_range,
    nrounds   = n_estimators_range,
    gamma     = gamma_range,
    lambda    = reg_lambda_range
  )
  
  neg_count <- sum(y_train == 0)
  pos_count <- sum(y_train == 1)
  scale_pos_weight <- neg_count / pos_count
  
  set.seed(random_state)
  n_rows <- nrow(X_train)
  shuffled_indices <- sample(1:n_rows)
  folds <- split(shuffled_indices, rep(1:cv, length.out = n_rows))
  
  calc_balanced_accuracy <- function(actual, pred_probs) {
    pred_classes <- ifelse(pred_probs > 0.5, 1, 0)
    
    TP <- sum(actual == 1 & pred_classes == 1)
    TN <- sum(actual == 0 & pred_classes == 0)
    FP <- sum(actual == 0 & pred_classes == 1)
    FN <- sum(actual == 1 & pred_classes == 0)
    
    sensitivity <- if ((TP + FN) == 0) 0 else TP / (TP + FN)
    specificity <- if ((TN + FP) == 0) 0 else TN / (TN + FP)
    
    return((sensitivity + specificity) / 2)
  }
  
  best_score <- -Inf
  best_tune <- NULL
  
  message(paste("Starting Native Grid Search with", cv, "-fold Cross-Validation..."))
  
  for (i in 1:nrow(grid)) {
    current_params <- grid[i, ]
    fold_scores <- numeric(cv)
    
    for (f in 1:cv) {
      val_idx <- folds[[f]]
      
      X_tr <- X_train[-val_idx, , drop = FALSE]
      y_tr <- y_train[-val_idx]
      X_va <- X_train[val_idx, , drop = FALSE]
      y_va <- y_train[val_idx]
      
      dtrain <- xgb.DMatrix(data = as.matrix(X_tr), label = y_tr)
      X_va_mat <- as.matrix(X_va)
      
      full_params <- list(
        objective        = "binary:logistic",
        eval_metric      = "logloss",
        max_depth        = current_params$max_depth,
        eta              = current_params$eta,
        gamma            = current_params$gamma,
        lambda           = current_params$lambda,
        scale_pos_weight = scale_pos_weight
      )
      
      model_native <- xgb.train(
        params  = full_params,
        data    = dtrain,
        nrounds = current_params$nrounds,
        verbose = 0
      )
      
      pred_probs <- predict(model_native, X_va_mat)
      fold_scores[f] <- calc_balanced_accuracy(y_va, pred_probs)
    }
    
    mean_cv_score <- mean(fold_scores, na.rm = TRUE)
    
    if (mean_cv_score > best_score) {
      best_score <- mean_cv_score
      best_tune <- current_params
    }
  }
  
  cat("\nBest parameters found (Native CV):\n")
  print(best_tune)
  cat(paste("  mean balanced_accuracy =", round(best_score, 4), "\n"))
  
  return(best_tune)
}