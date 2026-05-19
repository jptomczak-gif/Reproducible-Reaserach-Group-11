library(caret)
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
  
  balanced_accuracy_summary <- function(data, lev = NULL, model = NULL) {
    lvls <- levels(data$obs)
    cm <- confusionMatrix(data$pred, data$obs, positive = lvls[2])
    out <- cm$byClass["Balanced Accuracy"]
    names(out) <- "Balanced_Accuracy"
    out
  }
  
  control <- trainControl(
    method = "cv",
    number = cv,
    summaryFunction = balanced_accuracy_summary,
    classProbs = TRUE,
    verboseIter = FALSE,
    allowParallel = TRUE
  )
  
  y_factor <- as.factor(paste0("class_", y_train))
  
  class_counts <- table(y_factor)
  scale_pos_weight <- class_counts[1] / class_counts[2]
  
  grid <- expand.grid(
    max_depth = max_depth_range,
    learning_rate = learning_rate_range,
    n_estimators = n_estimators_range,
    gamma = gamma_range,
    reg_lambda = reg_lambda_range
  )
  
  message("Starting Grid Search for XGBoost...")
  
  best_score <- -Inf
  best_tune_result <- NULL
  
  set.seed(random_state)
  model_tune <- train(
    x = X_train,
    y = y_factor,
    method = "xgbTree",
    tuneGrid = grid,
    trControl = control,
    metric = "Balanced_Accuracy",
    scale_pos_weight = scale_pos_weight,
    objective = "binary:logistic"
  )
  
  cat("\nBest parameters found:\n")
  print(model_tune$bestTune)
  cat(paste("  balanced_accuracy =", round(max(model_tune$results$Balanced_Accuracy, na.rm = TRUE), 4), "\n"))
  
  return(model_tune$bestTune)
}
