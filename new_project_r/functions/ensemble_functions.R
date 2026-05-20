library(caret)
library(glmnet)
library(ranger)
library(xgboost)

perform_model <- function(model, X_train, y_train, X_val, y_val, cv = 5, random_state = 123) {
  
  # Set up cross-validation
  set.seed(random_state)
  
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
    verboseIter = FALSE
  )
  
  y_factor <- as.factor(paste0("class_", y_train))
  
  # Train model with cross-validation
  model_cv <- train(
    x = X_train,
    y = y_factor,
    method = model$method,
    trControl = control,
    metric = "Balanced_Accuracy",
    tuneGrid = model$tuneGrid,
    ...
  )
  
  # Calculate CV stats
  cv_balanced_accuracy_mean <- mean(model_cv$resample$Balanced_Accuracy, na.rm = TRUE)
  cv_balanced_accuracy_sd <- sd(model_cv$resample$Balanced_Accuracy, na.rm = TRUE)
  
  # Make predictions on validation set
  y_val_factor <- as.factor(paste0("class_", y_val))
  predictions <- predict(model_cv, X_val, type = "raw")
  
  val_cm <- confusionMatrix(predictions, y_val_factor)
  val_balanced_accuracy <- val_cm$byClass["Balanced Accuracy"]
  
  # Return results
  return(list(
    model = model_cv,
    cv_balanced_accuracy_mean = cv_balanced_accuracy_mean,
    cv_balanced_accuracy_sd = cv_balanced_accuracy_sd,
    val_balanced_accuracy = val_balanced_accuracy
  ))
}
