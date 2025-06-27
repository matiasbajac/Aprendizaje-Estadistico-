# Librerías
library(targets)
library(future)
library(tidymodels)
library(tune)
library(workflowsets)
library(ranger)
library(glmnet)
library(caret) 

tar_option_set(
  packages = c("tidymodels", "tune", "workflowsets", "ranger", "glmnet","caret","baguette","xgboost","kknn"), # packages that your targets need to run
  format = "rds", # default storage format
  # Set other options as needed.
  memory = "transient"  # Esto evitará que `.Random.seed` se guarde
  )
actres_train <- readRDS("datos/train.RDS")

set.seed(501)
list(

  
  # Preprocesamiento general
  tar_target(
    general_rec,
    recipe(aprobada ~ ., data = actres_train) %>%
      update_role(id, new_role = "id") %>%
      step_normalize(all_numeric_predictors()) |>
      step_dummy(all_nominal_predictors())),
  
  # Especificaciones de modelos
  tar_target(
    bag_cart_spec,
    bag_tree(min_n = tune()) %>% 
      set_engine("rpart", times = 50) %>%
      set_mode("classification")),
  tar_target(
    rf_spec,
    rand_forest(min_n = tune(), trees = 50) %>%
      set_engine("ranger") %>%
      set_mode("classification")),
  tar_target(
    reg_log_lasso_spec,
    logistic_reg(penalty = tune(), mixture = 1) %>%
      set_mode("classification") %>%
      set_engine("glmnet")),
  tar_target(
    knn_spec,
    nearest_neighbor(
      neighbors = tune()  #Con tune() indicamos parámetro a ser ajustado.
    ) |>
      set_engine('kknn') %>%
      set_mode("classification")
  ),

  # Workflow conjunto
  tar_target(
    general,
    workflow_set(
      preproc = list(general_rec),
      models = list(
        Bagging = bag_cart_spec, 
        RF = rf_spec,
        Reg_log = reg_log_lasso_spec,
        KNN = knn_spec
      ))),
  
  # Validación cruzada
  tar_target(
    actres_folds,
    vfold_cv(actres_train, repeats = 5)),
  
  # Control para el grid search
  tar_target(
    grid_ctrl,
    control_grid(
      save_pred = TRUE,
      parallel_over = "resamples",
      save_workflow = TRUE)),
  
  # Resultados del grid search
  tar_target(
    grid_results,
    general %>%
      workflow_map(
       # seed = 1503,
        resamples = actres_folds,
        grid = 25,
        control = grid_ctrl)),

  # Evaluación
  tar_target(
    best_results,
    grid_results %>%
      rank_results(select_best = TRUE))
)
