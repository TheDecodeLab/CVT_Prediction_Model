library(readxl)
library(dplyr)
library(readr)
library(caret)
library(mice)
library(ROCR)
library(pROC)
library(MLmetrics)
library(missForest)
library(reshape2)

non_significant_char_cols <- c('Type_Motor_Presentation_Symptoms', 'Type_Sensory_Presentation_Symptoms', 'cranial_type', 'Mechanical_type', 'Long_Flights_RiskFactor')
df <- read_excel("~/Data_CVT/final_data_revised_10022023_used_for_training.xlsx") %>%
  select(-non_significant_char_cols)
df[df == "nan"] <- NA
continuous_cols <- c("Age_of_presentation", 
                     "day_since_presentation",
                     "GCS_Score_ScoresAtAdmission",
                     "Gestational_age",
                     "Postpartum_Weeks_RiskFactor"
)
string_cols <- c("pt_id",
                 "Course_Presentation_Syndrome",
                 "Headache_Duration_Presentation_Symptoms",
                 "Level_consciousness",	
                 "Content_Confusion_level",
                 "Laterality_Visual_Symptoms_Presentation_Symptoms",
                 "Laterality_Motor_Presentation_Symptoms",
                 #"Type_Motor_Presentation_Symptoms",
                 "Laterality_Sensory_Presentation_Symptoms",
                 #"Type_Sensory_Presentation_Symptoms",
                 #"cranial_type",
                 #"Mechanical_type",
                 "Type_Seizure_Presentation_Symptoms"
)
binary_cols <- setdiff(names(df), c(continuous_cols, string_cols))
df[continuous_cols] <- lapply(df[continuous_cols], function(x) as.numeric(as.character(x)))
df[string_cols] <- lapply(df[string_cols], function(x) as.character(as.character(x)))
df[binary_cols] <- lapply(df[binary_cols], function(x) as.factor(as.character(x)))
missing_percent <- data.frame(round(100*colMeans(is.na(df)),2))
#############################
# Approach-1: use ALL variables, ignore missingness
#############################
mice_df_ALL_vars <- mice(df %>% select(-c("pt_id", "target_label")),
                         m = 10, maxit = 10, method = "pmm", seed=99)
#mice_df <- mice::complete(mice_df)
df_mice_ALL_vars <- cbind(df[2], mice::complete(mice_df_ALL_vars))
# train ML
# convert to a factor with levels 'X1' and 'X0'
df_mice_ALL_vars$target_label <- factor(df_mice_ALL_vars$target_label, levels = c(1, 0), labels = c("X1", "X0"))
row.has.na <- apply(df_mice_ALL_vars, 1, function(x){any(is.na(x))})
predictors_no_NA <- df_mice_ALL_vars[!row.has.na, ]
index <- caTools::sample.split(predictors_no_NA$target_label, SplitRatio = .7)
trainSet_ALL_vars <- subset(predictors_no_NA, index == TRUE)
testSets_ALL_vars <- subset(predictors_no_NA, index == FALSE)
ml.train.ALL_vars <- list()
classifiers <- c("glm","rf","svmRadial","xgbDART")
paramGrid <- trainControl(method = "repeatedcv",
                          number = 5,
                          repeats = 5,
                          summaryFunction = twoClassSummary,                      # Evaluate performance
                          classProbs = T,                                         # Estimate class probabilities
                          allowParallel = T,
                          search = "random")
for (c in 1:length(classifiers)) {
  print(paste("started:",classifiers[[c]],"at",Sys.time()))
  ml.train.ALL_vars[[paste(classifiers[[c]],sep = "")]] <- train(target_label~.,
                                                                            data = trainSet_ALL_vars,
                                                                            method = classifiers[[c]],
                                                                            preProcess = c("center","scale"),
                                                                            metric = "ROC",
                                                                            trControl = paramGrid,
                                                                            tuneLength = 5
  )
  print(paste("finished:",classifiers[[c]],"at",Sys.time()))
}
fetchResults <- function(x,y){
  z <- as.data.frame(x)
  colnames(z) <- names(y)
  return(z)
}
results.ALL.vars <- as.data.frame(list())
for  (c in 1:length(classifiers)) {
  predictions <- setNames(
    data.frame(
      testSets_ALL_vars$target_label,
      predict(object = ml.train.ALL_vars[[c]], testSets_ALL_vars, type = "prob"),
      predict(object = ml.train.ALL_vars[[c]], testSets_ALL_vars, type = "raw")
    ),
    c("obs","X0","X1","pred")
  )
  predictions$obs <- factor(predictions$obs)
  cm <- confusionMatrix(
    reference = predictions$obs,
    data = predictions$pred,
    mode = "everything",
    positive = "X1"
  )
  tmp <- as.data.frame(t(rbind(
    fetchResults(cm$byClass,ml.train.ALL_vars[c]),                                                             # Fetch Recall,Specificity,Precision
    fetchResults(cm$overall,ml.train.ALL_vars[c]),                                                             # Fetch Accuracy,95%CI
    fetchResults(as.data.frame(cm$table)$Freq,ml.train.ALL_vars[c]),                                           # Fetch TP,FP,FN,TN
    roc(predictor = predictions$X1,response = predictions$obs,
        levels = rev(levels(predictions$obs)))$auc,                                           # Calculate AUROC
    prSummary(predictions, lev = rev(levels(predictions$obs)))[1]                             # Calculate AUPR
  )))
  results.ALL.vars <- rbind(results.ALL.vars,
                            tmp%>%
                              mutate(
                                "Classifier" = names(ml.train.ALL_vars[c]),
                                "95%CI"= paste0("(",round(AccuracyLower,3),",",round(AccuracyUpper,3),")")
                              )%>%
                              select(
                                c("Classifier",AUROC="23",AUPR="24","Accuracy","95%CI",NIR="AccuracyNull",
                                  "Kappa","Sensitivity","Specificity",
                                  "Precision","F1","Prevalence",TP="4",FP="2",FN="3",TN="1")
                              )
  )
  rm(tmp,cm,predictions)
}
}
################
## Approach-2: drop missing> 50% i.e only 1 var: 'Cough_Aggravation_OR_Positive_Valsalva'
###############
mice_df_drop50perc_vars <- mice(df %>% select(-c("pt_id", "target_label", "Cough_Aggravation_OR_Positive_Valsalva")),
                                m = 10, maxit = 10, method = "pmm", seed=99)
df_drop50perc_vars <- cbind(df[2], mice::complete(mice_df_drop50perc_vars))
# train ML
# convert to a factor with levels 'X1' and 'X0'
df_drop50perc_vars$target_label <- factor(df_drop50perc_vars$target_label, levels = c(1, 0), labels = c("X1", "X0"))
row.has.na <- apply(df_drop50perc_vars, 1, function(x){any(is.na(x))})
predictors_no_NA <- df_drop50perc_vars[!row.has.na, ]
index <- caTools::sample.split(predictors_no_NA$target_label, SplitRatio = .7)
trainSet_drop50perc_vars <- subset(predictors_no_NA, index == TRUE)
testSets_drop50perc_vars <- subset(predictors_no_NA, index == FALSE)
ml.train.drop50perc_vars <- list()
for (c in 1:length(classifiers)) {
  print(paste("started:",classifiers[[c]],"at",Sys.time()))
  ml.train.drop50perc_vars[[paste(classifiers[[c]],sep = "")]] <- train(target_label~.,
                                                                        data = trainSet_drop50perc_vars,
                                                                        method = classifiers[[c]],
                                                                        preProcess = c("center","scale"),
                                                                        metric = "ROC",
                                                                        trControl = paramGrid,
                                                                        tuneLength = 5
  )
  print(paste("finished:",classifiers[[c]],"at",Sys.time()))
}
results.drop50perc.vars <- as.data.frame(list())
for  (c in 1:length(classifiers)) {
  predictions <- setNames(
    data.frame(
      testSets_drop50perc_vars$target_label,
      predict(object = ml.train.drop50perc_vars[[c]], testSets_drop50perc_vars, type = "prob"),
      predict(object = ml.train.drop50perc_vars[[c]], testSets_drop50perc_vars, type = "raw")
    ),
    c("obs","X0","X1","pred")
  )
  predictions$obs <- factor(predictions$obs)
  cm <- confusionMatrix(
    reference = predictions$obs,
    data = predictions$pred,
    mode = "everything",
    positive = "X1"
  )
  tmp <- as.data.frame(t(rbind(
    fetchResults(cm$byClass,ml.train.drop50perc_vars[c]),                                                             # Fetch Recall,Specificity,Precision
    fetchResults(cm$overall,ml.train.drop50perc_vars[c]),                                                             # Fetch Accuracy,95%CI
    fetchResults(as.data.frame(cm$table)$Freq,ml.train.drop50perc_vars[c]),                                           # Fetch TP,FP,FN,TN
    roc(predictor = predictions$X1,response = predictions$obs,
        levels = rev(levels(predictions$obs)))$auc,                                           # Calculate AUROC
    prSummary(predictions, lev = rev(levels(predictions$obs)))[1]                             # Calculate AUPR
  )))
  results.drop50perc.vars <- rbind(results.drop50perc.vars,
                                   tmp%>%
                                     mutate(
                                       "Classifier" = names(ml.train.drop50perc_vars[c]),
                                       "95%CI"= paste0("(",round(AccuracyLower,3),",",round(AccuracyUpper,3),")")
                                     )%>%
                                     select(
                                       c("Classifier",AUROC="23",AUPR="24","Accuracy","95%CI",NIR="AccuracyNull",
                                         "Kappa","Sensitivity","Specificity",
                                         "Precision","F1","Prevalence",TP="4",FP="2",FN="3",TN="1")
                                     )
  )
  rm(tmp,cm,predictions)
}
}
################
## Approach-3: drop missing> 25% i.e 4 vars 
###############
mice_df_drop25perc_vars <- mice(df %>% select(-c("pt_id", "target_label", "Cough_Aggravation_OR_Positive_Valsalva",
                                                       "Abortion", "Dizziness_Presentation_Symptoms", "Headache_Quality_Presentation_Symptoms")),
                                m = 25, maxit = 25, method = "pmm", seed=99)
df_drop25perc_vars <- cbind(df[2], mice::complete(mice_df_drop25perc_vars))
# train ML
# convert to a factor with levels 'X1' and 'X0'
df_drop25perc_vars$target_label <- factor(df_drop25perc_vars$target_label, levels = c(1, 0), labels = c("X1", "X0"))
row.has.na <- apply(df_drop25perc_vars, 1, function(x){any(is.na(x))})
predictors_no_NA <- df_drop25perc_vars[!row.has.na, ]
index <- caTools::sample.split(predictors_no_NA$target_label, SplitRatio = .7)
trainSet_drop25perc_vars <- subset(predictors_no_NA, index == TRUE)
testSets_drop25perc_vars <- subset(predictors_no_NA, index == FALSE)
ml.train.drop25perc_vars <- list()
for (c in 1:length(classifiers)) {
  print(paste("started:",classifiers[[c]],"at",Sys.time()))
  ml.train.drop25perc_vars[[paste(classifiers[[c]],sep = "")]] <- train(target_label~.,
                                                                        data = trainSet_drop25perc_vars,
                                                                        method = classifiers[[c]],
                                                                        preProcess = c("center","scale"),
                                                                        metric = "ROC",
                                                                        trControl = paramGrid,
                                                                        tuneLength = 5
  )
  print(paste("finished:",classifiers[[c]],"at",Sys.time()))
}
results.drop25perc.vars <- as.data.frame(list())
for  (c in 1:length(classifiers)) {
  predictions <- setNames(
    data.frame(
      testSets_drop25perc_vars$target_label,
      predict(object = ml.train.drop25perc_vars[[c]], testSets_drop25perc_vars, type = "prob"),
      predict(object = ml.train.drop25perc_vars[[c]], testSets_drop25perc_vars, type = "raw")
    ),
    c("obs","X0","X1","pred")
  )
  predictions$obs <- factor(predictions$obs)
  cm <- confusionMatrix(
    reference = predictions$obs,
    data = predictions$pred,
    mode = "everything",
    positive = "X1"
  )
  tmp <- as.data.frame(t(rbind(
    fetchResults(cm$byClass,ml.train.drop25perc_vars[c]),                                                             # Fetch Recall,Specificity,Precision
    fetchResults(cm$overall,ml.train.drop25perc_vars[c]),                                                             # Fetch Accuracy,95%CI
    fetchResults(as.data.frame(cm$table)$Freq,ml.train.drop25perc_vars[c]),                                           # Fetch TP,FP,FN,TN
    roc(predictor = predictions$X1,response = predictions$obs,
        levels = rev(levels(predictions$obs)))$auc,                                           # Calculate AUROC
    prSummary(predictions, lev = rev(levels(predictions$obs)))[1]                             # Calculate AUPR
  )))
  results.drop25perc.vars <- rbind(results.drop25perc.vars,
                                   tmp%>%
                                     mutate(
                                       "Classifier" = names(ml.train.drop25perc_vars[c]),
                                       "95%CI"= paste0("(",round(AccuracyLower,3),",",round(AccuracyUpper,3),")")
                                     )%>%
                                     select(
                                       c("Classifier",AUROC="23",AUPR="24","Accuracy","95%CI",NIR="AccuracyNull",
                                         "Kappa","Sensitivity","Specificity",
                                         "Precision","F1","Prevalence",TP="4",FP="2",FN="3",TN="1")
                                     )
  )
  rm(tmp,cm,predictions)
}
###################
# merge results
###################
results <- rbind(
  results.ALL.vars %>% mutate(iteration = "All_vars"),
  results.drop50perc.vars %>% mutate(iteration = "drop_gt50percMissing_vars"),
  results.drop25perc.vars %>% mutate(iteration = "drop_gt25percMissing_vars vars")
) %>%
  select(iteration, everything())
# radar plot
coord_radar <- function (theta = "x", start = 0, direction = -1) {
  theta <- match.arg(theta, c("x", "y"))
  r <- if (theta == "x") "y" else "x"
  ggproto("CordRadar", CoordPolar, theta = theta, r = r, start = start, 
          direction = sign(direction),
          is_linear = function(coord) TRUE)
}
fig.df <- melt(results %>% select('iteration', 'Classifier', 'AUROC', 'Accuracy', 'Sensitivity', 'Precision'), id.vars = c('Classifier', 'iteration')) %>%
  mutate(
    Classifier = case_when(
      Classifier == "glm"~"GLM", Classifier == "rf"~"Random Forest", Classifier == "xgbDART"~"XGB", Classifier == "svmRadial"~"SVM"
    ),
    iteration = case_when(
      iteration == "All_vars"~"using: All variables"
      , iteration == "drop_gt25percMissing_vars vars"~"using: variables with < 75% missingness"
      , iteration == "drop_gt50percMissing_vars"~"using: variables with < 50% missingness"
    ),
    variable = case_when(variable == "Sensitivity"~"Recall", TRUE ~ variable)
  )
tiff("figure.tiff", units = "in", width =10,height = 10, res = 300, compression = 'jpeg')
fig.df %>% ggplot(aes(x = variable, y = value, group = Classifier, fill = Classifier)) +
  geom_polygon(alpha = .25, size = 1) +
  coord_polar(start = -pi) +
  facet_wrap(iteration ~.) +
  theme_light() +
  theme(
    axis.title.x = element_blank() 
    , strip.text = element_text(face = 'bold')
    , legend.position = 'top'
  ) +
  labs(y = 'Performance Measure') +
  guides(fill = guide_legend(nrow = 1))
dev.off()
# feature importance
get_feaImp <- function(ML.iteration, tmp){
  for (i in 1:length(ML.iteration)) {
    ifelse(
      grepl(paste(c('glm','rf','xgb'), collapse = '|'),                           # Check for 'pattern'(glm|rf|xgb) 
            names(ML.iteration[i])),                                               # in the 'string'(female.age.lt.40.glm)
      # if TRUE then do
      tmp <- full_join(tmp, varImp(ML.iteration[[i]])$importance%>%              
                                     mutate(features = rownames(.),
                                            model = names(ML.iteration[i]))%>%
                                     select(features, model, Overall)%>%
                                     arrange(model, features)%>%
                                     rename(!!quo_name(names(ML.iteration[i])) := Overall), # dynamically rename the default 'Overall' column
                                   by = c('features')
      )%>%
        select(-starts_with('model'))%>%
        replace(is.na(.), 0) , 
      # Else, then do
      print("it's SVM, ignore!!")                                               
    )
  }
  return(tmp)
}
feaImp.ALL_vars <- data.frame(features = character(0))                                   # create empty data frame with a column to append data iterative
feaImp.ALL_vars <- get_feaImp(ml.train.ALL_vars, feaImp.ALL_vars)
feaImp.ALL_vars$iteration <- "All Variables"
feaImp.ALL_vars <- melt(feaImp.ALL_vars)

feaImp.drop25perc_vars <- data.frame(features = character(0))                                   # create empty data frame with a column to append data iterative
feaImp.drop25perc_vars <- get_feaImp(ml.train.drop25perc_vars, feaImp.drop25perc_vars)
feaImp.drop25perc_vars$iteration <- "Drop > 25% missing Variables"
feaImp.drop25perc_vars <- melt(feaImp.drop25perc_vars)

feaImp.drop50perc_vars <- data.frame(features = character(0))                                   # create empty data frame with a column to append data iterative
feaImp.drop50perc_vars <- get_feaImp(ml.train.drop50perc_vars, feaImp.drop50perc_vars)
feaImp.drop50perc_vars$iteration <- "Drop > 50% missing variables"
feaImp.drop50perc_vars <- melt(feaImp.drop50perc_vars)

feaImp.avg <- rbind(feaImp.ALL_vars, feaImp.drop25perc_vars, feaImp.drop50perc_vars)%>%
  select(-c(variable))%>%
  group_by(features,iteration)%>%
  summarise_all(.,funs(mean = mean))

feaImp.avg %>%
  ggplot(aes(x = mean, y = features)) +
  geom_bar(stat="identity") +
  facet_wrap(.~iteration) +
  theme_bw() +
  theme(
      axis.text = element_text(size = 8)
    , axis.title = element_text(size = 10)
    , strip.text = element_text(size = 10)
  ) +
  labs(x = "Mean of Feature Importance", y = "")
