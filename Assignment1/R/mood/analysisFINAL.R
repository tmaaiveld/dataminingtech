##################################################
## Project: Data Mining - Smartphone Data
## Created: 04/04/2019
## Last modified: 17/04/2019
## Author: Tommy Maaiveld
##################################################
##################################################
## Libraries & Constants
##################################################
# install.packages("caret", "tidyverse", "Hmisc", "ggplot", "ggcorrplot", 
#                   "corrplot", "party","psych", "e1071", "neuralnet", 
#                   "gridExtra", "ggpubr", "dplyr","tidyr", "factoextra"
#                   "mlbench", "keras", "devtools", "beepr") # if not installed

library(caret, quietly=TRUE)
library(ggplot2, quietly=TRUE)
library(ggcorrplot, quietly=TRUE)
library(Hmisc, quietly=TRUE)
library(rpart, quietly=TRUE)
library(party, quietly=TRUE)
library(psych, quietly=TRUE)
library(e1071, quietly=TRUE)
library(neuralnet, quietly=TRUE)
library(gridExtra, quietly=TRUE)
library(ggpubr, quietly=TRUE)
library(factoextra, quietly=TRUE)
library(BBmisc, quietly=TRUE)
library(mlbench, quietly=TRUE)
library(tidyverse, quietly=TRUE)
library(beepr, quietly=TRUE)
library(MLmetrics, quietly=TRUE)
library(doParallel, quietly=TRUE)
library(rnn, quietly=TRUE)

registerDoParallel(cores = 8)

SEED = 123
PREDICTION_WINDOW = 5
TEST_TRAIN_PROP = 0.66
FEATURE_COUNT = 19

##################################################
## Raw Data Processing
##################################################

data <- read.csv(file="data/dataset_mood_smartphone.csv") %>%
  tbl_df() %>%
  separate(time, c("date", "time"), sep=" ") %>%
  spread(variable,value) %>% # organise
  mutate(appCat.builtin=replace(appCat.builtin, appCat.builtin<0, NA)) %>%
  group_by(id,date)

data_raw <- data

ordvars = names(data)[1:4]
meanvars = c(names(data)[5],names(data)[19:21])
sumvars = names(data)[! names(data) %in% c(meanvars, ordvars)]

data_agg = merge(
  arrange(aggregate(data[meanvars], by=list(data$id, data$date),
                    FUN=mean, na.rm=TRUE), Group.1,Group.2),
  arrange(aggregate(data[sumvars], by=list(data$id, data$date),
                    FUN=sum,  na.rm=TRUE), Group.1,Group.2)
  )[, c(1,2,6,4,5,3,7:21)] %>% 
  dplyr::rename(id=Group.1, date=Group.2) %>%
  tbl_df()

##################################################
## Standardization, Normalization & Transformation
##################################################

# Produce a standardized version of the data
preprocessParams.data_agg.std <- preProcess(data_agg, method=c("center", "scale"))
print(preprocessParams.data_agg.std)

data_agg.std <- data_agg %>% 
  predict(preprocessParams.data_agg.std, .)
summary(data_agg.std)

# Produce a normalized version of the data
preprocessParams.data_agg.nrm <- preProcess(data_agg, method=c("range"))
print(preprocessParams.data_agg.nrm)

data_agg.nrm <- data_agg %>%
  predict(preprocessParams.data_agg.nrm, .)
summary(data_agg.nrm)

# perform data transformations
data_agg.raw <- data_agg
data_agg <- data_agg %>%
  ungroup() %>%
  mutate(
    activity=sqrt(data_agg.nrm$activity)^(1/2),
    circumplex.arousal=(data_agg.std$circumplex.arousal),   # didn't transform, as they are already normally distributed.
    circumplex.valence=(data_agg.std$circumplex.valence),
    appCat.builtin=(data_agg.nrm$appCat.builtin)^(1/2),
    appCat.communication=(data_agg.nrm$appCat.communication)^(1/2),
    appCat.entertainment=(data_agg.nrm$appCat.entertainment)^(1/2),
    appCat.finance=(data_agg.nrm$appCat.finance)^(1/2),
    appCat.game=(data_agg.nrm$appCat.game)^(1/2),
    appCat.office=(data_agg.nrm$appCat.office)^(1/2),
    appCat.other=(data_agg.nrm$appCat.other)^(1/2),
    appCat.social=(data_agg.nrm$appCat.social)^(1/2),
    appCat.travel=(data_agg.nrm$appCat.travel)^(1/2),
    appCat.unknown=(data_agg.nrm$appCat.unknown)^(1/2),
    appCat.utilities=(data_agg.nrm$appCat.utilities)^(1/2),
    appCat.weather=(data_agg.nrm$appCat.weather)^(1/2),
    screen=(data_agg.nrm$screen)^(1/2),
    call=(data_agg$call)^(1/2),
    sms=(data_agg$sms)^(1/2)
  )
# Report: did not transform count data. See resources.

##################################################
## Exploration & Visualisation [Commented]
##################################################

# Distribution of the target variable
par(mfrow=c(1,2)); boxplot(data_agg.raw[3],main='mood');hist(data_agg[3],main='mood')

# Transformation comparisons for three variables
{
par(mfrow=c(1,3))
boxplot(c(data_agg.raw[4],data_agg[4]),col="red",main="circumplex.arousal (Standardized)",names=c("untransformed","transformed"))
boxplot(c(data_agg.raw[5],data_agg[5]),col="cyan",main="circumplex.valence (Standardized)",names=c("untransformed","transformed"))
boxplot(c(data_agg.raw[6],data_agg[6]),col="green",main="activity (Normalized,sqrt())",names=c("untransformed","transformed"))
}

# Pre-transformed visualisations
data_agg.raw[7:18] %>% replace(.== 0, NA) %>%
  hist()

# Transformed visualisations
data_agg[7:18] %>% replace(.== 0, NA) %>%
  hist(.,main="distribution of appCat variables")

# Generate a pairplot of the first four variables
pairs.panels(data_agg.raw[3:6],
             method = "pearson",
             hist.col = "#00AFBB", cex.cor = 0.8,
             density = TRUE, ellipses = FALSE
)

## generate correlation plot to examine collinearity
data_agg.rcorr <- rcorr(as.matrix(data_agg.raw[3:length(data_agg)]))
ggcorrplot(data_agg.rcorr$r)

##################################################
## Feature Construction 
##################################################

# create a feature set
Fset <- as.tibble(matrix(data=NA,
                         nrow=nrow(data_agg),
                         ncol=FEATURE_COUNT)) %>%
  bind_cols(data_agg[,1:2], .,data_agg[,"mood"])

# store and rename feature names
features <- sprintf("F%s", 1:19)
feature_interp <- paste("5d", colnames(data_agg)[3:21], sep = ".")
colnames(Fset)[3:length(Fset)] <- c(features, "target")

# Data imputation via the mean
data_missing <- data_agg
for (i in 4:length(data_agg)) {
data_agg[i] <- data_agg[[i]] %>% 
    replace_na(mean(data_agg[[i]], na.rm=TRUE)) 
}

# populate cells with 5-day estimates of each variable
attach(data_agg)
for (patient in levels(id)) {
  for(i in ((PREDICTION_WINDOW+1):nrow(data_agg[id==patient,]))) {
    window <- ((i-PREDICTION_WINDOW):(i-1)) + (match(patient,id)-1)

    for (k in (3:21)) {
      if (k<7) Fset[(i-1+match(patient, data_agg$id)),k]  <- mean(pull(data_agg[window,k]), na.rm = TRUE)
      if (k>6) Fset[(i-1+match(patient, data_agg$id)),k]  <- sum(pull(data_agg[window,k]), na.rm = TRUE)
    }
  }
}
detach(data_agg)

# drop missing value rows after imputation
Fset <- Fset %>% drop_na()

summary (Fset)

# write a .csv of daily aggregates, with normalisation and transformation applied
write.csv(data_agg,file="aggregates.csv")
# write a .csv of the feature set (5 day aggregates), with missing values imputed and missing target value rows removed
write.csv(Fset,file="features.csv")

##################################################
## Modeling
##################################################
 
# Test / Train set split parameters
set.seed(SEED)
train_size = TEST_TRAIN_PROP * nrow(Fset)

### train indices for a train/test split
train_ind <- sample(seq_len(nrow(Fset)), size = train_size)

## PCA
preprocessParams.pca <- preProcess(Fset[train_ind,3:(length(Fset)-1)], 
                                   method=c("center", "scale", "pca"))
print(preprocessParams.pca)
Fset.pca <- Fset[,3:(length(Fset)-1)] %>%
  predict(preprocessParams.pca, .) %>%
  bind_cols(Fset[,"target"]) %>%
  tbl_df()
summary(Fset.pca)

princomps <- colnames(Fset.pca[1:(length(Fset.pca)-1)])

# Models
## Decision Tree 
tree.unt <- ctree(as.formula(paste("target~", paste(features, collapse="+"))), 
                  data=Fset[train_ind,])
tree.pca <- ctree(as.formula(paste("target~", paste(princomps, collapse="+"))), 
                  data=Fset.pca[train_ind,])
tree.hyp <- ctree(as.formula(paste("target~", paste(c(features,"F1*F2","F1*F3","F1*F4"), collapse="+"))), 
                  data=Fset[train_ind,])

## Generalized Linear Model
glm.unt <- glm(as.formula(paste("target~", paste(features, collapse="+"))), 
               data=Fset[train_ind,])
glm.pca <- glm(as.formula(paste("target~", paste(princomps, collapse="+"))), 
               data=Fset.pca[train_ind,])
glm.hyp <- glm(as.formula(paste("target~", paste(c(features,"(F1*F2)","(F1*F3)","(F1*F4)"), collapse="+"))), 
               data=Fset[train_ind,])

## SVM
svm.unt <- svm(as.formula(paste("target~", paste(features, collapse="+"))), 
               data=Fset[train_ind,])
svm.pca <- svm(as.formula(paste("target~", paste(princomps, collapse="+"))), 
               data=Fset.pca[train_ind,])
svm.hyp <- svm(as.formula(paste("target~", paste(c(features,"F1*F2","F1*F3","F1*F4"), collapse="+"))), 
               data=Fset[train_ind,])

## Individual models
### Random examples
# random_parts <- sample(levels(data$id),3)
random_parts <- c("AS14.02","AS14.12","AS14.25") # working set...

svm.P01 <- svm(as.formula(paste("target~", paste(features, collapse="+"))), 
               data=filter(Fset[train_ind,],id==random_parts[1]))
svm.P01.pr <- predict(svm.P01, newdata=filter(Fset[-train_ind,],id==random_parts[1]), type="response")

svm.P02 <- svm(as.formula(paste("target~", paste(features, collapse="+"))), 
               data=filter(Fset[train_ind,],id==random_parts[2]))
svm.P02.pr <- predict(svm.P02, newdata=filter(Fset[-train_ind,],id==random_parts[2]), type="response")

svm.P03 <- svm(as.formula(paste("target~", paste(features, collapse="+"))), 
               data=filter(Fset[train_ind,],id==random_parts[3]))
svm.P03.pr <- predict(svm.P03, newdata=filter(Fset[-train_ind,],id==random_parts[3]), type="response")

### Full estimate sets, combining all participants
part_estimates.tree <- NULL
part_estimates.glm  <- NULL
part_estimates.svm  <- NULL

for (part in levels(data$id)) {
  # tree
  part_estimates.tree <- ctree(as.formula(paste("target~", paste(features, collapse="+"))), 
                               data=filter(Fset[train_ind,],id==part)) %>%
    predict(newdata=filter(Fset[-train_ind,],id==part),type="response") %>%
    c(part_estimates.tree,.) %>%
    as.matrix()
  
  # glm
  part_estimates.glm <- glm(as.formula(paste("target~", paste(features, collapse="+"))), 
                            data=filter(Fset[train_ind,],id==part)) %>%
    predict(newdata=filter(Fset[-train_ind,],id==part),type="response") %>%
    c(part_estimates.glm,.) %>%
    as.matrix()
  
  # svm
  part_estimates.svm <- svm(as.formula(paste("target~", paste(features, collapse="+"))), 
                            data=filter(Fset[train_ind,],id==part)) %>%
    predict(newdata=filter(Fset[-train_ind,],id==part),type="response") %>%
      c(part_estimates.svm,.) %>%
    as.matrix()
}

trees <- list(tree.unt, tree.pca)
# generate plots of the models 
for (i in 1:2) {
  plot(trees[[i]],
       main = c("Tree model trained on standard features",
                "Tree model trained on principal components of features")[i])
}
plot(svm.unt) # (doesn't return anything for regression)
plot(glm.unt)

##################################################
## Prediction Generation
##################################################

# Generating predictions for all model and feature set combinations
pr <- bind_cols(
  ## tree model predictions
  "tree.unt.pr"=predict(tree.unt, newdata=Fset[-train_ind,], type="response"),
  "tree.pca.pr"=predict(tree.pca, newdata=Fset.pca[-train_ind,], type="response"),
  "tree.prs.pr"=part_estimates.tree,

  ## generalized linear model predictions
  "glm.unt.pr"=predict(glm.unt, newdata=Fset[-train_ind,], type="response"),
  "glm.pca.pr"=predict(glm.pca, newdata=Fset.pca[-train_ind,], type="response"),
  "glm.prs.pr"=part_estimates.glm,
  

  ## SVM model predictions
  "svm.unt.pr"=predict(svm.unt, newdata=Fset[-train_ind,], type="response"),
  "svm.pca.pr"=predict(svm.pca, newdata=Fset.pca[-train_ind,], type="response"),
  "svm.prs.pr"=part_estimates.svm,
  
  # Target variable
  Fset[-train_ind,"target"]
) %>% tbl_df

MSE <- function(x,y) {mean(as.matrix(((x-y)^2)),na.rm=TRUE)}

attach(pr)
MSEscores = data.frame(row.names=c("Untransformed","Principal Components","Individuals"),
                    "tree"=c(MSE(tree.unt.pr,target),
                             MSE(tree.pca.pr,target),
                             MSE(tree.prs.pr,target)),
                     "glm"=c(MSE(glm.unt.pr,target),
                             MSE(glm.pca.pr,target),
                             MSE(glm.prs.pr,target)),
                     "svm"=c(MSE(svm.unt.pr,target),
                             MSE(svm.pca.pr,target),
                             MSE(svm.prs.pr,target)),
               "benchmark"=  c(MSE(Fset[-train_ind,][-1,"target"],target),
                               NA,NA)
) %>% round(2)

MisClas = data.frame(row.names=c("Untransformed","Principal Components","Individuals"),
            "tree"=c(colSums((round(tree.unt.pr)-round(target))!=0),
                     colSums((round(tree.pca.pr)-round(target))!=0),
                     colSums((round(tree.prs.pr)-round(target))!=0)),
             "glm"=c(colSums((round(as.matrix(glm.unt.pr))-round(target))!=0),
                     colSums((round(as.matrix(glm.pca.pr))-round(target))!=0),
                     colSums((round(as.matrix(glm.prs.pr))-round(target))!=0)),
             "svm"=c(colSums((round(as.matrix(svm.unt.pr))-round(target))!=0),
                     colSums((round(as.matrix(svm.pca.pr))-round(target))!=0),
                     colSums((round(as.matrix(svm.prs.pr))-round(target))!=0)),
       "benchmark"=c(colSums((round(Fset[-train_ind,][-1,"target"])-round(target))!=0), 
                     NA,NA))
detach(pr)

# tabulate personal results
MSEPers <- bind_cols(P1=MSE(svm.P01.pr,filter(Fset[-train_ind,],id==random_parts[1])[,"target"]),
                     P2=MSE(svm.P02.pr,filter(Fset[-train_ind,],id==random_parts[2])[,"target"]),
                     P3=MSE(svm.P03.pr,filter(Fset[-train_ind,],id==random_parts[3])[,"target"])
)
colnames(MSEPers) <- random_parts
MSEPers <- MSEPers[,order(colnames(MSEPers))]

print(MSEscores)
print(MisClas)
print(round(MSEPers,2))
print(round(MisClas*(1/nrow(data_agg)),2))

beep()