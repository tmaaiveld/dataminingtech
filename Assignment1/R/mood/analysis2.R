##################################################
## Project: Data Mining - Smartphone Data
## Date:08/04/2019
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
  mutate(appCat.builtin=replace(appCat.builtin, appCat.builtin<0, NA)) %>% #clean
  group_by(id,date)
data$date <- as.Date(data$date, origin = min(data$date))

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
## Exploration & Visualisation
##################################################

# histogram visualisations
summary(data_agg)

par(mfrow=c(1,2)); boxplot(data_agg[3],main='mood');hist(data_agg[3],main='mood')
par(mfrow=c(1,2)); boxplot(data_agg[4:5],main='arousal, valence');boxplot(data_agg[6],main='activity')
par(mfrow=c(1,1))

# histogram with zero appCat values removed
hist(data_agg[,-c(1,2)])
data_agg[7:18] %>% replace(.== 0, NA) %>%
  hist()

pairs.panels(data_agg[3:6], 
             method = "pearson",
             hist.col = "#00AFBB", cex.cor = 0.8,
             density = TRUE, ellipses = FALSE
)

## generate correlation plot to examine collinearity
data_agg.rcorr <- rcorr(as.matrix(data_agg[3:length(data_agg)]))
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
features <- sprintf("F%s", 1:(length(Fset)-3))
feature_interp <- paste("5d", colnames(data_agg)[3:21], sep = ".")
colnames(Fset)[3:length(Fset)] <- c(features, "target")

# Data imputation via the mean
data_missing <- data_agg
for (i in 4:length(data_agg)) {
data_agg[i] <-  data_agg[[i]] %>% 
    replace_na(mean(data_agg[[i]], na.rm=TRUE)) 
}

# populate cells with 5-day estimates of each variable
attach(data_agg)
for (patient in levels(id)) {
  for(i in (6:nrow(data_agg[id==patient,]))) {
    window <- ((i-PREDICTION_WINDOW):(i-1)) + (match(patient,id)-1)

    for (k in (3:21)) {
      if (k<7) Fset[(i-1+match(patient, data_agg$id)),k]  <- mean(pull(data_agg[window,k]), na.rm = TRUE)
      if (k>6) Fset[(i-1+match(patient, data_agg$id)),k]  <- sum(pull(data_agg[window,k]), na.rm = TRUE)
    }
  }
}
detach(data_agg)

# drop missing value rows after imputation (buggy?)
Fset <- Fset %>% drop_na()

# Feature transformation

summary (Fset)

## standardized data
preprocessParams.std <- preProcess(Fset, method=c("center", "scale"))
print(preprocessParams.std)
Fset.std <- Fset %>% 
  predict(preprocessParams.std, .) %>%
  {bind_cols((.[1:(length(.)-1)]),Fset[,"target"])}
summary(Fset.std)

## normalized data
preprocessParams.nrm <- preProcess(Fset, method=c("range"))
print(preprocessParams.nrm)
Fset.nrm <- Fset %>%
  predict(preprocessParams.nrm, .) %>%
  {bind_cols(.[1:(length(Fset)-1)], Fset[,"target"])}
summary(Fset.nrm)
  
## mixed data (norm sumvars, stand contvars, do BoxCox?)
Fset.mix <- bind_cols(Fset.std[1:4], Fset.nrm[5:length(Fset)])
summary(Fset.mix)

# try more of this with a quick NN, see if you can maximise predictive power, then run a bigger nn.
#implement boxcox, etc.? More custom dataset variation

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
tree.std <- ctree(as.formula(paste("target~", paste(features, collapse="+"))), 
                  data=Fset.std[train_ind,])
tree.nrm <- ctree(as.formula(paste("target~", paste(features, collapse="+"))), 
                  data=Fset.nrm[train_ind,])
tree.pca <- ctree(as.formula(paste("target~", paste(princomps, collapse="+"))), 
                  data=Fset.pca[train_ind,])
tree.mix <- ctree(as.formula(paste("target~", paste(features, collapse="+"))), 
                  data=Fset.mix[train_ind,])
trees <- list(tree.unt,tree.std, tree.nrm, tree.pca, tree.mix)

## Generalized Linear Model
glm.unt <- glm(as.formula(paste("target~", paste(features, collapse="+"))), 
               data=Fset[train_ind,])
glm.std <- glm(as.formula(paste("target~", paste(features, collapse="+"))), 
               data=Fset.std[train_ind,])
glm.nrm <- glm(as.formula(paste("target~", paste(features, collapse="+"))), 
               data=Fset.nrm[train_ind,])
glm.pca <- glm(as.formula(paste("target~", paste(princomps, collapse="+"))), 
               data=Fset.pca[train_ind,])
glm.mix <- glm(as.formula(paste("target~", paste(features, collapse="+"))), 
               data=Fset.mix[train_ind,])
glms = list(glm.unt,glm.std,glm.nrm,glm.pca,glm.mix)

## SVM
svm.unt <- svm(as.formula(paste("target~", paste(features, collapse="+"))), 
               data=Fset[train_ind,])
svm.std <- svm(as.formula(paste("target~", paste(features, collapse="+"))), 
               data=Fset.std[train_ind,])
svm.nrm <- svm(as.formula(paste("target~", paste(features, collapse="+"))), 
               data=Fset.nrm[train_ind,])
svm.pca <- svm(as.formula(paste("target~", paste(princomps, collapse="+"))), 
               data=Fset.pca[train_ind,])
svm.mix <- svm(as.formula(paste("target~", paste(features, collapse="+"))), 
               data=Fset.mix[train_ind,])
svms = list(svm.unt,svm.std,svm.nrm,svm.pca,svm.mix)

# Old neural net
{
# nn.unt <- Fset.std[train_ind,] %>%
#   neuralnet(as.formula(paste("target~", paste(features, collapse="+"))),
#             hidden=c(10,5),stepmax=1e6,threshold=0.01,linear.output=T,data=.)
# nn.std <- Fset.std[train_ind,] %>%
#   neuralnet(as.formula(paste("target~", paste(features, collapse="+"))),
#             hidden=c(10,5),stepmax=1e6,threshold=0.01,linear.output=T,data=.)
# nn.nrm <- Fset.nrm[train_ind,] %>%
#   neuralnet(as.formula(paste("target~", paste(features, collapse="+"))),
#             hidden=c(10,5),stepmax=1e6,threshold=0.02, linear.output=T,data=.)
# nn.pca <- Fset.pca[train_ind,] %>%
#   neuralnet(as.formula(paste("target~", paste(princomps, collapse="+"))),
#             hidden=c(10,5),stepmax=1e6,threshold=0.01,linear.output=T,data=.)
# nn.mix <- Fset.mix[train_ind,] %>%
#   neuralnet(as.formula(paste("target~", paste(features, collapse="+"))),
#             hidden=c(10,5),stepmax=1e6,threshold=0.01,linear.output=T,data=.)
}

# New neural net
data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, num_hidden=1)
lro <- mx.symbol.LinearRegressionOutput(fc1)
mx.set.seed(0)
model <- mx.model.FeedForward.create(
  lro, 
  X=data.matrix(Fset[train_ind,3:(length(Fset)-1)]), 
  y=data.matrix(Fset[-train_ind,(length(Fset))]),
  eval.data=list(data=data.matrix(Fset[-train_ind,(3:(length(Fset)-1))]), 
                 label=data.matrix(Fset[-train_ind,length(Fset)])),
  
  ctx=mx.cpu(), num.round=10, array.batch.size=20,array.layout ="auto",
  learning.rate=2e-6, momentum=0.9, eval.metric=mx.metric.rmse)
mx.visualization.print_summary() 

#  nns = list(nn.unt,nn.std,nn.nrm,nn.pca,nn.mix)


# try generating a table for parameter adjustment nn? different layer counts etc
# ^prob not feasible

# generate plots of the models 
for (i in 1:5) {
  plot(trees[[i]],
       main = c("Tree model trained on untransformed data",
                "Tree model trained on standardized features",
                "Tree model trained on normalized features",
                "Tree model trained on principal components of features",
                "Tree model trained on mixed features")[i])
}
plot(svm.std) # (doesn't return anything for regression)
plot(glm.std)
plot(nn.pca)

##################################################
## Prediction Generation
##################################################

#if the NN fails to converge, fill in NA:


# Generating predictions for all model and feature set combinations
pr <- bind_cols(
  ## tree model predictions
  "tree.unt.pr"=predict(tree.unt, newdata=Fset[-train_ind,], type="response"),
  "tree.std.pr"=predict(tree.std, newdata=Fset.std[-train_ind,], type="response"),
  "tree.nrm.pr"=predict(tree.nrm, newdata=Fset.nrm[-train_ind,], type="response"),
  "tree.pca.pr"=predict(tree.pca, newdata=Fset.pca[-train_ind,], type="response"),
  "tree.mix.pr"=predict(tree.mix, newdata=Fset.mix[-train_ind,], type="response"),
  
  ## generalized linear model predictions
  "glm.unt.pr"=predict(glm.unt, newdata=Fset[-train_ind,], type="response"),
  "glm.std.pr"=predict(glm.std, newdata=Fset.std[-train_ind,], type="response"),
  "glm.nrm.pr"=predict(glm.nrm, newdata=Fset.nrm[-train_ind,], type="response"),
  "glm.pca.pr"=predict(glm.pca, newdata=Fset.pca[-train_ind,], type="response"),
  "glm.mix.pr"=predict(glm.mix, newdata=Fset.mix[-train_ind,], type="response"),

  ## SVM model predictions
  "svm.unt.pr"=predict(svm.unt, newdata=Fset[-train_ind,], type="response"),
  "svm.std.pr"=predict(svm.std, newdata=Fset.std[-train_ind,], type="response"),
  "svm.nrm.pr"=predict(svm.nrm, newdata=Fset.nrm[-train_ind,], type="response"),
  "svm.pca.pr"=predict(svm.pca, newdata=Fset.pca[-train_ind,], type="response"),
  "svm.mix.pr"=predict(svm.mix, newdata=Fset.mix[-train_ind,], type="response"),

  # ## Neural net predictions
  # "nn.unt.pr"=predict(nn.unt, newdata=Fset[-train_ind,], type="response"),
  # "nn.std.pr"=predict(nn.std, newdata=Fset.std[-train_ind,], type="response"),
  # "nn.nrm.pr"=predict(nn.nrm, newdata=Fset.nrm[-train_ind,], type="response"),
  # "nn.pca.pr"=predict(nn.pca, newdata=Fset.pca[-train_ind,], type="response"),
  # "nn.mix.pr"=predict(nn.mix, newdata=Fset.mix[-train_ind,], type="response"),
  
  # Target variable
  Fset[-train_ind,"target"]
) %>% tbl_df

MSE <- function(x,y) {mean(as.matrix(((x-y)^2)),na.rm=TRUE)}

attach(pr)
MSEscores = data.frame(row.names=c("Untransformed","Standardized","Normalized",
                               "Principal Components", "Mixed Variables"),
                    "tree"=c(MSE(tree.unt.pr,target),
                             MSE(tree.std.pr,target),
                             MSE(tree.nrm.pr,target),
                             MSE(tree.pca.pr,target),
                             MSE(tree.mix.pr,target)),
                     "glm"=c(MSE(glm.unt.pr,target),
                             MSE(glm.std.pr,target),
                             MSE(glm.nrm.pr,target),
                             MSE(glm.pca.pr,target),
                             MSE(glm.mix.pr,target)),
                     "svm"=c(MSE(svm.unt.pr,target),
                             MSE(svm.std.pr,target),
                             MSE(svm.nrm.pr,target),
                             MSE(svm.pca.pr,target),
                             MSE(svm.mix.pr,target)),
                      # "nn"=c(MSE(nn.unt.pr,target),
                      #        MSE(nn.std.pr,target),
                      #        MSE(nn.nrm.pr,target),
                      #        MSE(nn.pca.pr,target),
                      #        MSE(nn.mix.pr,target)),
               "benchmark"=  c(MSE(Fset[-train_ind,][-1,"target"],target),
                               NA,NA,NA,NA)
) %>% round(2)

MisClas = data.frame(row.names=c("Untransformed","Standardized","Normalized",
                       "Principal Components", "Mixed Variables"),
            "tree"=c(colSums((round(tree.unt.pr)-round(target))!=0),
                     colSums((round(tree.std.pr)-round(target))!=0),
                     colSums((round(tree.nrm.pr)-round(target))!=0),
                     colSums((round(tree.pca.pr)-round(target))!=0),
                     colSums((round(tree.mix.pr)-round(target))!=0)),
             "glm"=c(colSums((round(as.matrix(glm.unt.pr))-round(target))!=0),
                     colSums((round(as.matrix(glm.std.pr))-round(target))!=0),
                     colSums((round(as.matrix(glm.nrm.pr))-round(target))!=0),
                     colSums((round(as.matrix(glm.pca.pr))-round(target))!=0),
                     colSums((round(as.matrix(glm.mix.pr))-round(target))!=0)),
             "svm"=c(colSums((round(as.matrix(svm.unt.pr))-round(target))!=0),
                     colSums((round(as.matrix(svm.std.pr))-round(target))!=0),
                     colSums((round(as.matrix(svm.nrm.pr))-round(target))!=0),
                     colSums((round(as.matrix(svm.pca.pr))-round(target))!=0),
                     colSums((round(as.matrix(svm.mix.pr))-round(target))!=0)),
              "nn"=c(colSums((round(as.matrix(nn.unt.pr))-round(target))!=0),
                     colSums((round(as.matrix(nn.std.pr))-round(target))!=0),
                     colSums((round(as.matrix(nn.nrm.pr))-round(target))!=0),
                     colSums((round(as.matrix(nn.pca.pr))-round(target))!=0),
                     colSums((round(as.matrix(nn.mix.pr))-round(target))!=0)),
       "benchmark"=c(colSums((round(Fset[-train_ind,][-1,"target"])-round(target))!=0), 
                     NA,NA,NA,NA))

detach(pr)

print(MSEscores)
print(MisClas)
print(round(MisClas*(1/nrow(data_agg)),2))

beep()

##################################################
## Evaluation
##################################################
