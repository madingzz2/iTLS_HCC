#--- z-score----
rm(list = ls())
library(caret)
library(dplyr)

train <- read.csv('resnet50/xyh_DL_features.csv')
val1 <- read.csv('resnet50/hainan_DL_features.csv')
val2 <- read.csv('resnet50/mhh_DL_features.csv')
val3 <- read.csv('resnet50/tcia_DL_features.csv')

normal_para1 <- preProcess(x = train,method = c("center","scale"))
normal_para2 <- preProcess(x = val1,method = c("center","scale"))
normal_para3 <- preProcess(x = val2,method = c("center","scale"))
normal_para4 <- preProcess(x = val3,method = c("center","scale"))

df_center1 <- predict(object = normal_para1,newdata = train)
df_center2 <- predict(object = normal_para2,newdata = val1)
df_center3 <- predict(object = normal_para3,newdata = val2)
df_center4 <- predict(object = normal_para4,newdata = val3)

write.csv(df_center1, row.names = FALSE,file="resnet50/xyyy_DL_features_z.csv")
write.csv(df_center2, row.names = FALSE,file="resnet50/hainan_DL_features_z.csv")
write.csv(df_center3, row.names = FALSE,file="resnet50/mhh_DL_features_z.csv")
write.csv(df_center4, row.names = FALSE,file="resnet50/tcia_DL_features_z.csv")

#--- model building ----
rm(list=ls())
library(caret)
library(glmnet)
library(rms)
library(dplyr)
library(mRMRe)
library(tidyverse)
library(data.table)
library(tibble)
set.seed(169) 
Clin <- read.csv('Clinicals.csv')
df <- read.csv("resnet50/xyyy_DL_features_z.csv")
df1 <- cbind(
  read.csv("resnet50/hainan_DL_features_z.csv"),
  read.csv("resnet50/mhh_DL_features_z.csv"))
df2 <- read.csv("resnet50/tcia_DL_features_z.csv")


df1$Label <- Clin$iTLS[match(df1$ID,Clin$ID)]
df2$Label <- Clin$iTLS[match(df2$ID,Clin$ID)]


train_Label <- Clin[match(df$ID,Clin$ID),] %>% 
  select(ID,label)
index <- createDataPartition(train_Label$label,p = 0.8) #8:2
saveRDS(index,'Divided.rds')
training_set <- df[index$Resample1,]
validation_set <- df[-index$Resample1,]
training_set$Label <- train_Label$label[index$Resample1]
validation_set$Label <- train_Label$label[-index$Resample1]
prop.table(table(training_set$Label))
prop.table(table(validation_set$Label))

train_label <- training_set$Label
valid_label <- validation_set$Label
test1_label <- df1$Label
test2_label <- df2$Label
rownames(training_set) <- NULL
train_set_nolabel <- training_set %>% 
  column_to_rownames('ID') %>% 
  select(-Label)
dim(training_set)
dim(validation_set)
## Feature Selection ####
if(T){
####  mRMR ####
mrmr_feature<-train_set_nolabel 
mrmr_feature$y<-train_label
target_indices = which(names(mrmr_feature)=='y')

for (m in which(sapply(mrmr_feature, class)!="numeric")){
  mrmr_feature[,m]=as.numeric(unlist(mrmr_feature[,m]))
}

data4 <- mRMR.data(data = data.frame(mrmr_feature))
mrmr=mRMR.ensemble(data = data4, target_indices = target_indices,
                   feature_count = 20, solution_count = 1)
index=mrmr@filters[[as.character(mrmr@target_indices)]]
index=as.numeric(index)
data_reduce = train_set_nolabel[,index]
#### LASSO ####
cv_x <- as.matrix(data_reduce)
cv_y <- train_label
lasso_selection <- cv.glmnet(x=cv_x,
                             y=cv_y,
                             family = "binomial",
                             type.measure = "deviance",
                             alpha = 1,
                             nfolds = 5)
pdf(file='TL_lasso1.pdf',width = 6,height = 10)
par(font.lab = 2, mfrow = c(2,1), mar = c(4.5,5,3,2))
plot(x = lasso_selection, las = 1, xlab = "Log(lambda)")
dev.off()
nocv_lasso <- glmnet(x = cv_x, y = cv_y, family = "binomial",alpha = 1)
pdf(file='TL_lasso2.pdf',width = 6,height = 10)
plot(nocv_lasso,xvar = "lambda",las=1,lwd=2,xlab="Log(lambda)")
abline(v = log(lasso_selection$lambda.min),lwd=1,lty=3,col="black")
dev.off()
coefPara <- coef(object = lasso_selection,s="lambda.min")
lasso_values <- as.data.frame(which(coefPara != 0, arr.ind = T))

lasso_names <- rownames(lasso_values)[-1]
lasso_coef <- data.frame(Feature = rownames(lasso_values),
                         Coef = coefPara[which(coefPara !=0,arr.ind = T)])
lasso_coef
lasso_coef_len <- length(lasso_coef[,1])
lasso_coef_save <- lasso_coef[,1][2:lasso_coef_len]
lasso_coef_save
train_set_lasso <- data.frame(cv_x)[lasso_names]
valid_set_lasso <- validation_set[names(train_set_lasso)]
test1_set_lasso <- df1[names((train_set_lasso))]
test2_set_lasso <- df2[names((train_set_lasso))]

test1_all = as.matrix(test1_set_lasso)
test2_all = as.matrix(test2_set_lasso)

Data_all = as.matrix(rbind(train_set_lasso,valid_set_lasso))
xn = nrow(Data_all)
yn = ncol(Data_all)
xn1 = nrow(test1_set_lasso)
yn1 = ncol(test1_set_lasso)
xn2 = nrow(test2_set_lasso)
yn2 = ncol(test2_set_lasso)

beta = as.matrix(coefPara[which(coefPara !=0),])
betai_Matrix = as.matrix(beta[-1])
beta0_Matrix = matrix(beta[1],xn,1)
Radcore_Matrix = Data_all %*% betai_Matrix +beta0_Matrix
radscore_all = as.numeric(Radcore_Matrix)

beta0_Matrix2 = matrix(beta[1],xn2,1)
Radcore_Matrix2 = test2_all %*% betai_Matrix +beta0_Matrix2
Radscore_test2 = as.numeric(Radcore_Matrix2)

beta0_Matrix1 = matrix(beta[1],xn1,1)
Radcore_Matrix1 = test1_all %*% betai_Matrix +beta0_Matrix1
Radscore_test1 = as.numeric(Radcore_Matrix1)

Radscore_train = radscore_all[1:nrow(train_set_lasso)]
Radscore_valid = radscore_all[(nrow(train_set_lasso)+1):xn]

Radscore_train_matrix <- matrix(Radscore_train,ncol = 1)
predata_train_matrix <- data.frame(ID=training_set$ID,
                                   Label=training_set$Label)
lasso_coef_train_matrix <- select(training_set,lasso_coef_save)
radscore_train_data1 <- cbind(predata_train_matrix,Radscore_train_matrix)
radscore_train_data2 <- cbind(predata_train_matrix,lasso_coef_train_matrix)
write.csv(radscore_train_data2,file = "select feature/train DL features.csv",row.names = FALSE)

Radscore_valid_matrix <- matrix(Radscore_valid,ncol = 1)
predata_valid_matrix <- data.frame(ID=validation_set$ID,
                                   Label=validation_set$Label)
lasso_coef_valid_matrix <- select(validation_set,lasso_coef_save)
radscore_valid_data1 <- cbind(predata_valid_matrix,Radscore_valid_matrix)
radscore_valid_data2 <- cbind(predata_valid_matrix,lasso_coef_valid_matrix)
write.csv(radscore_valid_data2,file = "select feature/test DL features.csv",row.names = FALSE)

Radscore_test1_matrix <- matrix(Radscore_test1,ncol = 1)
predata_test1_matrix <- data.frame(ID=df1$ID,
                                   Label=df1$Label)
lasso_coef_test1_matrix <- select(df1,lasso_coef_save)
radscore_test1_data1 <- cbind(predata_test1_matrix,Radscore_test1_matrix)
radscore_test1_data2 <- cbind(predata_test1_matrix,lasso_coef_test1_matrix)
write.csv(radscore_test1_data2,file = "select feature/external DL features.csv",
          row.names = FALSE)

Radscore_test2_matrix <- matrix(Radscore_test2,ncol = 1)
predata_test2_matrix <- data.frame(ID=df2$ID,
                                   Label=df2$Label)
lasso_coef_test2_matrix <- select(df2,lasso_coef_save)
radscore_test2_data1 <- cbind(predata_test2_matrix,Radscore_test2_matrix)
radscore_test2_data2 <- cbind(predata_test2_matrix,lasso_coef_test2_matrix)
write.csv(radscore_test2_data2,file = "select feature/TCIA DL features.csv",
          row.names = FALSE)
}
####  TL model save ####

  rm(list=ls())
  library(pROC)
  train <- read.csv("select feature/train DL features.csv",row.names = "ID")
  val <- read.csv("select feature/test DL features.csv",row.names = "ID")
  model <- glm(Label ~ ., data = train,family='binomial')
  summary(model)
  saveRDS(model,file = 'TL_model.rds')
  #fwrite(data.frame(model$coefficients),file = 'TL_coefficients.csv',row.names = T)
  train_predicted= predict(model,train,type = 'response')
  val_predicted= predict(model,val,type = 'response')
  trainroc = roc(train$Label,train_predicted)
  valroc = roc(val$Label,val_predicted)
  roc_result_train <- coords(trainroc, "best") #Optimal cutoff based on Youden Index
  




