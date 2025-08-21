rm(list = ls())
library(MASS)
library(dplyr)
library(rms)
library(autoReg)
load('DCA_input_1102.Rdata')
train <- data.frame(
  Rad=rms::predict(
  readRDS('Rad_model.rds'),
  read.csv("select feature/train Rad features.csv",row.names = "ID")),

  TL=rms::predict(
    readRDS('TL_model.rds'),
    read.csv("select feature/train DL features.csv",row.names = "ID"))) 
df <- read.csv('Clinicals.csv') 
df <- df[match(train$ID,df$ID),] %>% 
  cbind(train)

df1 <- df %>% 
  mutate(Age=case_when(Age>50~'>50',
                       Age<= 50~'<50'),
         Number=case_when(Number=='1'~'<2',
                          Number=='≥2'~'≥2'),
         BCLC=case_when(BCLC==1~'0-A',
                        BCLC==2~'B-C'),
         AFP=case_when(AFP> 400~'>400',
                       AFP<= 400~'<400'),
         ALT=case_when(ALT<= 50~'<50',
                        ALT>50~'>50'),
         AST=case_when(AST<= 33.2~'<33.2',
                        AST> 33.2~'>33.2'))
tmp <-   glm(iTLS ~ Age+Gender+HBV+BCLC+Cirrhosis+AFP+ALT+AST+PLT+
               Large+Number+NAPHE+Non_peripheral_washout+APHE+
               Enhancing_capsule+APE+hemorrhage+Necrosis+Rad+TL+
               Intra_arterial+Unsmooth_margin,data = df1,family=binomial)
autoReg(tmp,uni=T,threshold=0.1)

tmp1 <-   glm(label ~  ALT + hemorrhage + Necrosis + 
                Rad + TL,data = df1,family=binomial);tmp1
autoReg(MHH1,threshold=0.1)
back_features <- stepAIC(tmp1, direction = 'backward')
final_model <-   glm(label ~ 
                Rad + TL,data = df,family=binomial);MHH2
saveRDS(final_model,file = 'final_model.rds')
