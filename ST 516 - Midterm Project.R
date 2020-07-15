## Submission by Chaitanya Rajeev, Rehan Sheikh, Vishakha Patil and Rucha Girgaonkar

library(boot)
require(ISLR)
require(glmnet)
require(faraway)
require(dplyr)
require(leaps)
require(MASS)
require(DAAG)
#require(plotmo)

bikes= read.csv("bikes.csv")
bikes_casual = bikes[,c(-11,-1)]
bikes_reg = bikes[,c(-10,-1)]
###############################################################################################################
# EDA on the dataset
# plotting variation in casual and registered ridership with temp and weathersit
par(mfrow=c(2,2))
plot(bikes$temp,bikes$casual,pch=19,main="Temperature vs. Casual Riders")
abline(lm(casual~temp,bikes_casual),col='red',lwd=2)

plot(bikes$temp,bikes$registered,pch=19,main="Temperature vs. Registered Riders")
abline(lm(registered~temp,bikes_reg),col='red',lwd=2)

plot(bikes$weathersit,bikes$casual,pch=19,type="p",main="Seasons vs. Casual Riders",xlab="1-Clear, 2-Clouds/Mist, 3-Light Snow/Rain")
abline(lm(casual~weathersit,bikes_casual),col='red',lwd=2)

plot(bikes$weathersit,bikes$registered,pch=19,main="Seasons vs. Registered Riders",xlab="1-Clear, 2-Clouds/Mist, 3-Light Snow/Rain")
abline(lm(registered~weathersit,bikes_reg),col='red',lwd=2)

# plotting variation in casual and registered ridership with humidity(hum), Working day and Year
par(mfrow=c(3,2))
plot(bikes$workingday,bikes$casual,pch=19,main="Working Day vs. Casual Riders",xlab="1-Working Day, 0-Weekend/Holiday")
abline(lm(casual~workingday,bikes_casual),col='red',lwd=2)

plot(bikes$workingday,bikes$registered,pch=19,main="Working Day vs. Registered Riders",xlab="1-Working Day, 0-Weekend/Holiday")
abline(lm(registered~workingday,bikes_reg),col='red',lwd=2)

plot(bikes$hum,bikes$casual,pch=19,type="p",main="Humidity vs. Casual Riders")
abline(lm(casual~hum,bikes_casual),col='red',lwd=2)

plot(bikes$hum,bikes$registered,pch=19,main="Humidity vs. Registered Riders")
abline(lm(registered~hum,bikes_reg),col='red',lwd=2)

plot(bikes$yr,bikes$casual,pch=19,type="p",main="Year vs. Casual Riders",xlab="0-2011, 1-2012")
abline(lm(casual~yr,bikes_casual),col='red',lwd=2)

plot(bikes$yr,bikes$registered,pch=19,main="Year vs. Registered Riders",xlab="0-2011, 1-2012")
abline(lm(registered~yr,bikes_reg),col='red',lwd=2)


########################################## Data Preparation ######################################################
# Relations among given predictors
bikes$dteday= as.Date(bikes$dteday,format='%m/%d/%Y')
bikes$season=as.factor(bikes$season)
bikes$yr=as.factor(bikes$yr)
bikes$weathersit=as.factor(bikes$weathersit)
bikes$workingday=as.factor(bikes$workingday)

# removing serial no. column
bikes_casual = bikes[,c(-11,-1)]
bikes_reg = bikes[,c(-10,-1)]

# pairplot of bikes.csv
pairs(bikes)


x_casual = model.matrix(casual~.,bikes_casual)[,-1]
data_casual = data.frame(x_casual,bikes$casual)

x_reg = model.matrix(registered~.,bikes_reg)[,-1]
data_reg = data.frame(x_reg,bikes$registered)

####################### Fit 1 - simple additive linear models #################################################################

set.seed(1000)

# removing outliers from a previous analysis
data_casual2 = data_casual[c(-442,-463,-478),]
data_reg2 = data_reg[c(-668,-669,-724),]

# fitting model for casual and registered riders
fit1_casual = glm(bikes.casual~.,data_casual2,family=gaussian)
fit1_reg = glm(bikes.registered~.,data_reg2,family=gaussian)

#fitting CV model with 10 folds
fit1_casualcv = cv.glm(data_casual2,fit1_casual,K=104)
fit1_regcv = cv.glm(data_reg2,fit1_reg,K=104)

# errors for casual rider model
Train_MSE_fit1_casual = mean((predict(fit1_casual,data.frame(data_casual2))-data_casual2$bikes.casual)^2)
Test_MSE_fit1_casual = fit1_casualcv$delta[1]

# errors for registered rider model
Train_MSE_fit1_reg = mean((predict(fit1_reg,data.frame(data_reg2))-data_reg2$bikes.registered)^2)
Test_MSE_fit1_reg = fit1_regcv$delta[1]

# plotting residuals for casual model and registered model
par(mfrow=c(2,2))
plot(fit1_casual,1:2,pch=19)
plot(fit1_reg,1:2,pch=19)

summary(fit1_casual)
summary(fit1_reg)

# outlier plots 
par(mfrow=c(1,2))
halfnorm(cooks.distance(fit1_casual),3)
halfnorm(cooks.distance(fit1_reg),3)

####################### Fit 2 - Second-order models #######################################################################

set.seed(1000)
#generating second order predictors for casual riders model matrix
data_poly_casual = model.matrix(bikes.casual~.,data_casual)[,-1]
data_poly_casual = data.frame(poly(data_poly_casual,2,raw=TRUE),bikes$casual)
#removing redundant predictors to prevent singularities
data_poly_casual = data_poly_casual[,-match(c("X0.2.0.0.0.0.0.0.0.0.0","X0.1.1.0.0.0.0.0.0.0.0","X0.0.2.0.0.0.0.0.0.0.0","X0.1.0.1.0.0.0.0.0.0.0","X0.0.1.1.0.0.0.0.0.0.0","X0.0.0.2.0.0.0.0.0.0.0","X0.0.0.0.2.0.0.0.0.0.0","X0.0.0.0.0.2.0.0.0.0.0","X0.0.0.0.0.0.2.0.0.0.0","X0.0.0.0.0.0.1.1.0.0.0","X0.0.0.0.0.0.0.2.0.0.0"),names(data_poly_casual))]

#generating second order predictors for registered riders model matrix
data_poly_reg = model.matrix(bikes.registered~.,data_reg)[,-1]
data_poly_reg = data.frame(poly(data_poly_reg,2,raw=TRUE),bikes$registered)
#removing redundant predictors to prevent singularities
data_poly_reg = data_poly_reg[,-match(c("X0.2.0.0.0.0.0.0.0.0.0","X0.1.1.0.0.0.0.0.0.0.0","X0.0.2.0.0.0.0.0.0.0.0","X0.1.0.1.0.0.0.0.0.0.0","X0.0.1.1.0.0.0.0.0.0.0","X0.0.0.2.0.0.0.0.0.0.0","X0.0.0.0.2.0.0.0.0.0.0","X0.0.0.0.0.2.0.0.0.0.0","X0.0.0.0.0.0.2.0.0.0.0","X0.0.0.0.0.0.1.1.0.0.0","X0.0.0.0.0.0.0.2.0.0.0"),names(data_poly_reg))]

#centering predictors of both model matrixes
for (i in 1:(dim(data_poly_casual)[2]-1)){
  data_poly_casual[,i]=(data_poly_casual[,i]-mean(data_poly_casual[,i]))
  data_poly_reg[,i]=(data_poly_reg[,i]-mean(data_poly_reg[,i]))
}

# Outlier removal from a previous analysis
data_poly_casual5 = data_poly_casual[c(-90,-442,-726),]
data_poly_reg5 = data_poly_reg[c(-69,-668,-726),]

#correlation analysis
cor(data_poly_casual5)
cor(data_poly_reg5)

# fitting models for casual and registered riders
fit2_casual = glm(bikes.casual~.,data_poly_casual5,family = gaussian)
fit2_reg = glm(bikes.registered~.,data_poly_reg5,family = gaussian)

# CV models
fit2_casualcv = cv.glm(data_poly_casual5,fit2_casual,K=104)
fit2_regcv = cv.glm(data_poly_reg5,fit2_reg,K=104)

# Errors for casual model
Train_MSE_fit2_casual = mean((predict(fit2_casual,data.frame(data_poly_casual5))-data_poly_casual5$bikes.casual)^2)
Test_MSE_fit2_casual = fit2_casualcv$delta[1]

# Errors for registered model
Train_MSE_fit2_reg = mean((predict(fit2_reg,data.frame(data_poly_reg5))-data_poly_reg5$bikes.registered)^2)
Test_MSE_fit2_reg = fit2_regcv$delta[1]

# Residual plots for casual and registered model
par(mfrow=c(2,2))
plot(fit2_casual,1:2,pch=19)
plot(fit2_reg,1:2,pch=19)

summary(fit2_casual)
summary(fit2_reg)

# outlier plots 
par(mfrow=c(1,2))
halfnorm(cooks.distance(fit2_casual),3)
halfnorm(cooks.distance(fit2_reg),3)

####################### Fit 3 - Lasso on Second order models #################################################################

set.seed(1000)
# removing the outliers from a previous analysis
data_poly_casual3 = data_poly_casual[c(-442,-448,-472),]
data_poly_reg3 = data_poly_reg[c(-669,-692,-693),]


# developing model matrixes without the response column
x_casual = model.matrix(bikes.casual~.,data_poly_casual3)[,-1]
x_reg = model.matrix(bikes.registered~.,data_poly_reg3)[,-1]

# creating a grid of lambda values
grid=10^seq(2,-6,length=100) # lambda ranges from 100 to 0.000001 

# fitting models for casual and registered riders
fit3_casual = glmnet(x_casual,data_poly_casual3$bikes.casual,alpha=1,lambda=grid)
fit3_reg = glmnet(x_reg,data_poly_reg3$bikes.registered,alpha=1,lambda=grid)

# CV models
fit3_casualcv = cv.glmnet(x_casual,data_poly_casual3$bikes.casual,alpha=1,lambda=grid)
fit3_regcv = cv.glmnet(x_reg,data_poly_reg3$bikes.registered,alpha=1,lambda=grid)

# finding best lambda for both models
lambda_fit3_casual=fit3_casualcv$lambda.min
lambda_fit3_reg=fit3_regcv$lambda.min

# predicting values for both models
fit3_casual_pred = predict(fit3_casual,x_casual)
fit3_reg_pred = predict(fit3_reg,x_reg)

#plotting coefficient shrinking plots
par(mfrow=c(1,2))
plot(fit3_casual)
plot(fit3_reg)


# Errors for casual riders model
Test_MSE_fit3_casual=min(fit3_casualcv$cvm)## test mse obtained using lasso regression
Train_MSE_fit3_casual = mean((fit3_casual_pred - data_poly_casual3$bikes.casual)^2)

# Errors for registered riders model
Test_MSE_fit3_reg=min(fit3_regcv$cvm)## test mse obtained using lasso regression
Train_MSE_fit3_reg = mean((fit3_reg_pred - data_poly_reg3$bikes.registered)^2)

#below code requires the plotmo package. We used it for residual diagnostics and outlier detection
#plotres(fit3_casual)

# getting no. of non-zero predictors after LASSO
lasso.coef=predict(fit3_casual,type="coefficients",s=lambda_fit3_casual)
lasso.coef2=predict(fit3_reg,type="coefficients",s=lambda_fit3_reg)
lasso.coef
lasso.coef2
sum(abs(matrix(lasso.coef))>0)
sum(abs(matrix(lasso.coef2))>0)



###################### Fit 4 - Ridge on second order Models ################################################################

set.seed(1000)
# removing the outliers from a previous analysis
data_poly_casual3 = data_poly_casual[c(-442,-448,-472),]
data_poly_reg3 = data_poly_reg[c(-669,-692,-693),]

# developing model matrixes without the response column
x_casual = model.matrix(bikes.casual~.,data_poly_casual3)[,-1]
x_reg = model.matrix(bikes.registered~.,data_poly_reg3)[,-1]

# creating a grid of lambda values
grid=10^seq(2,-6,length=100) # lambda ranges from 100 to 0.000001 

# fitting models for casual and registered riders
fit4_casual = glmnet(x_casual,data_poly_casual3$bikes.casual,alpha=0,lambda=grid)
fit4_reg = glmnet(x_reg,data_poly_reg3$bikes.registered,alpha=0,lambda=grid)

# CV models
fit4_casualcv = cv.glmnet(x_casual,data_poly_casual3$bikes.casual,alpha=0,lambda=grid)
fit4_regcv = cv.glmnet(x_reg,data_poly_reg3$bikes.registered,alpha=0,lambda=grid)

# finding best lambda for both models
lambda_fit4_casual=fit4_casualcv$lambda.min
lambda_fit4_reg=fit4_regcv$lambda.min

# predicting values for both models
fit4_casual_pred = predict(fit4_casual,x_casual)
fit4_reg_pred = predict(fit4_reg,x_reg)

#plotting coefficient shrinking plots
par(mfrow=c(1,2))
plot(fit4_casual)
plot(fit4_reg)

# Errors for casual riders model
Test_MSE_fit4_casual=min(fit4_casualcv$cvm)## test mse obtained using ridge regression
Train_MSE_fit4_casual = mean((fit4_casual_pred - data_poly_casual3$bikes.casual)^2)

# Errors for registered riders model
Test_MSE_fit4_reg=min(fit4_regcv$cvm)## test mse obtained using ridge regression
Train_MSE_fit4_reg = mean((fit4_reg_pred - data_poly_reg3$bikes.registered)^2)

#below code requires the plotmo package. We used it for outlier detection
#plotres(fit4_reg)

####################### Fit 5 - Best subset selection on Linear models #####################################################

############### First order casual riders model##########################
best.mods=regsubsets(bikes.casual~.,data=data_casual,nvmax=11, method="exhaustive")
best.sum=summary(best.mods)
best.sum
names(best.sum)
summary(best.mods)


# Use 5-fold CV to determine which best model for each number of predictors
# has the lowest estimated test error
# create function to predict from reg subsets object

pred.sbs=function(obj,new,id,...){
  form=as.formula(obj$call[[2]])
  mat=model.matrix(form,new)
  coefi=coef(obj,id=id)
  xvars=names(coefi)
  mat[,xvars]%*%coefi
}
# set up for cross validation

k=5  # set number of folds
set.seed(1000)
# create an index with id 1-5 to assign observations to folds
folds=sample(1:5,nrow(data_casual),replace=T) 
# create dummy matrix to store CV error estimates
cv.err=matrix(NA,k,11,dimnames=list(NULL,paste(1:11)))

# perform CV
for (j in 1:k){
  # pick models with lowest RSS with 1-9 predictors fit without kth fold
  best.mods=regsubsets(bikes.casual~.,data=data_casual[folds!=j,],
                       nvmax=11,method="exhaustive")
  # estimate test error for all nine models by predicting kth fold 
  for (i in 1:11){
    pred=pred.sbs(best.mods,data_casual[folds==j,],id=i)
    cv.err[j,i]=mean((data_casual$bikes.casual[folds==j]-pred)^2)  # save error est
  }
}

mse.cv=apply(cv.err,2,mean) # cdompute mean MSE for each number of predictors
Test_MSE_fit5_casual = min(mse.cv)  # find minimum mean MSE
Train_MSE_fit5_casual = mean((pred.sbs(best.mods,data_casual,11) - bikes$casual)^2)

################# First order registered riders model###############################
best.mods=regsubsets(bikes.registered~.,data=data_reg,nvmax=11, method="exhaustive")
best.sum=summary(best.mods)
best.sum
names(best.sum)
summary(best.mods)


# Use 5-fold CV to determine which best model for each number of predictors
# has the lowest estimated test error
# create function to predict from reg subsets object

pred.sbs=function(obj,new,id,...){
  form=as.formula(obj$call[[2]])
  mat=model.matrix(form,new)
  coefi=coef(obj,id=id)
  xvars=names(coefi)
  mat[,xvars]%*%coefi
}
# set up for cross validation

k=5  # set number of folds
set.seed(1000)
# create an index with id 1-5 to assign observations to folds
folds=sample(1:5,nrow(data_reg),replace=T) 
# create dummy matrix to store CV error estimates
cv.err=matrix(NA,k,11,dimnames=list(NULL,paste(1:11)))

# perform CV
for (j in 1:k){
  # pick models with lowest RSS with 1-9 predictors fit without kth fold
  best.mods=regsubsets(bikes.registered~.,data=data_reg[folds!=j,],
                       nvmax=11,method="exhaustive")
  # estimate test error for all nine models by predicting kth fold 
  for (i in 1:11){
    pred=pred.sbs(best.mods,data_reg[folds==j,],id=i)
    cv.err[j,i]=mean((data_reg$bikes.registered[folds==j]-pred)^2)  # save error est
  }
}

mse.cv=apply(cv.err,2,mean) # cdompute mean MSE for each number of predictors
Test_MSE_fit5_reg=min(mse.cv)  # find minimum mean MSE
Train_MSE_fit5_reg = mean((pred.sbs(best.mods,data_reg,11) - bikes$registered)^2)

###################### Fit 6 -  Log transform on second order model ##############################################################

set.seed(1000)

# removing outliers from a previous analysis
data_poly_casual2 = data_poly_casual[c(-69,-668,-90),]# unreasonable outliers
data_poly_reg2 = data_poly_reg[c(-69,-668,-726),]# unreasonable outliers

# fitting models for casual and registered riders
fit6_casual = lm(log(bikes.casual)~.,data_poly_casual2)
fit6_reg = lm(log(bikes.registered)~.,data_poly_reg2)

# fitting CV models
fit6_casualcv = cv.lm(data_poly_casual2,fit6_casual,m=104)
fit6_regcv = cv.lm(data_poly_reg2,fit6_reg,m=104)

# Errors for casual riders model
Train_MSE_fit6_casual = mean((fit6_casualcv$bikes.casual-exp(fit6_casualcv$Predicted))^2)
Test_MSE_fit6_casual = mean((fit6_casualcv$bikes.casual-exp(fit6_casualcv$cvpred))^2)

# Errors for registered riders model
Train_MSE_fit6_reg = mean((fit6_regcv$bikes.registered-exp(fit6_regcv$Predicted))^2)
Test_MSE_fit6_reg = mean((fit6_regcv$bikes.registered-exp(fit6_regcv$cvpred))^2)

# Residual diagnostics plots
par(mfrow=c(2,2))
plot(fit6_casual,1:2,pch=19)
plot(fit6_reg,1:2,pch=19)

summary(fit6_casual)
summary(fit6_reg)
# outlier detection plots
par(mfrow=c(1,2))
halfnorm(cooks.distance(fit6_casual),3)
halfnorm(cooks.distance(fit6_reg),3)

##################### Fit 7 - Principal Components Regression on second order model ###########################################################################

set.seed(1000)

data_poly_casual4 = data_poly_casual[c(-90,-442,-726),]# unreasonable outliers
data_poly_reg4 = data_poly_reg[c(-69,-668,-726),]# unreasonable outliers

# developing model matrixes without the response column
x_casual = model.matrix(bikes.casual~.,data_poly_casual4)[,-1]
x_reg = model.matrix(bikes.registered~.,data_poly_reg4)[,-1]

# Getting principal components of both matrices
pca_bikes_casual=prcomp(x_casual,scale=T)
pca_bikes_reg=prcomp(x_reg,scale=T)

# making a data frame with response
bikes_pcr_casual = data.frame(pca_bikes_casual$x,data_poly_casual4$bikes.casual)
bikes_pcr_reg = data.frame(pca_bikes_reg$x,data_poly_reg4$bikes.registered)

# fitting casual and registered rider models
fit7_casual = glm(data_poly_casual4.bikes.casual~.,bikes_pcr_casual,family = gaussian)
fit7_reg = glm(data_poly_reg4.bikes.registered~.,bikes_pcr_reg,family = gaussian)

# fitting CV models
fit7_casualcv = cv.glm(bikes_pcr_casual,fit7_casual,K=104)
fit7_regcv = cv.glm(bikes_pcr_reg,fit7_reg,K=104)

# Getting Errors for the casual rider model
Train_MSE_fit7_casual = mean((predict(fit7_casual,data.frame(bikes_pcr_casual))-data_poly_casual4$bikes.casual)^2)
Test_MSE_fit7_casual = fit7_casualcv$delta[1]

# Getting Errors for the registered rider model
Train_MSE_fit7_reg = mean((predict(fit7_reg,data.frame(bikes_pcr_reg))-data_poly_reg4$bikes.registered)^2)
Test_MSE_fit7_reg = fit7_regcv$delta[1]

summary(fit7_casual)
summary(fit7_reg)

# plotting residuals for both models
par(mfrow=c(2,2))
plot(fit7_casual,1:2,pch=19)
plot(fit7_reg,1:2,pch=19)

# outlier detection
par(mfrow=c(1,2))
halfnorm(cooks.distance(fit7_casual),3)
halfnorm(cooks.distance(fit7_reg),3)

############################################## Final Error Plotting ###########################################################
#making vectors containing errors
Test_MSE_casual = c(Test_MSE_fit1_casual,Test_MSE_fit2_casual,Test_MSE_fit3_casual,Test_MSE_fit4_casual,Test_MSE_fit5_casual,Test_MSE_fit6_casual,Test_MSE_fit7_casual)
Train_MSE_casual = c(Train_MSE_fit1_casual,Train_MSE_fit2_casual,Train_MSE_fit3_casual,Train_MSE_fit4_casual,Train_MSE_fit5_casual,Train_MSE_fit6_casual,Train_MSE_fit7_casual)
Test_MSE_reg = c(Test_MSE_fit1_reg,Test_MSE_fit2_reg,Test_MSE_fit3_reg,Test_MSE_fit4_reg,Test_MSE_fit5_reg,Test_MSE_fit6_reg,Test_MSE_fit7_reg)
Train_MSE_reg = c(Train_MSE_fit1_reg,Train_MSE_fit2_reg,Train_MSE_fit3_reg,Train_MSE_fit4_reg,Train_MSE_fit5_reg,Train_MSE_fit6_reg,Train_MSE_fit7_reg)

# plotting errors
par(mfrow=c(1,1))
plot(c(1,2,3,4,5,6,7),Test_MSE_casual,type="l",ylim=c(50000,450000),xlab="1-1st order, 2-2nd order, 3-LASSO 2nd order, 4-Ridge 2nd order, 5-Subset 1st order, 6-Log transform 2nd order, 7-PCR",ylab="MSE",main="Model Error Plots")
points(c(1,2,3,4,5,6,7),Test_MSE_casual,pch=19)

lines(c(1,2,3,4,5,6,7),Train_MSE_casual,col="blue")
points(c(1,2,3,4,5,6,7),Train_MSE_casual,pch=19,col="blue")

lines(c(1,2,3,4,5,6,7),Test_MSE_reg,col="green")
points(c(1,2,3,4,5,6,7),Test_MSE_reg,pch=19,col="green")

lines(c(1,2,3,4,5,6,7),Train_MSE_reg,col="purple")
points(c(1,2,3,4,5,6,7),Train_MSE_reg,pch=19,col="purple")

legend("topright",legend=c("Test MSE - registered","Train MSE - registered","Test MSE - casual","Train MSE - casual"),
       lwd=c(1,1),lty=c(1,1),col=c("green","purple","black","blue"))

# marking the models with the least test errors with a red circle
points(which.min(Test_MSE_casual),Test_MSE_casual[which.min(Test_MSE_casual)],cex=2,col="red",lwd=2)
points(which.min(Test_MSE_reg),Test_MSE_reg[which.min(Test_MSE_reg)],cex=2,col="red",lwd=2)

############################################ Best Models ################################################################

##Based on the test errors calculated Fit 3 is chosen as the best model(LASSO on 2nd order model). It has the lowest test errors and good interpretability
fit3_casual.coef=predict(fit3_casual,type="coefficients",s=lambda_fit3_casual)
fit3_casual.coef
fit3_reg.coef=predict(fit3_reg,type="coefficients",s=lambda_fit3_reg)
fit3_reg.coef
Test_MSE_fit3_casual
Test_MSE_fit3_reg


