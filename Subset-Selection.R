library(readr)
parkinsons_updrs_data <- read_csv("C:/knot4u/UTD/6301/Lecture Material/Datasets/Lecture-14-Data/parkinsons_updrs-data.csv")
View(parkinsons_updrs_data)
myData=parkinsons_updrs_data
summary(myData)
# We will first drop the subject ID, not needed
myData = myData[,-1]
# Note sex needs to be a factor
myData$sex = as.factor(myData$sex)
summary(myData)
# Let's try best model selection first ...
# The predictor motor_UPDRS seems to be another version of total_UPDRS,
# which is what we will try to model. So we drop the model_UPDRS.
# We need the package leaps for subset selection
library("leaps", lib.loc="~/R/win-library/3.3")
park.model.best = regsubsets(total_UPDRS~. -motor_UPDRS, myData)
summary(park.model.best)
# We need to specify that we want to consider all subsets - the
# default is size 8
park.model.best = regsubsets(total_UPDRS~. -motor_UPDRS, myData, nvmax=19)
park.best.summary = summary(park.model.best)
names(park.best.summary)
park.best.summary$rsq
park.best.summary$adjr2
plot(park.best.summary$adjr2,xlab="Number of Variables",ylab="Adjusted RSq",type="l")
plot(park.best.summary$cp,xlab="Number of Variables",ylab="Cp",type="l")
plot(park.model.best,scale="adjr2")
park.best.summary$cp
# The model with 15 predictors seems to be the best
coef(park.model.best,15)
# Now let's try a forward selection ...
park.model.fwd = regsubsets(total_UPDRS~. -motor_UPDRS, data=myData, nvmax=19, method="forward")
summary(park.model.fwd)
park.fwd.summary = summary(park.model.fwd)
park.fwd.summary$adjr2
park.fwd.summary$cp
coef(park.model.fwd,15)
coef(park.model.best,15)
# In this case, FWD gave the same coefficients
# Note regsubsets does not support CV - we will need to do this
# manually
# We will use a data set from the ISLR package
library("ISLR", lib.loc="~/R/win-library/3.3")
fix(Hitters)
# Fix gives a pop-up editor for editing data files (be careful with big ones)
names(Hitters)
dim(Hitters)
sum(is.na(Hitters$Salary))
# This tells us 59 records have NAs, so we are safe removing them
Hitters=na.omit(Hitters)
dim(Hitters)
sum(is.na(Hitters))
set.seed(1)
# We do a simple CV first
train=sample(c(TRUE,FALSE), nrow(Hitters),rep=TRUE)
test=(!train)
regfit.best=regsubsets(Salary~.,data=Hitters[train,],nvmax=19)
# Problem - the predict function does not work with resubsets ...
# We need to create our own predict function
# Here is the model matrix - it contains the equation fof the (full) model
test.mat=model.matrix(Salary~.,data=Hitters[test,])
val.errors=rep(NA,19)
for(i in 1:19){
coefi=coef(regfit.best,id=i)
pred=test.mat[,names(coefi)]%*%coefi
val.errors[i]=mean((Hitters$Salary[test]-pred)^2)
}
val.errors
# Now we can check which model gave the lowest MSE
which.min(val.errors)
coef(regfit.best,10)
# We may need this prediction function again - here is a general form
predict.regsubsets=function(object,newdata,id,...){
form=as.formula(object$call[[2]])
mat=model.matrix(form,newdata)
coefi=coef(object,id=id)
xvars=names(coefi)
mat[,xvars]%*%coefi
}
# We did the previous analysis for a simple CV - what about K-fold?
regfit.best=regsubsets(Salary~.,data=Hitters,nvmax=19)
k=10
set.seed(1)
folds=sample(1:k,nrow(Hitters),replace=TRUE)
cv.errors=matrix(NA,k,19, dimnames=list(NULL, paste(1:19)))
# This matrix will hold the MSE for each fold, for each model
for(j in 1:k){
best.fit=regsubsets(Salary~.,data=Hitters[folds!=j,],nvmax=19)
for(i in 1:19){
pred=predict(best.fit,Hitters[folds==j,],id=i)
cv.errors[j,i]=mean( (Hitters$Salary[folds==j]-pred)^2)
}
}
# Now we get the average over all the folds for each model size
mean.cv.errors=apply(cv.errors,2,mean)
mean.cv.errors
plot(mean.cv.errors,type='b')
# Looks like 11 variables is best ...
# We now build the model on all the full data, and select the one
# with 11 variables
reg.best=regsubsets(Salary~.,data=Hitters, nvmax=19)
coef(reg.best,11)
install.packages("glmnet")
library("glmnet", lib.loc="~/R/win-library/3.3")
# We need to feed glmnet the predictors and response variable separately
x=model.matrix(Salary~.,Hitters)[,-1]
y=Hitters$Salary
# We create a sequence of lambdas to test
grid=10^seq(10,-2,length=100)
# We now do Ridge reqression for each lambda
ridge.mod=glmnet(x,y,alpha=0,lambda=grid)
dim(coef(ridge.mod))
# Notice we set alpha = 0; this is the option for Ridge
# Also, the glmnet scales the data automatically
# The coef matrix has a row for each predictor (plus intercept)
# and col for each lambda value
names(ridge.mod)
# We can access the lambda values, but the coeff we need to access
# separately
ridge.mod$lambda [50]
coef(ridge.mod)[,50]
# The predict function allows us to calculate the coefficients
# for lambdas that were not in our original grid
# Here are the coefficients for lambda = 50 ...
predict(ridge.mod,s=50,type="coefficients")[1:20,]
# We would like to choose an optimal lambda - we'll demonstrate
# a simple CV method here (K-fold can be used with more work).
set.seed(1)
train=sample(1:nrow(x), nrow(x)/2)
test=(-train)
y.test=y[test]
ridge.mod=glmnet(x[train,],y[train],alpha=0,lambda=grid, thresh=1e-12)
# We need to find the optimal lambda - we can use CV
set.seed(1)
cv.out=cv.glmnet(x[train,],y[train],alpha=0)
# The cv function does 10-fold CV by default
plot(cv.out)
bestlam=cv.out$lambda.min
bestlam
ridge.pred=predict(ridge.mod,s=bestlam,newx=x[test,])
mean((ridge.pred-y.test)^2)
# ONce we do CV find the best lambda, we can use all of the data
# to build our model using this lambda ...
ridge.model=glmnet(x,y,alpha=0)
predict(ridge.model,type="coefficients",s=bestlam)[1:20,]
# Note all coef are included, but some a weighted heavier than others
# Now we apply the Lasso - note all we do is change the alpha option
lasso.mod =glmnet (x[train ,],y[train],alpha =1, lambda =grid)
plot(lasso.mod)
# Note that Lasso takes certain coeff to zero for large enough lambda
# We'll do CV to find the best lambda again ...
set.seed (1)
cv.out =cv.glmnet (x[train ,],y[train],alpha =1)
plot(cv.out)
bestlam =cv.out$lambda.min
lasso.pred=predict (lasso.mod,s=bestlam,newx=x[test,])
mean(( lasso.pred -y.test)^2)
# The MSE is similar to what we saw for Ridge
# Let's find the coefficients ...
out=glmnet (x,y,alpha =1, lambda =grid)
lasso.coef=predict(out,type ="coefficients",s=bestlam )[1:20,]
lasso.coef
lasso.coef[lasso.coef!=0]
