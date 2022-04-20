# 1. Load libraries  ---------------------------------------------------------
library(tidyverse)
library(Hmisc) # draw histogram frame
library(corrplot) # Draw corrplot
library(e1071) # Check skewness
library(caret) # Preprocessing
library(mice) # Missing value imputation
library(VIM) # Draw aggr_plot
library(gam)
library(mgcv) # GAM
library(car) # Check VIF
library(mlbench) # Check near-zero variance prediction
library(DHARMa) # Interpretable diagostic plots for logit-GAM
library(mgcViz)
library(ROCR) # Plot ROC curve


# 2. Import diabetes data --------------------------------------------------
data <- read.csv('D:/Yahui/Textbooks/STA6933_Adv Stat Mining/Project1/diabetes.csv',header=T)


# 3. Data Preprocessing -----------------------------------------

## 3.1 Data Quality Assessment ####

head(data)
str(data)
# colnames(data)[7] <- "DiabPedgFun"
### Mismatched data types ####
data$Outcome <- as.factor(data$Outcome)

summary(data)

### Outliers ----
par(mfrow=c(2,4)) 

for (i in 1:8) {
  boxplot(data[,i], main=colnames(data)[i],col=col[i])
}

# The box-plots suggest there exists outliers for all predictors

# pairwise scatterplots
panel.hist <- function(x, ...){
  #from help of pairs
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(usr[1:2], 0, 1.5) )
  h <- hist(x, plot = FALSE)
  breaks <- h$breaks; nB <- length(breaks)
  y <- h$counts; y <- y/max(y)
  rect(breaks[-nB], 0, breaks[-1], y,col = "#61a2bc")
}

panel.cor <- function(x, y, digits = 2, prefix = "", cex.cor, ...){
  #from help of pairs
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  r <- abs(cor(x, y,use="pairwise.complete.obs"))
  txt <- format(c(r, 0.123456789), digits = digits)[1]
  txt <- paste0(prefix, txt)
  if(missing(cex.cor)) cex.cor <- 0.8/strwidth(txt)
  text(0.5, 0.5, txt, cex = 1.25*cex.cor * r,col="#402a34")
}

# pairs(data[,-9],col="steelblue")

dev.off()
pairs(data[,-9], 
      lower.panel = panel.smooth,
      upper.panel = panel.cor,col=ifelse(data$Outcome==0, "#625a50", "firebrick"),
      diag.panel = panel.hist
)

### Impute missing values ----


###  Missing values ----
any(is.na(data))


zero.n <- sapply(data[,2:6], function(x) length(x[x ==0])/length(x))
col <- c("#61a2bc","#b93f1d",'#ab8fbe',"#d3a426","#88b998","#aaa197","#d67900","#cb7580")
barplot(zero.n, ylim=c(0,0.6), col=col[2:6])


# Frequency table of zero data for each variable
sapply(data, function(x) sum(x == 0))

# Frequency table of zero data for each variable is shown below. 
# It's observed that that predictors *Glucose*, *BloodPressure*, *SkinThickness*, *Insulin* and *Age* have invalid zero values, especially for *Insulin* and *SkinThickness*, 50% 30%. 

data[,2:6] <- apply(data[,2:6],c(1,2), function(x)ifelse(x==0,NA,x))

data %>% 
  group_by(Outcome) %>% 
  gather(variable, value, -Outcome) %>% 
  summarise(missing.n = sum(is.na(value))) %>%
  arrange(desc(missing.n))

# Frequency table of zero data for each variable
sapply(data, function(x) sum(is.na(x)))



# md.pattern(train[,-9])
dev.off()
aggr_plot <- aggr(data[,-9], col=c('steelblue','firebrick'), numbers=TRUE, sortVars=TRUE, 
                  labels=names(train), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))
marginplot(data[c(2,3)])
marginplot(data[c(2,4)])
marginplot(data[c(2,5)])
marginplot(data[c(2,6)])
marginplot(data[c(3,4)])
marginplot(data[c(3,5)])
marginplot(data[c(3,6)])
marginplot(data[c(4,5)])
marginplot(data[c(4,6)])
marginplot(data[c(5,6)])
# The zero values in *train* are not assigned at random, therefore are non-ignorable. 

# 'mice' package is used to impute missing values.
imputed_data <- mice(data, m=5, maxit=50, meth='pmm', seed=500)
summary(imputed_data)


# impute$imp$Glucose
completedData <- complete(imputed_data,1)

# Frequency table of missing data for each variable after imputation
# sapply(completedData, function(x)sum(is.na(x)))

xyplot(imputed_data, Glucose ~ BloodPressure + SkinThickness + Insulin + BMI, pch=18, cex=1)

densityplot(imputed_data)

stripplot(imputed_data, pch = 20, cex = 1.2)

## 3.2 Split the data ----------------------------------------------------------

set.seed(1)
split <- sample(c(rep(0, floor(0.8 * nrow(completedData))), rep(1, ceiling(0.2 * nrow(completedData)))))
train <- completedData[split == 0, ]
test <- completedData[split== 1, ]

## 3.3 Data Cleaning ####


## 3.4 Data transformation --------------------------------------------------------

### Data Reduction -----------------------------------------------------
# Find near zero variance predictors
names(train)[nearZeroVar(train)] 
# no zero variance predictor exists in the data.    



# ### Resolve skewness
# Calculate the skewness of of each predictor.
apply(train[,-9], 2, function(x)skewness(na.omit(x)))

# It is observed that predictors *Insulin*, *DiabetesPedigreeFunction*, and *Age* are highly skewed.

# Box-cox transformation is used to deal with skewness and therefore might improve the classification model.
BoxCoxTrans(na.omit(train$Insulin))
BoxCoxTrans(na.omit(train$DiabetesPedigreeFunction))
BoxCoxTrans(na.omit(train$Age))

# plot comparison histograms
par(mfrow=c(3,2))
hist(train$Insulin, prob=T,xlab = "Insulin", col= "#9a8f83",main="Insulin (Before)")
hist(log(train$Insulin), prob=T,xlab = "log(Insulin)", col= "#cac4be",main="Insulin (After)")
hist(train$DiabetesPedigreeFunction, prob=T, col = "#3f7d96", xlab = "DiabetesPedigreeFunction", main="DiabetesPedigreeFunction (Before)")
hist(log(train$DiabetesPedigreeFunction), prob=T, col = "#93bfd1", xlab = "log(DiabetesPedigreeFunction)", main="DiabetesPedigreeFunction (After)")
hist(train$Age, prob=T, xlab = "Age", col="#d74a21", main="Age (Before)")
hist(1/train$Age, prob=T, xlab = "1/Age", col="#ea937a", main="Age (After)")



par(mfrow=c(3,2))
hist(train$Insulin, prob=T,xlab = "Insulin", col= "#9a8f83",main="")
hist(log(train$Insulin), prob=T,xlab = "log(Insulin)", col= "#cac4be",main="")
hist(train$DiabetesPedigreeFunction, prob=T, col = "#3f7d96", xlab = "DiabetesPedigreeFunction", main="")
hist(log(train$DiabetesPedigreeFunction), prob=T, col = "#93bfd1", xlab = "log(DiabetesPedigreeFunction)", main="")
hist(train$Age, prob=T, xlab = "Age", col="#d74a21", main="")
hist(1/train$Age, prob=T, xlab = "1/Age", col="#ea937a", main="")



train.trans <- train
train.trans$Insulin <- log(train.trans$Insulin)
train.trans$DiabetesPedigreeFunction <- log(train.trans$DiabetesPedigreeFunction)
train.trans$Age <- 1/train.trans$Age
colnames(train.trans)[c(5,7,8)] <- c("log.Insulin","log.DiabetesPedigreeFunction","rev.Age")



### Resolve outliers ----

par(mfrow=c(2,4)) 
for (i in 1:8) {
  boxplot(train.trans[,i], main=colnames(train.trans)[i],col=col[i])
}

for (i in 1:8) {
  print(paste(i,': ',colnames(train.trans)[i]))
  print(boxplot.stats(train.trans[,i])$out)
}


for (i in c(1:8)) {
  x <- train.trans[,i]
  qnt <- quantile(x, probs=c(.25, .75), na.rm = T)
  caps <- quantile(x, probs=c(.05, .95), na.rm = T)
  H <- 1.5 * IQR(x, na.rm = T)
  x[x < (qnt[1] - H)] <- caps[1]
  x[x > (qnt[2] + H)] <- caps[2]
  train.trans[,i] <- x
}

par(mfrow=c(2,4)) 
for (i in 1:8) {
  boxplot(train.trans[,i], main=colnames(train.trans)[i],col=col[i])
}


# 6. Modeling --------------------------------------------------------
# Fit a logistic-GAM model-------

logitgam <- gam(Outcome ~ s(Pregnancies) + s(Glucose) + s(BloodPressure) +
                  s(SkinThickness) + s(log(Insulin)) + s(BMI) +
                  s(log(DiabetesPedigreeFunction)) + s(as.numeric(1/Age)),
                data = train, family = binomial)


logitgam3 <- gam(Outcome ~ s(Glucose) + s(BMI) +
                   s(log.DiabetesPedigreeFunction) + s(rev.Age),
                 data = train.trans, family = binomial)


summary(logitgam3)
par(mfrow = c(2,2))
plot(logitgam3,se=TRUE, lwd=1.5,col="steelblue")
gam.check(logitgam3)


# Fit a logistic regression model----
logit <- glm(Outcome ~ ., data=train.trans, family = binomial)
summary(logit)


# # Get diagnostic plots for GAM models using package 'mgcViz'
res <- simulateResiduals(logitgam3)
plot(res)


# Evaluate performance of the model --------

# confusion matrix
test.trans <- test
test.trans$Insulin <- log(test.trans$Insulin)
test.trans$DiabetesPedigreeFunction <- log(test.trans$DiabetesPedigreeFunction)
test.trans$Age <- 1/test.trans$Age
colnames(test.trans)[c(5,7,8)] <- c("log.Insulin","log.DiabetesPedigreeFunction","rev.Age")

model <- logit
model <- logitgam3
# training error rate
log.prob <- predict(model, train.trans[,-9],type = "response")
log.pred <- rep(0,length(log.prob))
log.pred[log.prob > 0.5] = 1

(train.err <- mean(train$Outcome != log.pred))

# test error rate
log.prob <- predict(model, test.trans[,-9],type = "response")
log.pred <- rep(0,length(log.prob))
log.pred[log.prob > 0.5] = 1

(test.err <- mean(test$Outcome != log.pred))

# define a function for visualization of confusion matrix
draw_confusion_matrix <- function(cm,mod) {
  
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title(paste0('Confusion Matrix for ',mod), cex.main=2)
  
  # create the matrix 
  rect(150, 430, 240, 370, col='#3F97D0')
  text(195, 435, 'Class1', cex=1.2)
  rect(250, 430, 340, 370, col='#F7AD50')
  text(295, 435, 'Class2', cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col='#F7AD50')
  rect(250, 305, 340, 365, col='#3F97D0')
  text(140, 400, 'Class1', cex=1.2, srt=90)
  text(140, 335, 'Class2', cex=1.2, srt=90)
  
  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[1]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  text(30, 85, names(cm$byClass[2]), cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)
  
  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
} 

draw_confusion_matrix2 <- function (y) {
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(y$byClass[1]), cex=1, font=1)
  text(10, 70, round(as.numeric(y$byClass[1]), 3), cex=1)
  text(30, 85, names(y$byClass[2]), cex=1, font=1)
  text(30, 70, round(as.numeric(y$byClass[2]), 3), cex=1)
  text(50, 85, names(y$byClass[5]), cex=1, font=1)
  text(50, 70, round(as.numeric(y$byClass[5]), 3), cex=1)
  text(70, 85, names(y$byClass[6]), cex=1, font=1)
  text(70, 70, round(as.numeric(y$byClass[6]), 3), cex=1)
  text(90, 85, names(y$byClass[7]), cex=1, font=1)
  text(90, 70, round(as.numeric(y$byClass[7]), 3), cex=1)
  
  # add in the accuracy information 
  text(30, 35, names(y$overall[1]), cex=1, font=1)
  text(30, 20, round(as.numeric(y$overall[1]), 3), cex=1)
  text(70, 35, names(y$overall[2]), cex=1, font=1)
  text(70, 20, round(as.numeric(y$overall[2]), 3), cex=1)
  return(invisible())
}

# create a confusion matrix for train data
train.cm.log <- confusionMatrix(as.factor(log.pred), train.trans$Outcome,
                                mode = "everything", positive = "1")
draw_confusion_matrix(train.cm.log,'Logistic-GAM')


# create a confusion matrix for test data
cm.log <- confusionMatrix(as.factor(log.pred), test.trans$Outcome,
                           mode = "everything", positive = "1")

# visualize CM
draw_confusion_matrix(cm.log,'Logistic-GAM')
dev.off()
par(mfrow=c(3,1))
fourfoldplot(cm.log$table,color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin = 1, main = "Confusion Matrix")
draw_confusion_matrix2(cm.log)

