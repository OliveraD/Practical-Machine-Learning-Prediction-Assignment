---
output: pdf_document
---
Practical Machine Learning Prediction Assignment
================================================

Executive Summary
-----------------

Using devices such as *Jawbone Up*, *Nike FuelBand*, and *Fitbit* we can
collect a large amount of data about personal activity with relatively
low cost. In this project we are using data from accelerometers on the
belt, forearm, arm, and dumbell of 6 participants which were asked to
perform barbell lifts correctly and incorrectly in 5 different ways. The
goal of the project is to predict the manner in which our participants
did the exercise.

Since we have many predictor variables we applied random forest model
which is often used for non linear modeling and for its speed, although
it can potentially lead to overfitting. Cleaning of the data included
checking for missing data and inspecting of the variables, after which
variables with missing data, index, respondent name, time stamp and
similar variables were removed. Final model accuracy over validation
dataset is very high - 99.39%, which resulted in excellent prediction
results with testing dataset generating all 20 accurate predictions
submitted in the Course Project Prediction Quiz.

Data preparation and cleaning
-----------------------------

The first step is to download the data and load them into data frames.
Data structure and summary, inspecting the class of the variables and
checking for the missing data followed. (Part of the code is included in
comment form due to the length of the assignment.)

    urlTrain <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    urlTest <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

    download.file(urlTrain, destfile="trainOriginal.csv")
    train <- read.csv("trainOriginal.csv", header=TRUE)

    download.file(urlTest, destfile="testOriginal.csv")
    test <- read.csv("testOriginal.csv", header=TRUE)

    #library(psych)

    #summary(train)
    #str(train)
    #describe(train)
    #head(train)

After initial inspecting "\#DIV/0!" values were added to missing values:

    train <- read.csv("trainOriginal.csv", header=TRUE, na.strings = c("NA","#DIV/0!"))

Since many variables had predominantly missing values they were
excluded. Indices variable, respondent name, time stamps and new and
numeric window variables were also excluded after initial checkup.

    trainSelected <- train[ , colSums(is.na(train)) == 0]
    #head(trainSelected)
    #table(trainSelected$new_window,trainSelected$classe)/length(trainSelected$classe)
    #table(trainSelected$num_window)
    #plot(trainSelected$classe,trainSelected$num_window)
    trainSelected <- trainSelected[, -c(1:7)]

Remaining variables were forced to numeric type, apart from the output
variable which remained a factor:

    numCol <- ncol(trainSelected)-1
    for(i in c(1:numCol)) {trainSelected[,i] = as.numeric(as.character(trainSelected[,i]))}
    #str(trainSelected)

Partitioning train data set
---------------------------

After loading *caret* and *randomForest* packages into current section,
*train* data set was partitioned into training and validation data sets
so we could perform model training and validation:

    library(caret)

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    library(randomForest)

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

    set.seed(1235)
    inTrain <- createDataPartition(trainSelected$classe, p=.7, list=FALSE)
    trainingData <- trainSelected[inTrain,]
    validationData <- trainSelected[-inTrain,]

Data preprocessing
------------------

Prediction variables were preprocessed by centering and scaling (data in
different variables were evidently in different scales):

    preProcValues <- preProcess(trainingData[,-c(53)], method = c("center", "scale"))

    trainingDataTransformed <- predict(preProcValues, trainingData)
    validationDataTransformed <- predict(preProcValues, validationData)

Modeling
--------

To speed up model training we parallelised the processing with the
*doParallel* package. Modeling was performed by defining *trainControl*
parameters including using cross validation, and specifying *random
forest* method. Training was done using all selected and preprocessed
variables.

    library(parallel)
    library(doParallel)

    ## Loading required package: foreach

    ## Loading required package: iterators

    #Set up parallel clusters
    Cl <- makeCluster(detectCores() - 1)
    registerDoParallel(Cl)

    set.seed(321)
    modelControl <- trainControl(method='cv', number=10, classProbs=TRUE, allowParallel=TRUE)

    set.seed(321)
    model1 <- train(classe~., data=trainingDataTransformed, method='rf', trControl=modelControl)

    #Stoping the clusters
    stopCluster(Cl)

    model1

    ## Random Forest 
    ## 
    ## 13737 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 12364, 12363, 12363, 12364, 12364, 12364, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    2    0.9914098  0.9891327
    ##   27    0.9914821  0.9892247
    ##   52    0.9808538  0.9757759
    ## 
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final value used for the model was mtry = 27.

Evaluation of the model on the training data set using the
confusionmatrix and accuracy, sensitivity and specificity metrics showed
maximal value of these metrics:

    predicted <- predict(model1, trainingDataTransformed)
    confusionMatrix(predicted, trainingDataTransformed$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 3906    0    0    0    0
    ##          B    0 2658    0    0    0
    ##          C    0    0 2396    0    0
    ##          D    0    0    0 2252    0
    ##          E    0    0    0    0 2525
    ## 
    ## Overall Statistics
    ##                                      
    ##                Accuracy : 1          
    ##                  95% CI : (0.9997, 1)
    ##     No Information Rate : 0.2843     
    ##     P-Value [Acc > NIR] : < 2.2e-16  
    ##                                      
    ##                   Kappa : 1          
    ##  Mcnemar's Test P-Value : NA         
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
    ## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
    ## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000

Evaluation of the model on the validation data set using the
confusionmatrix and accuracy, sensitivity and specificity metrics showed
99.4% accuracy (in other words very low expected out of sample error):

    predicted <- predict(model1, validationDataTransformed)
    confusionMatrix(predicted, validationDataTransformed$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1672    8    0    0    0
    ##          B    2 1129    5    0    0
    ##          C    0    2 1016   11    2
    ##          D    0    0    5  952    0
    ##          E    0    0    0    1 1080
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9939          
    ##                  95% CI : (0.9915, 0.9957)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9923          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9988   0.9912   0.9903   0.9876   0.9982
    ## Specificity            0.9981   0.9985   0.9969   0.9990   0.9998
    ## Pos Pred Value         0.9952   0.9938   0.9855   0.9948   0.9991
    ## Neg Pred Value         0.9995   0.9979   0.9979   0.9976   0.9996
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2841   0.1918   0.1726   0.1618   0.1835
    ## Detection Prevalence   0.2855   0.1930   0.1752   0.1626   0.1837
    ## Balanced Accuracy      0.9985   0.9949   0.9936   0.9933   0.9990

Displaying the final model
--------------------------

Finally we inspected the variable importance and estimated error rate
which is less than 1%:

    varImp(model1)

    ## rf variable importance
    ## 
    ##   only 20 most important variables shown (out of 52)
    ## 
    ##                      Overall
    ## roll_belt             100.00
    ## pitch_forearm          57.70
    ## yaw_belt               52.34
    ## roll_forearm           45.80
    ## magnet_dumbbell_y      45.74
    ## pitch_belt             45.38
    ## magnet_dumbbell_z      41.23
    ## accel_dumbbell_y       22.54
    ## magnet_dumbbell_x      16.90
    ## accel_forearm_x        16.77
    ## roll_dumbbell          16.61
    ## magnet_belt_z          14.35
    ## accel_dumbbell_z       13.72
    ## accel_belt_z           13.58
    ## magnet_forearm_z       13.40
    ## total_accel_dumbbell   12.83
    ## magnet_belt_y          12.56
    ## gyros_belt_z           11.63
    ## yaw_arm                10.24
    ## magnet_belt_x          10.16

    model1$finalModel

    ## 
    ## Call:
    ##  randomForest(x = x, y = y, mtry = param$mtry) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 27
    ## 
    ##         OOB estimate of  error rate: 0.72%
    ## Confusion matrix:
    ##      A    B    C    D    E class.error
    ## A 3900    3    1    0    2 0.001536098
    ## B   20 2631    5    2    0 0.010158014
    ## C    0   13 2374    9    0 0.009181970
    ## D    1    2   23 2223    3 0.012877442
    ## E    0    1    4   10 2510 0.005940594

Predicting the type of the respondents exercise on test data
------------------------------------------------------------

After inspecting and preprocessing the test data to adjust it based on
transformations performed on training (and validation) data sets, we
applied the final trained model to predict the class (type) of the
respondents' exercise:

    #summary(test)
    #str(test)
    #describe(test)
    #head(test)

    #Keeping only the variables included in training set
    selectedVar <- names(test) %in% names(trainingDataTransformed)
    testData <- test[, selectedVar]

    #Preprocessing by centering and scaling
    testDataTransformed <- predict(preProcValues, testData)

    #Adding classe variable
    testDataTransformed$classe <- rep(NA, length(testDataTransformed[,1]))
    testDataTransformed$classe <- as.factor(testDataTransformed$classe)

    #Forcing predictors into numeric 
    numCol2 <- ncol(testDataTransformed)-1
    for(i in c(1:numCol2)) {testDataTransformed[,i] = as.numeric(as.character(testDataTransformed[,i]))}

    #str(testDataTransformed)
    #any(is.na(testDataTransformed[,-c(53)]))

    #predicted <- predict(model1, testDataTransformed)

Note: Final predictions of the type of the respondents exercise for 20
test cases were omitted in this report due to plagiarism warnings.
