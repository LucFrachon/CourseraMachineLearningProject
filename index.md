# Practical Machine Learning Project: Human Activity Qualification
Luc Frachon  
21 janvier 2016  

# Abstract

Using data collected through accelerometers worn by a group of people performing weight lifting, we fit classification algorithms to the data to predict whether the exercise was performed correcty or with different types of typical errors.  
After trying several models and parameters, we conclude that the best compromise between absolute performance and speed is provided by a Random Forest algorithm applied to data dimensionally reduced through PCA.

---


# 1. Set up and data preparation

This analysis was performed using the following set up: 

*  Computer: Intel i5-4440 with 8GB RAM, solid-state main hard drive
*  OS: Windows 10, build number 10586
*  R version 3.2.2
*  RStudio version 0.99.484
*  Locale = French_France.1252

In this analysis we use the following libraries: caret, ggplot2, dplyr, parallel and doParallel. These last two enable us to take advantage of the quad-core configuration to speed up calculations. We also set the randomiser seed to 2302.


```
## [1] 16194
```

As usual, we read the data and tidy it up to make it easier to use.


The data comes is kindly made available from the team cited in References. It contains 19622 observations and 160 variables, 153 of which can be considered potential predictors. The outcome is 'classe', a categorical variable with 5 classes; A corresponds to correctly executed movement whereas B through E denote specific errors in execution.  
Exploratory analysis shows that the dataset is ordered by 'classe', then 'user_name', then 'num_window', then timestamps (*figure 1*). 

![*Figure 1: Structure of the raw dataset. Different colours indicate different human users.*](index_files/figure-html/unnamed-chunk-3-1.png) 


The timestamps correspond to  2.5s intervals with a 0.5s overlap. Presumably, each window corresponds to one instance of weight lifting. We considered the possibility of grouping observations within each window and using aggregated statistics as predictors. Although this approach might have made training and prediction easier, we decided against it for three reasons:

* The research team had already tried different time aggregatesand decided that 2.5s gave the best results; it seemed somewhat superfluous to use a second layer of aggregation.
* In real life, there is no cue from the user to let the sensors know when they start and stop an activity, therefore relevant algorithms need to be able to predict without that information.
* We came up with excellent predictive performance without the need for this aggregation.

# 2. Pre-processing

The data contains many sparse columns; it is really an "all-or-nothing" situation: the majority of columns are empty or near-empty.


![*Figure 2: Distribution of the proportion of NAs in each column*](index_files/figure-html/unnamed-chunk-5-1.png) 

We address this by setting a threshold at 50% and dropping any column containing more than NAs the threshold. This effectively drops all the NAs. There are no other zero- or near-zero variance predictors.


```r
thresh <- .50
fullSetNoNAs<- fullSet[ , colSums(is.na(fullSet)) <= 
                            thresh * length(fullSet[[1]])]
rm(fullSet)  # Housekeeping
# Check how the number of NAs remaining:
sum(is.na(fullSetNoNAs))
```

```
## [1] 0
```


```r
# There are no other zero or near-zero variance predictors:
nearZeroVar(fullSetNoNAs[ , 7 : 58], freqCut = 95/5, saveMetrics = FALSE)
```

```
## integer(0)
```

# 3. Data Partitioning

We have a large number of observations, therefore we can afford to partition the data in 3:

* Training set (60%)
* Cross-validation set (20%), which we will use to select and tune the best algorithm
* Test set (20%), on which to estimate the model's quality. A small sample of 20 test cases is also provided by Coursera, but that number is too small for a reliable assessment.


```r
trainIndex <- createDataPartition(fullSetNoNAs$classe, 
                                  p = 0.6, list = F)
train1 <- fullSetNoNAs[trainIndex, ]
testAndCv1 <- fullSetNoNAs[-trainIndex, ]
cvIndex <- createDataPartition(testAndCv1$classe, p = 0.5, list = F)
cv1 <- testAndCv1[cvIndex, ]
test1 <- testAndCv1[-cvIndex, ]
rm(fullSetNoNAs)  # Housekeeping
```

# 4. Model Fitting and Prediction
## 4.1. Algorithm selection
We tried several algorithms, starting with simple ones:

* Decision tree
* Linear Discriminant Analysis
* Naive Bayes
* Random Forest

The first three gave unsatisfactory prediction accuracy both on the training and cross-validation sets:  


Model        | Accuracy on CV set
------------ | -------------------
Tree         | 49.6%
LDA          | 70.6%
Naive Bayes  | 74.6%    

We will now focus on the Random Forest algorithm, which yielded the best results by far.

## 4.2. Random Forest on untransformed data
We first deploy the Random Forest algorithm on the training set without prior transformation, using the 'caret' package:


```r
forestFit <- train(classe ~ ., method = "rf", data = train1[, 7:59],
                   prox = T, trControl = fitControl)
```

For Random Forest, calculating accuracy by predicting on the training set is incorrect. A better measure of training accuracy is the OOB prediction error. Here, it is excellent as reported here:  


```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, proximity = ..1) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.77%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3340    6    0    0    2 0.002389486
## B   15 2252   10    2    0 0.011847301
## C    0    8 2037    9    0 0.008276534
## D    0    0   25 1904    1 0.013471503
## E    0    1    3    9 2152 0.006004619
```

Accuracy on the cross-validation set is also excellent:


```r
forestCvPred <- predict(forestFit, newdata = cv1)
sum(forestCvPred == cv1[, 59]) / length(forestCvPred)
```

```
## [1] 0.9923528
```

```r
table(Actual = cv1[, 59], Prediction = forestCvPred)
```

```
##       Prediction
## Actual    A    B    C    D    E
##      A 1115    0    1    0    0
##      B    6  750    2    1    0
##      C    0    8  673    3    0
##      D    0    0    4  639    0
##      E    0    0    1    4  716
```

Out-of-bag error is 1 - accuracy, therefore 0.0076472.  

The training time is quite long: 100.9 minutes. However the more important measure for the designed application is prediction time, as the user expects near-instant feedback. From that perspective, we found that Naive-Bayes had the worst prediction time by far. Fortunately the Random Forest prediction time is quick (a few seconds).


## 4.3 Random Forest on Principal Components
An interesting question for datasets with a relatively large numbers of variables is how much accuracy we lose in relation to data compression levels. The method used here is Principal Component Analysis and we will assess accuracy over a range of different values for the threshold parameter (levels of variance explained by the principal components).

The RF algorithm includes bootstrapping which theoretically reduces the need for a cross-validation set. However we are going to evaluate the model against different threshold values, which means that to a certain extent, we will tune our parameters to the cross-validation set. We therefore need an untouched test set to evaluate our model's final performance.

We rely on the 'train' function to find the optimal number of tries for the Random Forest Algorithm, while we manually vary the PCA threshold (proportion of variance explained by the computed Principal Components)


```r
PCA.thresh <- c(0.95, 0.9, 0.8, 0.7, 0.5)

rfResultsWithPCA <- function(t, outcome, model, trainData, cvData,
                              control) {
    results = list()
    outcomeIdx <- outcome != colnames(trainData)
    trainData.PCAselection <- trainData[ , outcomeIdx]
    cvData.PCAselection <- cvData[ , outcomeIdx]
    # Compute PCA with specified threshold:
    pcaModel <- preProcess(trainData.PCAselection, 
                           method = "pca", thresh = t)
    # Training set recomputed using PCs:
    trainPCA <- predict(pcaModel, trainData.PCAselection)
    # Train specified classification model using the PC dataset:
    modelFitPCA <- train(trainData[[outcome]] ~ . , method = model,
                         data = trainPCA, trControl = control)
    # Calculate predictions for the CV dataset based on those PCs:
    cvDataPCA <- predict(pcaModel, cvData.PCAselection)
    cvPredPCA <- predict(modelFitPCA, newdata = cvDataPCA)
    results$accuracy <- sum(cvPredPCA == 
                                cvData[[outcome]])/length(cvPredPCA)
    results$fitObj <- modelFitPCA
    results$pcaObj <- pcaModel

    return(results)
}

rfModels <- lapply(PCA.thresh, FUN = rfResultsWithPCA,
                        outcome = "classe",
                        model = "rf",
                        trainData = train1[, 7:59],
                        cvData = cv1[, 7:59],
                        control = fitControl)
```




We can plot accuracy vs running time:


```r
accuracy <- numeric()
runtime <- numeric()

for (i in seq(1, length(PCA.thresh))){
    accuracy[i] <- rfModels[[i]]$accuracy
    runtime[i] <- rfModels[[i]]$fitObj$times$everything[3]
}
print(round(runtime))
```

```
## [1] 643 479 332 248 200
```

```r
print(round(accuracy, 3))
```

```
## [1] 0.970 0.964 0.951 0.931 0.831
```

![*Figure 3: Accuacy vs. Runtime and PCA threshold values*](index_files/figure-html/unnamed-chunk-15-1.png) 

It turns out that even at 50% variance retained in the PCA process, we already get 83% accuracy and the algorithm runs in 3.5 minutes. At 95% variance retained, the running time triples but accuracy is 97%. The sweet spot seems to be somewhere around 85-90% variance retained. In the remainder of this document, we will use a threshold of 90% but if we were very concerned with running times, 70% would be a perfectly viable choice.

#5. Performance assessment

Using our model including PCA (at 90% threshold) + Random Forest, we can run predictions on the test set, which we have not used so far:


```r
# Apply PCA to the test set:
testDataPCA <- predict(rfModels[[2]]$pcaObj, newdata = test1)
forestTestPred <- predict(rfModels[[2]]$fitObj, newdata = testDataPCA)
#Accuracy:
sum(forestTestPred == test1$classe) / nrow(test1)
```

```
## [1] 0.964313
```

```r
table(Actual = test1$classe, Prediction = forestTestPred)
```

```
##       Prediction
## Actual    A    B    C    D    E
##      A 1092    6    6    9    3
##      B   20  711   24    0    4
##      C    4    8  665    7    0
##      D    0    0   31  611    1
##      E    0    4    9    4  704
```

The prediction accuracy remains extremely satisfactory on the test set. Out-of-sample error is 1 - accuracy, therefore 0.035687.

#6. Assignment quiz cases

The Coursera assignment includes 20 test cases that are used to assess the model performance.


```r
quizSetRaw <- read.csv("data/pml-testing.csv",
                    na.strings = c("", "NA", "#DIV/0!"))
# Some variables improperly loaded as factors or integer vectors:
colSelect <- select(quizSetRaw, -(X : cvtd_timestamp), -new_window,
                    -num_window)
for (c in names(colSelect)) {
    colSelect[, c] <- as.numeric(colSelect[, c])
}
# We get rid of the problematic and incomplete cvtd_timestamp variable
# and reconstruct our data set:
quizSet <- data.frame(select(quizSetRaw, (X : raw_timestamp_part_2),
                             new_window, num_window), colSelect)

# Apply PCA to the test set:
quizSetPCA <- predict(rfModels[[2]]$pcaObj, newdata = quizSet)
# Calculate predictions:
forestQuizPred <- predict(rfModels[[2]]$fitObj, newdata = quizSetPCA)
forestQuizPred
```

```
##  [1] B A A A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

19 out of 20 predictions turned out to be correct (only #3 was incorrectly predicted as A).

Note: the "no-PCA" Random Forest algorithm finds all 20 cases correctly:

```r
forestQuizNoPCA <- predict(forestFit, newdata = quizSet)
forestQuizNoPCA
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

#Conclusions

This assignment demonstrated the predictive power of the Random Forest algorithm on high-dimensional data. The trade-off is fairly long computational times but this can be largely mitigated by operating dimensionality reduction during the pre-processing phase while retaining high levels of accuracy.

---

### References
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13).   Stuttgart, Germany: ACM SIGCHI, 2013.  
  
Read more: [http://groupware.les.inf.puc-rio.br/har#sbia_paper_section#ixzz3xsGogIbK]
