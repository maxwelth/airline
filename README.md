# Airline-Passenger-Satisfaction-Classification
Classifying passenger satisfaction using SVM, Gaussian Naive Bayes, Logistic Regression.

# Airline Passenger Satisfaction : Project Overview
* Predicted overall passenger satisfaction from the data collected from survey.
* Aggregated some feature that are highly correlated to reduce multicollinearity problem.
* Applied Support Vector Machine (SVM), Gaussian Naive Bayes and Logistic Regression and implemented GridsearchCV to tune the best hyperparameters. 
* Evaluated and chose the best model according to Accuracy metric.

## Code and Resources Used 
* **Python Version:** 3.8.5
* **Conda Version:** 4.9.2
* **Packages:** pandas, numpy, matplotlib, seaborn, sklearn, missingpy
* **Dataset:** https://www.kaggle.com/teejmahal20/airline-passenger-satisfaction

## Data Pre-processing

*	Dropped unimportant feature such as _index_ and _id_
* Checked if the data was imbalanced by plotting the class of label
*	Imputed missing values using _MissForest_ , _MissForest_ fits a random forest model algorithm to the data and predict the missing value iteratively
* Aggregated two features ('Arrival Delay' and 'Departure Delay') that were highly correlated by taking their average
* Mapping the label of categorical variable (based on intrinsic order for feature with unique value >2)
*	Eliminated some feature using _RFECV_ with RandomForestClassifier estimator
* Scaled the data and also applied _Principal Component Analysis_ (PCA) to get components that satisfy atleast 95% variance

## Model Training

After made sure every feature was scaled and had no missing value, the data was split into 70% training set dan 30% test set. Here, we splitted the scaled data with PCA and the scaled data without PCA. Later, both case were compared by evaluating the performance given the model.  
(All model were trained with default parameter)

## Model performance 
(_Accuracy Metric_)

### Scaled data (with PCA)
*	**SVM (rbf)**              : 94.75%
*	**SVM(linear)**            : 86.79%
*	**SVM(poly)**              : 92.10%
*	**Gaussian Naive Bayes**   : 82.67%
*	**Logistic Regression**    : 86.39%

### Scaled data (without PCA)
*	**SVM (rbf)**              : 95.54%
*	**SVM(linear)**            : 87.60%
*	**SVM(poly)**              : 93.93%
*	**Gaussian Naive Bayes**   : 86.68%
*	**Logistic Regression**    : 87.42%

## Tuned Model

Based on the result of the previous model performance, *SVM (rbf) - non PCA* was selected. Then, the hyperparameters were tuned by using _GridSearchCV_ with 5-fold cross validation (best params result: C=1, gamma=0.1). When the tuned model applied to the test data, the accuracy score yield a result of 95.99%, which is 0.45% higher than the untuned one.

_1 : satisfied, 0 : neutral or dissatisfied_
![](conf%20matrix.png)



