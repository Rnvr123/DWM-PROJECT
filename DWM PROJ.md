
The data that we will be using for this project are from the following sources:

- The [Breast Cancer Wisconsin (Diagnostic) Data Set](http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29) from the [UCI Machine Learning repository](http://archive.ics.uci.edu/ml/).

We will first load up the UCI dataset. The dataset itself does not contain column names, we've created a second file with only the column names, which we will use.
We will be using [tidyverse].

```r
library(tidyverse) # working with data frames, plotting
breastCancerData <- read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
               col_names = FALSE)
breastCancerDataColNames <- read_csv("https://raw.githubusercontent.com/fpsom/IntroToMachineLearning/gh-pages/data/wdbc.colnames.csv",
                                     col_names = FALSE)
colnames(breastCancerData) <- breastCancerDataColNames$X1
# Check out head of dataframe
breastCancerData %>% head()
```

We can see that our dataset contains 569 observations across 32 variables. This is what the first 6 lines look like:

```
# A tibble: 6 x 32
      ID Diagnosis Radius.Mean Texture.Mean Perimeter.Mean Area.Mean Smoothness.Mean
   <dbl> <chr>           <dbl>        <dbl>          <dbl>     <dbl>           <dbl>
1 8.42e5 M                18.0         10.4          123.      1001           0.118
2 8.43e5 M                20.6         17.8          133.      1326           0.0847
3 8.43e7 M                19.7         21.2          130       1203           0.110
4 8.43e7 M                11.4         20.4           77.6      386.          0.142
5 8.44e7 M                20.3         14.3          135.      1297           0.100
6 8.44e5 M                12.4         15.7           82.6      477.          0.128
# ... with 25 more variables: Compactness.Mean <dbl>, Concavity.Mean <dbl>,
#   Concave.Points.Mean <dbl>, Symmetry.Mean <dbl>, Fractal.Dimension.Mean <dbl>,
#   Radius.SE <dbl>, Texture.SE <dbl>, Perimeter.SE <dbl>, Area.SE <dbl>,
#   Smoothness.SE <dbl>, Compactness.SE <dbl>, Concavity.SE <dbl>, Concave.Points.SE <dbl>,
#   Symmetry.SE <dbl>, Fractal.Dimension.SE <dbl>, Radius.Worst <dbl>, Texture.Worst <dbl>,
#   Perimeter.Worst <dbl>, Area.Worst <dbl>, Smoothness.Worst <dbl>,
#   Compactness.Worst <dbl>, Concavity.Worst <dbl>, Concave.Points.Worst <dbl>,
#   Symmetry.Worst <dbl>, Fractal.Dimension.Worst <dbl>
```

We will also make our `Diagnosis` column a factor:



```r
# Make Diagnosis a factor
breastCancerData$Diagnosis <- as.factor(breastCancerData$Diagnosis)
```

We will first remove the first column, which is the unique identifier of each row:


```r
# Remove first column
breastCancerDataNoID <- breastCancerData[2:ncol(breastCancerData)]
# View head
breastCancerDataNoID %>% head()
```

The output should like like this:

```
# A tibble: 6 x 31
  Diagnosis Radius.Mean Texture.Mean Perimeter.Mean Area.Mean Smoothness.Mean
  <fct>           <dbl>        <dbl>          <dbl>     <dbl>           <dbl>
1 M                18.0         10.4          123.      1001           0.118
2 M                20.6         17.8          133.      1326           0.0847
3 M                19.7         21.2          130       1203           0.110
4 M                11.4         20.4           77.6      386.          0.142
5 M                20.3         14.3          135.      1297           0.100
6 M                12.4         15.7           82.6      477.          0.128
# ... with 25 more variables: Compactness.Mean <dbl>, Concavity.Mean <dbl>,
#   Concave.Points.Mean <dbl>, Symmetry.Mean <dbl>, Fractal.Dimension.Mean <dbl>,
#   Radius.SE <dbl>, Texture.SE <dbl>, Perimeter.SE <dbl>, Area.SE <dbl>,
#   Smoothness.SE <dbl>, Compactness.SE <dbl>, Concavity.SE <dbl>, Concave.Points.SE <dbl>,
#   Symmetry.SE <dbl>, Fractal.Dimension.SE <dbl>, Radius.Worst <dbl>, Texture.Worst <dbl>,
#   Perimeter.Worst <dbl>, Area.Worst <dbl>, Smoothness.Worst <dbl>,
#   Compactness.Worst <dbl>, Concavity.Worst <dbl>, Concave.Points.Worst <dbl>,
#   Symmetry.Worst <dbl>, Fractal.Dimension.Worst <dbl>
```

We have many variables in this dataset. Since maximum information is provided by the first 5 variables we will focus only on them.

```r
library(GGally)
ggpairs(breastCancerDataNoID[1:5], aes(color=Diagnosis, alpha=0.4))
```

![ggpairs output of the first 5 variables](https://raw.githubusercontent.com/fpsom/IntroToMachineLearning/gh-pages/static/images/ggpairs5variables.png "ggpairs output of the first 5 variables")

The above graph shows correlation between continous variables and box plots in the upper part , scatter plots on the lower part , density plots on the diagnal and the sides show the histograms for the combinations between the categorical and the continuous variables. The aes parameter allows us to create and fill the density plots, scatter plots and other plots with different colors based on the groups of the response variable.
From the above density plots x we understand that the features have widely varying centers and scales (means and standard deviations), so we'll want to center and scale them in some situations. We will use the `[caret]` package for this, and specifically, the `preProcess` function.

The `preProcess` function can be used for many operations on predictors, including centering and scaling. Centering and Scaling is done to normalise the features. Certain features have different ranges and therefore affect the final response variable differently. Normalisation helps with this isse . The function `preProcess` estimates the required parameters for each operation and `predict.preProcess` is used to apply them to specific data sets.
```r
library(caret)
# Center & scale data
ppv <- preProcess(breastCancerDataNoID, method = c("center", "scale"))
breastCancerDataNoID_tr <- predict(ppv, breastCancerDataNoID)
```

Let's have a look on the impact of this process by viewing the summary of the first 5 variables before and after the process:

```r
# Summarize first 5 columns of the original data
breastCancerDataNoID[1:5] %>% summary()
```

The resulting summary should look like this:

```
Diagnosis  Radius.Mean      Texture.Mean   Perimeter.Mean     Area.Mean     
B:357     Min.   : 6.981   Min.   : 9.71   Min.   : 43.79   Min.   : 143.5  
M:212     1st Qu.:11.700   1st Qu.:16.17   1st Qu.: 75.17   1st Qu.: 420.3  
          Median :13.370   Median :18.84   Median : 86.24   Median : 551.1  
          Mean   :14.127   Mean   :19.29   Mean   : 91.97   Mean   : 654.9  
          3rd Qu.:15.780   3rd Qu.:21.80   3rd Qu.:104.10   3rd Qu.: 782.7  
          Max.   :28.110   Max.   :39.28   Max.   :188.50   Max.   :2501.0
```

Let's check the summary of the re-centered and scaled data

```r
# Summarize first 5 columns of the re-centered and scaled data
breastCancerDataNoID_tr[1:5] %>% summary()
```

It now should look like this:

```
Diagnosis  Radius.Mean       Texture.Mean     Perimeter.Mean      Area.Mean      
B:357     Min.   :-2.0279   Min.   :-2.2273   Min.   :-1.9828   Min.   :-1.4532  
M:212     1st Qu.:-0.6888   1st Qu.:-0.7253   1st Qu.:-0.6913   1st Qu.:-0.6666  
          Median :-0.2149   Median :-0.1045   Median :-0.2358   Median :-0.2949  
          Mean   : 0.0000   Mean   : 0.0000   Mean   : 0.0000   Mean   : 0.0000  
          3rd Qu.: 0.4690   3rd Qu.: 0.5837   3rd Qu.: 0.4992   3rd Qu.: 0.3632  
          Max.   : 3.9678   Max.   : 4.6478   Max.   : 3.9726   Max.   : 5.2459  
```

As, we can observe here, all variables in our new data have a mean of 0 while maintaining the same distribution of the values. However, this also means that the absolute values do not correspond to the "real", original data - and is just a representation of them.

We can also check whether our plot has changed with the new data:

```r
library(GGally)
ggpairs(breastCancerDataNoID_tr[1:5], aes(color=Diagnosis, alpha=0.4))
```

![ggpairs output of the first 5 variables of the recentered/rescaled data](https://raw.githubusercontent.com/fpsom/IntroToMachineLearning/gh-pages/static/images/ggpairs5variables_tr.png "ggpairs output of the first 5 variables of the recentered/rescaled data")
From above we can tell that the relationships between variables have not changed and neither have their correlations , but the ranges on the far left hand side have been standardised.


#### Dimensionality Reduction and PCA

Since in the UCI dataset there are too many features to keep track of , Dimensionality reduction is used to reduce the number of features yet still keep much of the information from these variables.

Principal component analysis (PCA) is one of the most commonly used methods of dimensionality reduction, and extracts the features with the largest variance. What PCA essentially does is the following:
- The first step of PCA is to decorrelate your data and this corresponds to a linear transformation of the vector space your data lie in;
- The second step is the actual dimension reduction; what is really happening is that your decorrelation step (the first step above) transforms the features into new and uncorrelated features; this second step then chooses the features that contain most of the information about the data.

Let's have a look into the variables that we currently have, and apply PCA to them. As you can see, we will be using only the numerical variables (i.e. we will exclude the first two, `ID` and `Diagnosis`):

```r
ppv_pca <- prcomp(breastCancerData[3:ncol(breastCancerData)], center = TRUE, scale. = TRUE)
```

We can use the `summary()` function to get a summary of the PCA:

```r
summary(ppv_pca)
```

The resulting table, shows us the importance of each Principal Component; the standard deviation, the proportion of the variance that it captures, as well as the cumulative proportion of variance capture by the principal components.

```
Importance of components:
                          PC1    PC2     PC3     PC4     PC5     PC6     PC7     PC8    PC9
Standard deviation     3.6444 2.3857 1.67867 1.40735 1.28403 1.09880 0.82172 0.69037 0.6457
Proportion of Variance 0.4427 0.1897 0.09393 0.06602 0.05496 0.04025 0.02251 0.01589 0.0139
Cumulative Proportion  0.4427 0.6324 0.72636 0.79239 0.84734 0.88759 0.91010 0.92598 0.9399
                          PC10   PC11    PC12    PC13    PC14    PC15    PC16    PC17
Standard deviation     0.59219 0.5421 0.51104 0.49128 0.39624 0.30681 0.28260 0.24372
Proportion of Variance 0.01169 0.0098 0.00871 0.00805 0.00523 0.00314 0.00266 0.00198
Cumulative Proportion  0.95157 0.9614 0.97007 0.97812 0.98335 0.98649 0.98915 0.99113
                          PC18    PC19    PC20   PC21    PC22    PC23   PC24    PC25    PC26
Standard deviation     0.22939 0.22244 0.17652 0.1731 0.16565 0.15602 0.1344 0.12442 0.09043
Proportion of Variance 0.00175 0.00165 0.00104 0.0010 0.00091 0.00081 0.0006 0.00052 0.00027
Cumulative Proportion  0.99288 0.99453 0.99557 0.9966 0.99749 0.99830 0.9989 0.99942 0.99969
                          PC27    PC28    PC29    PC30
Standard deviation     0.08307 0.03987 0.02736 0.01153
Proportion of Variance 0.00023 0.00005 0.00002 0.00000
Cumulative Proportion  0.99992 0.99997 1.00000 1.00000
```

Principal Components are the underlying structure in the data. They are the directions where there is the most variance, the directions where the data is most spread out. This means that we try to find the straight line that best spreads the data out when it is projected along it. This is the first principal component, the straight line that shows the most substantial variance in the data.

PCA is a type of linear transformation on a given data set that has values for a certain number of variables (coordinates) for a certain amount of spaces. In this way, you transform a set of `x` correlated variables over `y` samples to a set of `p` uncorrelated principal components over the same samples.

Where many variables correlate with one another, they will all contribute strongly to the same principal component. Where your initial variables are strongly correlated with one another, you will be able to approximate most of the complexity in your dataset with just a few principal components. As you add more principal components, you summarize more and more of the original dataset. Adding additional components makes your estimate of the total dataset more accurate, but also more unwieldy.

Every eigenvector has a corresponding eigenvalue. Simply put, an eigenvector is a direction, such as "vertical" or "45 degrees", while an eigenvalue is a number telling you how much variance there is in the data in that direction. The eigenvector with the highest eigenvalue is, therefore, the first principal component. The number of eigenvalues and eigenvectors that exits is equal to the number of dimensions the data set has. In our case, we had 30 variables (32 original, minus the first two), so we have produced 30 eigenvectors / PCs. And we can see that we can address more than 95% of the variance (0.95157) using only the first 10 PCs.

We should also have a deeper look in our PCA object:

```r
str(ppv_pca)
```

The output should look like this:

```
List of 5
 $ sdev    : num [1:30] 3.64 2.39 1.68 1.41 1.28 ...
 $ rotation: num [1:30, 1:30] -0.219 -0.104 -0.228 -0.221 -0.143 ...
  ..- attr(*, "dimnames")=List of 2
  .. ..$ : chr [1:30] "Radius.Mean" "Texture.Mean" "Perimeter.Mean" "Area.Mean" ...
  .. ..$ : chr [1:30] "PC1" "PC2" "PC3" "PC4" ...
 $ center  : Named num [1:30] 14.1273 19.2896 91.969 654.8891 0.0964 ...
  ..- attr(*, "names")= chr [1:30] "Radius.Mean" "Texture.Mean" "Perimeter.Mean" "Area.Mean" ...
 $ scale   : Named num [1:30] 3.524 4.301 24.299 351.9141 0.0141 ...
  ..- attr(*, "names")= chr [1:30] "Radius.Mean" "Texture.Mean" "Perimeter.Mean" "Area.Mean" ...
 $ x       : num [1:569, 1:30] -9.18 -2.39 -5.73 -7.12 -3.93 ...
  ..- attr(*, "dimnames")=List of 2
  .. ..$ : NULL
  .. ..$ : chr [1:30] "PC1" "PC2" "PC3" "PC4" ...
 - attr(*, "class")= chr "prcomp"
```

The information listed captures the following:

1. The center point (`$center`), scaling (`$scale`) and the standard deviation(`$sdev`) of each principal component
2. The relationship (correlation or anticorrelation, etc) between the initial variables and the principal components (`$rotation`)
3. The values of each sample in terms of the principal components (`$x`)

Let's try to visualize the results we've got so far. We will be using the [`ggbiplot` library](https://github.com/vqv/ggbiplot) for this purpose.

```r
ggbiplot(ppv_pca, choices=c(1, 2),
         labels=rownames(breastCancerData),
         ellipse=TRUE,
         groups = breastCancerData$Diagnosis,
         obs.scale = 1,
         var.axes=TRUE, var.scale = 1) +
  ggtitle("PCA of Breast Cancer Dataset")+
  theme_minimal()+
  theme(legend.position = "bottom")
```

![Visualization of the first two PCs on the UCI Breast Cancer dataset](https://raw.githubusercontent.com/fpsom/IntroToMachineLearning/gh-pages/static/images/pc12Visualization_Full.png "Visualization of the first two PCs on the UCI Breast Cancer dataset")

Using Decision Tree Model on the above dataset-
```
tree.variable=tree(Diagnosis~.,breastCancerData)
summary(tree.variable)
```
The output for the above is as follows-
```
Classification tree:
tree(formula = Diagnosis ~ ., data = breastCancerData)
Variables actually used in tree construction:
[1] "Perimeter.Worst"      "Concave.Points.Worst" "Radius.SE"           
[4] "Texture.Worst"        "Symmetry.Worst"       "Smoothness.Worst"    
[7] "Radius.Mean"          "Smoothness.Mean"     
Number of terminal nodes:  12 
Residual mean deviance:  0.09294 = 51.77 / 557 
Misclassification error rate: 0.01757 = 10 / 569
```
The misclassification error rate for this case is 0.01757. Misclassification error rate is simply the number of observations that were misclassified divided by the total number of observations

In order to properly evaluate the performance of a classification tree on these data, we must estimate the test error rather than simply computing the training error. We split the observations into a training set and a test set, build the tree using the training set, and evaluate its performance on the test data. The predict() function can be used for this purpose. In the case of a classification tree, the argument type="class" instructs R to return the actual class prediction. Here we have performed a 50-50 split for training and test data.
```
train=sample(1:nrow(breastCancerData),nrow(breastCancerData)/2)
tree.test.variable=breastCancerData[-train,]
Diagnosis=breastCancerData$Diagnosis
Diagnosis.test=Diagnosis[-train]
tree.variable=tree(Diagnosis~.,breastCancerData,subset=train)
tree.pred=predict(tree.variable,tree.test.variable,type="class")
table(tree.pred,Diagnosis.test)
```
Confusion matrix for the above is -
```
         Diagnosis.test
tree.pred   B   M
        B 170  12
        M   9  94
```
The accuracy of this case is 95.3%

For Random forest case-
```
rforest.variable=randomForest(Diagnosis~.,breastCancerData)
rforest.pred=predict(rforest.variable,tree.test.variable,type="class")
table(rforest.pred,Diagnosis.test)
rforest.variable=randomForest(Diagnosis~.,breastCancerData)
rforest.pred=predict(rforest.variable,tree.test.variable,type="class")
table(rforest.pred,Diagnosis.test)
```
Output confusion matrix-
```
            Diagnosis.test
rforest.pred   B   M
           B 179   0
           M   0 106
```
The accuracy of the random forest case is 100%
