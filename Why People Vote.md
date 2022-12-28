Why People Vote
================
2022-12-20

### About the Study and the Data Set

In 2006, three researchers conducted a study in Michigan to examine the
idea that people might be influenced to vote because of social pressure
or a sense of civic duty. They used logistic regression and
classification trees to analyze the data they collected and published
their findings in a research paper in 2008. The purpose of the study was
to understand why people choose to participate in elections, even though
an individual’s vote may not make a significant difference in the
outcome. One theory they explored was that people feel obligated to vote
because they believe it is a civic duty or because they fear being
judged by others if they don’t participate.

In this study, logistic regression and decision trees will be utilized
to try to answer this matter.

The researchers divided a group of approximately 344,000 voters into
five different groups for their study: a control group and four
treatment groups. The control group represented the typical voting
situation, and did not receive any special treatment. The other four
groups received different messages in an attempt to influence their
voting behavior. The “civic duty” group received a message simply
stating “DO YOUR CIVIC DUTY - VOTE!” The “hawthorne” group received the
same message as the “civic duty” group, but with the added message “YOU
ARE BEING STUDIED” and an explanation that their voting behavior would
be examined using public records. The “self” group received the “civic
duty” message as well as a list of recent voting records for all members
of their household, and a promise that an updated list would be sent
after the election. The “neighbors” group received the same message as
the “self” group, but with the added voting records of their neighbors
in order to increase social pressure. The researchers also included
variables in their dataset such as sex, year of birth, and whether or
not the person ended up voting.

### EDA

``` r
gerber = read.csv("gerber.csv")
```

``` r
library(knitr)
kable(table(gerber$voting))
```

| Var1 |   Freq |
|:-----|-------:|
| 0    | 235388 |
| 1    | 108696 |

``` r
barplot(table(gerber$voting))
```

![](Why%20People%20Vote_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

Let’s see what proportion of records in the data set did in fact vote.

``` r
table(gerber$voting)[2]/nrow(gerber)
```

    ##         1 
    ## 0.3158996

31.5%.  
Let’s see which treatment group had the highest voting rate.

``` r
tapply(gerber$voting, gerber$civicduty, mean)
```

    ##         0         1 
    ## 0.3160698 0.3145377

``` r
tapply(gerber$voting, gerber$hawthorne, mean)
```

    ##         0         1 
    ## 0.3150909 0.3223746

``` r
tapply(gerber$voting, gerber$self, mean)
```

    ##         0         1 
    ## 0.3122446 0.3451515

``` r
tapply(gerber$voting, gerber$neighbors, mean)
```

    ##         0         1 
    ## 0.3081505 0.3779482

As we can see the highest voting rate between all the 4 treatment group
is Neighbors group.

### Building a Logistic Regression Model

``` r
model = glm(voting ~ self + neighbors + hawthorne + civicduty, data = gerber, family = binomial)
summary(model)
```

    ## 
    ## Call:
    ## glm(formula = voting ~ self + neighbors + hawthorne + civicduty, 
    ##     family = binomial, data = gerber)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -0.9744  -0.8691  -0.8389   1.4586   1.5590  
    ## 
    ## Coefficients:
    ##              Estimate Std. Error  z value Pr(>|z|)    
    ## (Intercept) -0.863358   0.005006 -172.459  < 2e-16 ***
    ## self         0.222937   0.011867   18.786  < 2e-16 ***
    ## neighbors    0.365092   0.011679   31.260  < 2e-16 ***
    ## hawthorne    0.120477   0.012037   10.009  < 2e-16 ***
    ## civicduty    0.084368   0.012100    6.972 3.12e-12 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 429238  on 344083  degrees of freedom
    ## Residual deviance: 428090  on 344079  degrees of freedom
    ## AIC: 428100
    ## 
    ## Number of Fisher Scoring iterations: 4

It apperas that a model that predict the voting behaviour depending on
the four treatment groups, have all the four groups as significant
variables. Let’s investigate further.

``` r
predictmodel = predict(model, type = "response")
kable(table(gerber$voting, predictmodel >= 0.3))
```

|     |  FALSE |   TRUE |
|:----|-------:|-------:|
| 0   | 134513 | 100875 |
| 1   |  56730 |  51966 |

Let’s see the accuracy of the model given this threshold.

``` r
x = table(gerber$voting, predictmodel >= 0.3)
(x[1] + x[4])/ sum(x)
```

    ## [1] 0.5419578

The accuracy of the model is 0.54. Now let’s try to raise this
threshold.

``` r
table(gerber$voting, predictmodel > 0.5)
```

    ##    
    ##      FALSE
    ##   0 235388
    ##   1 108696

``` r
x = table(gerber$voting, predictmodel > 0.5)
(x[1])/ sum(x)
```

    ## [1] 0.6841004

Let’s see the AUC of the Model

``` r
library(ROCR)
pred = prediction(predictmodel, gerber$voting)
as.numeric(performance(pred, "auc")@y.values)
```

    ## [1] 0.5308461

Given that the AUC of the model is 0.53, which is quite low, and given
that the accuracy of the model is low compared to a base line model that
predict “not voting” all the time, it appears that the model is
perfoming poorly even though the assumption is that treatment groups
should make a difference.

### Trees

Lets build a classification and regression tree

``` r
library(rpart)
library(rpart.plot)
CARTmodel = rpart(voting ~ neighbors + self + civicduty + hawthorne, data = gerber)
prp(CARTmodel)
```

![](Why%20People%20Vote_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

So it appears that non of the variables is siginificant enough in this
model to make a split in the tree. Let’s build another one with
different configirations.

``` r
CARTmodel2 = rpart(voting ~ neighbors + self + civicduty + hawthorne, data = gerber, cp=0.0)
prp(CARTmodel2)
```

![](Why%20People%20Vote_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

This appears to be a better model since not only it makes splits but
also the first split in the tree is the ***neighbors*** variable which
has the highest voting ratio as we descovered earlier on in the EDA.  
We can see, for instance, that a fraction of 31% of ***civicduty***
treatment group has voted.  
Let’s try to further improve this model.

``` r
CARTmodel3 = rpart(voting ~ neighbors + self + civicduty + hawthorne + sex, data = gerber, cp=0.0)
prp(CARTmodel3)
```

![](Why%20People%20Vote_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

We can see that the ***sex*** variable appears to be a secondary split
in each group. For instance, in ***control group***, it appears that men
(sex = 0) are more likely to vote than women.

Now let’s utilize tree’s ability to handle non-linear relations, to
further draw conclusions about the datase.

``` r
CARTcontrol = rpart(voting ~ control, data = gerber, cp=0.0)
CARTsex = rpart(voting ~ sex + control, data = gerber, cp = 0.0)
prp(CARTcontrol)
```

![](Why%20People%20Vote_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

``` r
prp(CARTsex)
```

![](Why%20People%20Vote_files/figure-gfm/unnamed-chunk-14-2.png)<!-- -->

From the first tree, we can understand that the difference in the voting
predicted probability between being in a treatment group and not being
in one (control) is 0.04. In the second tree, we can see the difference
in probabilities across males and females and how being in control
groups will affect different genders. For instance, we can see that Men
are more affected by being in treatment groups than being left in
control group when it comes to voting pattern probability. Albeit, the
difference is not much significant (0.01).

Now let’s build a regression model and compare it to the CART model.

``` r
Modelcons = glm(voting ~ control + sex, data = gerber, family = "binomial")
summary(Modelcons)
```

    ## 
    ## Call:
    ## glm(formula = voting ~ control + sex, family = "binomial", data = gerber)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -0.9220  -0.9012  -0.8290   1.4564   1.5717  
    ## 
    ## Coefficients:
    ##              Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept) -0.635538   0.006511 -97.616  < 2e-16 ***
    ## control     -0.200142   0.007364 -27.179  < 2e-16 ***
    ## sex         -0.055791   0.007343  -7.597 3.02e-14 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 429238  on 344083  degrees of freedom
    ## Residual deviance: 428443  on 344081  degrees of freedom
    ## AIC: 428449
    ## 
    ## Number of Fisher Scoring iterations: 4

So it appears that this logistic regression model assign a negative
coefficient for the ***sex*** variable, indicating that the higher the
value of the variable ( being woman), the lower the probability of
voting being in a control group.

The regression tree was able to accurately predict the percentage of
people who voted in each of four different groupings: men who were not
in the control group, men who were in the control group, women who were
not in the control group, and women who were in the control group.
However, when logistic regression was used to analyze the “sex” and
“control” variables separately, rather than considering them jointly, it
did not perform as well in making predictions.

Let’s try to overcome this limitation.

``` r
possibilities = data.frame(sex = c(0, 0, 1, 1), control = c(0, 1, 0, 1))
predict(Modelcons, newdata = possibilities, type = "response")
```

    ##         1         2         3         4 
    ## 0.3462559 0.3024455 0.3337375 0.2908065

We can see that, we could calculate the probabilities of the four
possibilities precisely and we can see it in the tree.

This example demonstrated that decision trees, such as regression trees,
are able to identify complex, nonlinear relationships in the data that
logistic regression may not be able to capture. However, it is possible
to use variables that are a combination of two other variables in order
to improve the performance of logistic regression in certain cases.
