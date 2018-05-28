---
layout: single
title:  "Linear Regression Assumptions"
excerpt: "Testing linear regression assumptions in Python"
date:   2018-05-26 21:04:11 -0500
categories: post
classes: wide
header:
    image: /assets/images/assumptionPainting.jpg
    caption: "Photo credit: [Antonio Gisbert](https://en.wikipedia.org/wiki/Antonio_Gisbert#/media/File:Fusilamiento_de_Torrijos_(Gisbert).jpg)"
---

## Note: This is a draft that is still in progress

### To:Do

- Check & ensure consistency
- Proof read for typos
- Add note about assuming outliers are already accounted for
- Fix image links

---

Checking model assumptions is like commenting code. Everybody should be doing it often, but it sometimes ends up being looked over in reality. A failure to do either can result in a lot of time being confused, going down rabbit holes, and can have pretty serious consequences from not being interpreted correctly. 

Linear regression is a fundamental tool that has distinct advantages over other regression algorithms. Due to its simplicity, it's an exceptionally quick algorithm to train which typically makes it a good baseline algorithm for common regression scenarios. More importantly, models trained with linear regression are the most interpretable kind of regression models available - meaning it's easier to take action from the results of a linear regression model. However, if the assumptions are not satisfied, the interpretation of the results are not valid. This can be very dangerous depending on the application.

This post contains code for tests on the assumptions of linear regression and examples with both a real-world dataset and a toy dataset.

## The Data

Here are the variable descriptions straight from [the documentation](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html):

- **CRIM:** Per capita crime rate by town

- **ZN:** Proportion of residential land zoned for lots over 25,000 sq.ft.

- **INDUS:** Proportion of non-retail business acres per town.

- **CHAS:** Charles River dummy variable (1 if tract bounds river; 0 otherwise)

- **NOX:** Nitric oxides concentration (parts per 10 million)

- **RM:** Average number of rooms per dwelling

- **AGE:** Proportion of owner-occupied units built prior to 1940

- **DIS:** Weighted distances to five Boston employment centres

- **RAD:** Index of accessibility to radial highways

- **TAX:** Full-value property-tax rate per \$10,000

- **PTRATIO:** Pupil-teacher ratio by town

- **B:** 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town

    - **Note:** I really don't like this variable because I think it's both highly unethical to determine house prices by the color of people's skin in a given area in a predictive modeling scenario and it irks me that it singles out one ethnicity rather than including all others. I am leaving it in for this post to keep the code simple, but I would remove it in a real-world situation.

- **LSTAT:** % lower status of the population

- **MEDV:** Median value of owner-occupied homes in \$1,000's


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
%matplotlib inline


"""
Real-world data of Boston housing prices
Additional Documentation: https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html

Attributes:
data: Features/predictors
label: Target/label/response variable
feature_names: Abbreviations of names of features
"""
boston = datasets.load_boston()


"""
Artificial linear data uaing the same number of features and observations as the
Boston housing prices dataset for assumption test comparison
"""
linear_X, linear_y = datasets.make_regression(n_samples=boston.data.shape[0],
                                              n_features=boston.data.shape[1],
                                              noise=75)

# Setting feature names to x1, x2, x3, etc. if they are not defined
linear_feature_names = ['X'+str(feature+1) for feature in range(linear_X.shape[1])]
```

Now that the data is loaded in, let's preview it:


```python
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['HousePrice'] = boston.target

df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>HousePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>



### Initial Setup

Before we test the assumptions, we'll need to run the linear regressions themselves. My master function for performing all of the tests at the bottom does this automatically, but to abstract the assumption tests out to view them independently we'll have to re-write the individual tests to take the trained model as a parameter. 

Additionally, a few of the tests use residuals, so we'll write a quick function to calculate residuals. These are calculated once in the master function, but this extra function is to adhere to [DRY](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself) typing for the individual tests that use residuals.


```python
from sklearn.linear_model import LinearRegression

# Fitting the model
boston_model = LinearRegression()
boston_model.fit(boston.data, boston.target)

# Returning the R^2 for the model
boston_r2 = boston_model.score(boston.data, boston.target)
print('R^2: {0}'.format(boston_r2))
```

    R^2: 0.7406077428649428
    


```python
# Fitting the model
linear_model = LinearRegression()
linear_model.fit(linear_X, linear_y)

# Returning the R^2 for the model
linear_r2 = linear_model.score(linear_X, linear_y)
print('R^2: {0}'.format(linear_r2))
```

    R^2: 0.849848829621323
    


```python
def calculate_residuals(model, features, label):
    """
    Creates predictions on the features with the model and calculates residuals
    """
    predictions = model.predict(features)
    df_results = pd.DataFrame({'Actual': label, 'Predicted': predictions})
    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])
    
    return df_results
```

## Assumptions

### I) Linearity

This assumes that there is a linear relationship between the predictors and the response variable. If this assumption is violated, it may be resolved by either adding polynomial terms of some of the predictors or by adding additional variables to help capture the relationship between the predictors and the label.

If there is only one predictor, this is pretty easy to test with a scatter plot. Most cases aren't so we'll have to modify this by using a scatter plot to see our predicted values versus the actual values (in other words, view the residuals). Ideally, the points should lie on or around a diagonal line on the scatter plot.


```python
def linear_assumption(model, features, label):
    """
    Linearity: Assumes that there is a linear relationship between the predictors and
               the response variable. If not, either a quadratic term or another
               algorithm should be used.
    """
    print('Assumption 1: Linear Relationship between the Target and the Feature')
    print()
        
    print('Checking with a scatter plot of actual vs. predicted. Predictions should follow the diagonal line.')
    
    # Calculating residuals for the plot
    df_results = calculate_residuals(model, features, label)
    
    # Plotting the actual vs predicted values
    sns.lmplot(x='Actual', y='Predicted', data=df_results, fit_reg=False, size=5)
        
    # Plotting the diagonal line
    line_coords = np.arange(df_results.min().min(), df_results.max().max())
    plt.plot(line_coords, line_coords,  # X and y points
             color='darkorange', linestyle='--')
    plt.title('Actual vs. Predicted')
    plt.show()
```

We'll start with our linear dataset:


```python
linear_assumption(linear_model, linear_X, linear_y)
```

    Assumption 1: Linear Relationship between the Target and the Feature
    
    Checking with a scatter plot of actual vs. predicted. Predictions should follow the diagonal line.
    


![png](output_11_1.png)


We can see a relatively even spread around the diagonal line.

Now, let's compare it to the Boston dataset:


```python
linear_assumption(boston_model, boston.data, boston.target)
```

    Assumption 1: Linear Relationship between the Target and the Feature
    
    Checking with a scatter plot of actual vs. predicted. Predictions should follow the diagonal line.
    


![png](output_13_1.png)


We can see in this case that there is not a perfect linear relationship. Our predictions are biased towards lower values in both the lower end (around 5-10) and especially at the higher values (above 40).

### II) Normality

**TO DO:** Explain why this is important and what to do to fix it


```python
def multivariate_normal_assumption(model, features, label, feature_names=None, p_value_thresh=0.05):
    """
    Normality: Assumes that the predictors have normal distributions. If they are not normal,
               a non-linear transformation like a log transformation or box-cox transformation
               can be performed on the non-normal variable.
    """
    from statsmodels.stats.diagnostic import normal_ad
    print('Assumption 2: All variables are multivariate normal')
    print()
    
    print('Using the Anderson-Darling test for normal distribution')
    print('p-values from the test - below 0.05 generally means non-normal:')
    print()
    non_normal_variables = 0
        
    # Performing the Anderson-Darling test on each variable to test for normality
    for feature in range(features.shape[1]):
        p_value = normal_ad(features[:, feature])[1]
            
        # Adding to total count of non-normality if p-value exceeds threshold
        if p_value < p_value_thresh:
            non_normal_variables += 1
            
        # Printing p-values from the test
        print('{0}: {1}'.format(feature_names[feature], p_value))
                    
    print('\n{0} non-normal variables'.format(non_normal_variables))
    print()

    if non_normal_variables == 0:
        print('Assumption satisfied')
    else:
        print('Assumption not satisfied')
```

As with our previous assumption, we'll start with the linear dataset:


```python
multivariate_normal_assumption(linear_model, linear_X, linear_y,
                               feature_names=linear_feature_names)
```

    Assumption 2: All variables are multivariate normal
    
    Using the Anderson-Darling test for normal distribution
    p-values from the test - below 0.05 generally means non-normal:
    
    X1: 0.25800777524419044
    X2: 0.26112460545161537
    X3: 0.8397275309140688
    X4: 0.3280033132782176
    X5: 0.9347585904836029
    X6: 0.5514823178829662
    X7: 0.44063138327372103
    X8: 0.5409575811996296
    X9: 0.006039124306500998
    X10: 0.04704501213363774
    X11: 0.8810382690276023
    X12: 0.9276013770608211
    X13: 0.1386892103377878
    
    2 non-normal variables
    
    Assumption not satisfied
    

Interestingly, not all of our features for the linear data set are normally distributed:


```python
import seaborn as sns
%matplotlib inline

# Plotting histograms & kernel density estimates
sns.distplot(linear_X[:, 11], label='Normal')
sns.distplot(linear_X[:, 8], label='X7')
sns.distplot(linear_X[:, 9], label='X8')
plt.legend()
plt.title('Distributions of linear variables')
```




![png](output_20_1.png)


While these two failed the normality test, they don't violate it egregiously.

Now let's run the same test on the Boston dataset:


```python
multivariate_normal_assumption(boston_model, boston.data, boston.target,
                               feature_names=boston.feature_names)
```

    Assumption 2: All variables are multivariate normal
    
    Using the Anderson-Darling test for normal distribution
    p-values from the test - below 0.05 generally means non-normal:
    
    CRIM: 0.0
    ZN: 0.0
    INDUS: 0.0
    CHAS: 0.0
    NOX: 2.631393323760771e-20
    RM: 4.72343897192116e-15
    AGE: 0.0
    DIS: 0.0
    RAD: 0.0
    TAX: 0.0
    PTRATIO: 0.0
    B: 0.0
    LSTAT: 2.5860442633430127e-19
    
    13 non-normal variables
    
    Assumption not satisfied
    
    

Not a single variable passed the test. Let's plot a few to see why:


```python
sns.distplot(boston.data[:, 0], label='CRIM')
sns.distplot(boston.data[:, 1], label='ZN')
sns.distplot(boston.data[:, 2], label='INDUS')
plt.legend()
plt.title('Distributions of Misc. Boston Variables')
```






![png](output_24_1.png)


It's pretty clear that we're having violations of the normality assumption here.

### III) Multicollinearity

This assumes that the predictors used in the regression are not correlated with each other. Multicollinearity causes issues with the interpretation of the coefficients. Specifically, you can interpret a coefficient as "An increase of 1 in this predictor results in a change of (coefficient) in the response variable holding all other predictors constant." This becomes problematic when multicollinearity is present because you can't hold correlated predictors constant.

The other problem with multicollinearity is that it increases the standard error of the coefficients, which means that they may be shown as statistically insignificant even though they really are.

This can be fixed by other removing predictors with a high variance inflation factor (VIF) or performing dimensionality reduction.


```python
def multicollinearity_assumption(model, features, label, feature_names=None):
        """
        Multicollinearity: Assumes that predictors are not correlated with each other. If there is
                           correlation among the predictors, then either remove prepdictors with high
                           Variance Inflation Factor (VIF) values or perform dimensionality reduction
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        print('Assumption 3: Little to no multicollinearity among predictors')
        
        # Plotting the heatmap
        ax = plt.subplot(111)
        sns.heatmap(pd.DataFrame(features, columns=feature_names).corr())
        plt.show()
        
        print('Variance Inflation Factors (VIF)')
        print('> 10: An indication that multicollinearity may be present')
        print('> 100: Certain multicollinearity among the variables')
        print('-------------------------------------')
       
        # Gathering the VIF for each variable
        VIF = [variance_inflation_factor(features, i) for i in range(features.shape[1])]
        for idx, vif in enumerate(VIF):
            print('{0}: {1}'.format(feature_names[idx], vif))
        
        # Gathering and printing total cases of possible or definite multicollinearity
        possible_multicollinearity = sum([1 for vif in VIF if vif > 10])
        definite_multicollinearity = sum([1 for vif in VIF if vif > 100])
        print()
        print('{0} cases of possible multicollinearity'.format(possible_multicollinearity))
        print('{0} cases of definite multicollinearity'.format(definite_multicollinearity))
        print()

        if definite_multicollinearity == 0:
            if possible_multicollinearity == 0:
                print('Assumption satisfied')
            else:
                print('Assumption possibly satisfied')
        else:
            print('Assumption not satisfied')
```

Starting with the linear dataset:


```python
multicollinearity_assumption(linear_model, linear_X, linear_y, linear_feature_names)
```

    Assumption 3: Little to no multicollinearity among predictors
    


![png](output_29_1.png)


    Variance Inflation Factors (VIF)
    > 10: An indication that multicollinearity may be present
    > 100: Certain multicollinearity among the variables
    -------------------------------------
    X1: 1.022266152680125
    X2: 1.0225722553296535
    X3: 1.0310378514903915
    X4: 1.0133709178960577
    X5: 1.013079198380822
    X6: 1.019699482579504
    X7: 1.029942022560231
    X8: 1.033255775873013
    X9: 1.0249832417949483
    X10: 1.0136517315424105
    X11: 1.0177291796546448
    X12: 1.033627615412328
    X13: 1.0128771415883007
    
    0 cases of possible multicollinearity
    0 cases of definite multicollinearity
    
    Assumption satisfied
    

Everything looks peachy keen. Onto the Boston dataset:


```python
multicollinearity_assumption(boston_model, boston.data, boston.target, boston.feature_names)
```

    Assumption 3: Little to no multicollinearity among predictors
    


![png](output_31_1.png)


    Variance Inflation Factors (VIF)
    > 10: An indication that multicollinearity may be present
    > 100: Certain multicollinearity among the variables
    -------------------------------------
    CRIM: 2.0746257632525675
    ZN: 2.8438903527570782
    INDUS: 14.484283435031545
    CHAS: 1.1528909172683364
    NOX: 73.90221170812129
    RM: 77.93496867181426
    AGE: 21.38677358304778
    DIS: 14.699368125642422
    RAD: 15.154741587164747
    TAX: 61.226929320337554
    PTRATIO: 85.0273135204276
    B: 20.066007061121244
    LSTAT: 11.088865100659874
    
    10 cases of possible multicollinearity
    0 cases of definite multicollinearity
    
    Assumption possibly satisfied
    

This isn't quite as egregious as our normality assumption violation, but there is possible multicollinearity for most of the variables in this dataset.

### IV) No Autocorrelation of the error terms

This assumes no autocorrelation of the error terms. Autocorrelation being present typically indicates that we are missing some information that should be captured by the model.

A simple fix of adding lag variables can fix this problem.


```python
def autocorrelation_assumption(model, features, label):
    """
    Autocorrelation: Assumes that there is no autocorrelation in the residuals. If there is
                     autocorrelation, then there is a pattern that is not explained due to
                     the current value being dependent on the previous value.
                     This may be resolved by adding a lag variable of either the dependent
                     variable or some of the predictors.
    """
    from statsmodels.stats.stattools import durbin_watson
    print('Assumption 4: No Autocorrelation')
    print()
    
    # Calculating residuals for the Durbin Watson-tests
    df_results = calculate_residuals(model, features, label)

    print('\nPerforming Durbin-Watson Test')
    print('Values of 1.5 < d < 2.5 generally show that there is no autocorrelation in the data')
    print('0 to 2< is positive autocorrelation')
    print('>2 to 4 is negative autocorrelation')
    print('-------------------------------------')
    durbinWatson = durbin_watson(df_results['Residuals'])
    print('Durbin-Watson:', durbinWatson)
    if durbinWatson < 1.5:
        print('Signs of positive autocorrelation')
        print('\nAssumption not satisfied')
    elif durbinWatson > 2.5:
        print('Signs of negative autocorrelation')
        print('\nAssumption not satisfied')
    else:
        print('Little to no autocorrelation')
        print('\nAssumption satisfied')
```

Testing with our ideal dataset:


```python
autocorrelation_assumption(linear_model, linear_X, linear_y)
```

    Assumption 4: No Autocorrelation
    
    
    Performing Durbin-Watson Test
    Values of 1.5 < d < 2.5 generally show that there is no autocorrelation in the data
    0 to 2< is positive autocorrelation
    >2 to 4 is negative autocorrelation
    -------------------------------------
    Durbin-Watson: 1.94882623697
    Little to no autocorrelation
    
    Assumption satisfied
    

And with our Boston dataset:


```python
autocorrelation_assumption(boston_model, boston.data, boston.target)
```

    Assumption 4: No Autocorrelation
    
    
    Performing Durbin-Watson Test
    Values of 1.5 < d < 2.5 generally show that there is no autocorrelation in the data
    0 to 2< is positive autocorrelation
    >2 to 4 is negative autocorrelation
    -------------------------------------
    Durbin-Watson: 1.0713285604
    Signs of positive autocorrelation
    
    Assumption not satisfied
    

We're having signs of positive autocorrelation here. Adding lag variables could potentially fix this.

### V) Homoscedasticity 

This assumes homoscedasticity, or the same variance within our error terms. Heteroscedasticity, the violation of homoscedasticity, occurs when we don't have an even variance within the error terms. This is problematic because it means the standard errors are biased, therefore causing issues with the significance tests for coefficients. 

Heteroscedasticity (can you tell I like the *scedasticity* words?) can be solved either by using [weighted least squares regression](https://en.wikipedia.org/wiki/Least_squares#Weighted_least_squares) instead of the standard OLS or transforming either the dependent or highly skewed variables.


```python
def homoscedasticity_assumption(model, features, label):
    """
    Homoscedasticity: Assumes that the errors exhibit constant variance
    """
    print('Assumption 5: Homoscedasticity of Error Terms')
    print()
    
    print('Residuals should have relative constant variance')
        
    # Calculating residuals for the plot
    df_results = calculate_residuals(model, features, label)

    # Plotting the residuals
    ax = plt.subplot(111)
    plt.scatter(x=df_results.index, y=df_results.Residuals, alpha=0.5)
    plt.plot(np.repeat(0, df_results.index.max()), color='darkorange', linestyle='--')
    ax.spines['right'].set_visible(False)  # Removing the right spine
    ax.spines['top'].set_visible(False)  # Removing the top spine
    plt.title('Residuals')
    plt.show()  
```

Plotting the residuals of our ideal dataset:


```python
homoscedasticity_assumption(linear_model, linear_X, linear_y)
```

    Assumption 5: Homoscedasticity of Error Terms
    
    Residuals should have relative constant variance
    


![png](output_43_1.png)


There don't appear to be any obvious problems with that.

Next, looking at the residuals of the Boston dataset:


```python
homoscedasticity_assumption(boston_model, boston.data, boston.target)
```

    Assumption 5: Homoscedasticity of Error Terms
    
    Residuals should have relative constant variance
    


![png](output_45_1.png)


We can't see a uniform variance across our residuals, so this could potentially be problematic.

And there we have it! We can clearly see that a linear regression on the Boston dataset violates a number of assumptions which cause significant problems with the interpretation of the model itself. It's not uncommon for assumptions to be violated on real-world data, but it's important to check them so we can either fix them and/or be aware of the flaws in the model.

Here is a function for performing all of these assumption tests on a dataset:

## Code for full function


```python
def linear_regression_assumptions(features, label, feature_names=None):
    """
    Tests a linear regression on the model to see if assumptions are being met
    """
    from sklearn.linear_model import LinearRegression
    
    # Setting feature names to x1, x2, x3, etc. if they are not defined
    if feature_names is None:
        feature_names = ['X'+str(feature+1) for feature in range(features.shape[1])]
    
    print('Fitting linear regression')
    # Multi-threading if the dataset is a size where doing so is beneficial
    if features.shape[0] < 100000:
        model = LinearRegression(n_jobs=-1)
    else:
        model = LinearRegression()
        
    model.fit(features, label)
    
    # Returning linear regression R^2 and coefficients before performing diagnostics
    r2 = model.score(features, label)
    print('\nR^2:', r2)
    print('\nCoefficients')
    print('-------------------------------------')
    print('Intercept:', model.intercept_)
    
    for feature in range(len(model.coef_)):
        print('{0}: {1}'.format(feature_names[feature], model.coef_[feature]))

    print('\nPerforming linear regression assumption testing')
    
    # Creating predictions and calculating residuals for assumption tests
    predictions = model.predict(features)
    df_results = pd.DataFrame({'Actual': label, 'Predicted': predictions})
    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])

    
    def linear_assumption():
        """
        Linearity: Assumes there is a linear relationship between the predictors and
                   the response variable. If not, either a quadratic term or another
                   algorithm should be used.
        """
        print('\n=======================================================================================')
        print('Assumption 1: Linear Relationship between the Target and the Features')
        
        print('Checking with a scatter plot of actual vs. predicted. Predictions should follow the diagonal line.')
        
        # Plotting the actual vs predicted values
        sns.lmplot(x='Actual', y='Predicted', data=df_results, fit_reg=False, size=5)
        
        # Plotting the diagonal line
        line_coords = np.arange(df_results.min().min(), df_results.max().max())
        plt.plot(line_coords, line_coords,  # X and y points
                 color='darkorange', linestyle='--')
        plt.title('Actual vs. Predicted')
        plt.show()
        
        
    def multivariate_normal_assumption(p_value_thresh=0.05):
        """
        Normality: Assumes that the predictors have normal distributions. If they are not normal,
                   a non-linear transformation like a log transformation or box-cox transformation
                   can be performed on the non-normal variable.
        """
        from statsmodels.stats.diagnostic import normal_ad
        print('\n=======================================================================================')
        print('Assumption 2: All variables are multivariate normal')
        print('Using the Anderson-Darling test for normal distribution')
        print('p-values from the test - below 0.05 generally means normality:')
        print()
        non_normal_variables = 0
        
        # Performing the Anderson-Darling test on each variable to test for normality
        for feature in range(features.shape[1]):
            p_value = normal_ad(features[:, feature])[1]
            
            # Adding to total count of non-normality if p-value exceeds threshold
            if p_value > p_value_thresh:
                non_normal_variables += 1
            
            # Printing p-values from the test
            print('{0}: {1}'.format(feature_names[feature], p_value))
                    
        print('\n{0} non-normal variables'.format(non_normal_variables))
        print()

        if non_normal_variables == 0:
            print('Assumption satisfied')
        else:
            print('Assumption not satisfied')
        
        
    def multicollinearity_assumption():
        """
        Multicollinearity: Assumes that predictors are not correlated with each other. If there is
                           correlation among the predictors, then either remove prepdictors with high
                           Variance Inflation Factor (VIF) values or perform dimensionality reduction
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        print('\n=======================================================================================')
        print('Assumption 3: Little to no multicollinearity among predictors')
        
        # Plotting the heatmap
        ax = plt.subplot(111)
        sns.heatmap(pd.DataFrame(features, columns=feature_names).corr())
        plt.show()
        
        print('Variance Inflation Factors (VIF)')
        print('> 10: An indication that multicollinearity may be present')
        print('> 100: Certain multicollinearity among the variables')
        print('-------------------------------------')
       
        # Gathering the VIF for each variable
        VIF = [variance_inflation_factor(features, i) for i in range(features.shape[1])]
        for idx, vif in enumerate(VIF):
            print('{0}: {1}'.format(feature_names[idx], vif))
        
        # Gathering and printing total cases of possible or definite multicollinearity
        possible_multicollinearity = sum([1 for vif in VIF if vif > 10])
        definite_multicollinearity = sum([1 for vif in VIF if vif > 100])
        print()
        print('{0} cases of possible multicollinearity'.format(possible_multicollinearity))
        print('{0} cases of definite multicollinearity'.format(definite_multicollinearity))
        print()

        if definite_multicollinearity == 0:
            if possible_multicollinearity == 0:
                print('Assumption satisfied')
            else:
                print('Assumption possibly satisfied')
        else:
            print('Assumption not satisfied')
        
        
    def autocorrelation_assumption():
        """
        Autocorrelation: Assumes that there is no autocorrelation in the residuals. If there is
                         autocorrelation, then there is a pattern that is not explained due to
                         the current value being dependent on the previous value.
                         This may be resolved by adding a lag variable of either the dependent
                         variable or some of the predictors.
        """
        from statsmodels.stats.stattools import durbin_watson
        print('\n=======================================================================================')
        print('Assumption 4: No Autocorrelation')
        print('\nPerforming Durbin-Watson Test')
        print('Values of 1.5 < d < 2.5 generally show that there is no autocorrelation in the data')
        print('0 to 2< is positive autocorrelation')
        print('>2 to 4 is negative autocorrelation')
        print('-------------------------------------')
        durbinWatson = durbin_watson(df_results['Residuals'])
        print('Durbin-Watson:', durbinWatson)
        if durbinWatson < 1.5:
            print('Signs of positive autocorrelation')
            print('\nAssumption not satisfied')
        elif durbinWatson > 2.5:
            print('Signs of negative autocorrelation')
            print('\nAssumption not satisfied')
        else:
            print('Little to no autocorrelation')
            print('\nAssumption satisfied')

            
    def homoscedasticity_assumption():
        """
        Homoscedasticity: Assumes that the errors exhibit constant variance
        """
        print('\n=======================================================================================')
        print('Assumption 5: Homoscedasticity of Error Terms')
        print('Residuals should have relative constant variance')
        
        # Plotting the residuals
        ax = plt.subplot(111)
        plt.scatter(x=df_results.index, y=df_results.Residuals, alpha=0.5)
        plt.plot(np.repeat(0, df_results.index.max()), color='darkorange', linestyle='--')
        ax.spines['right'].set_visible(False)  # Removing the right spine
        ax.spines['top'].set_visible(False)  # Removing the top spine
        plt.title('Residuals')
        plt.show()  
        
        
    linear_assumption()
    multivariate_normal_assumption()
    multicollinearity_assumption()
    autocorrelation_assumption()
    homoscedasticity_assumption()
```
