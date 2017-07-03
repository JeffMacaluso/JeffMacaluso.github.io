---
layout: post
title:  "Competition: The Ultimate Student Hunt"
date:   2016-10-19 20:04:11 -0500
categories: jekyll update
---
[Github link to notebook](http:github.com/JeffMacaluso/Competitions/blob/master/Hackathon%20-%20The%20Ultimate%20Student%20Hunt.ipynb)

# Hackathon - The Ultimate Student Hunt

Analytics Vidhya hosted a student-only [hackathon](https://datahack.analyticsvidhya.com/contest/the-ultimate-student-hunt/#activity_id) over predicting the number of visitors to parks given a set amount of variables.  This was my first data competition, and proved to be a huge learning experience.

Here's a brief overview of the data from the contest rules listed on the website:

- **ID**: Unique ID
- **Park_ID**: Unique ID for Parks
- **Date**: Calendar Date
- **Direction_Of_Wind**: Direction of winds in degrees
- **Average_Breeze_Speed**: Daily average Breeze speed
- **Max_Breeze_Speed**: Daily maximum Breeze speed
- **Min_Breeze_Speed**: Daily minimum Breeze speed
- **Var1**: A continuous feature
- **Average_Atmospheric_Pressure**: Daily average atmospheric pressure
- **Max_Atmospheric_Pressure**: Daily maximum atmospheric pressure
- **Min_Atmospheric_Pressure**: Daily minimum atmospheric pressure
- **Min_Ambient_Pollution**: Daily minimum Ambient pollution
- **Max_Ambient_Pollution**: Daily maximum Ambient pollution
- **Average_Moisture_In_Park**: Daily average moisture
- **Max_Moisture_In_Park**: Daily maximum moisture
- **Min_Moisture_In_Park**: Daily minimum moisture
- **Location_Type**: Location Type (1/2/3/4)
- **Footfall**: The target variable, daily Footfall

## Summary

This problem involved predicting the number of visitors (footfall) to parks on a given day with given conditions, which ultimately makes it a time series problem.  Specifically, it provided ten years as a training set, and five years for the test set.  This notebook is an annotated version of my final submission which ranked 13th on the leaderboard.  On a side note, the hackathon was only open for nine days, so there is of course a lot of room for improvement in this notebook. 

My process for this hackathon was as follows:
1. **Initial Exploration**: All data projects should begin with an initial exploration to understand the data itself.  I initially used the [pandas profiling](https://github.com/JosPolfliet/pandas-profiling) package, but excluded it from my final submission since it generates a lengthy report.  I left both a quick df.describe() and df.head() to showcase summary statistics and an example of the data.
2. **Outliers**: I created boxplots to look for outliers visually.  This can be done mathematically when you are more familiar with the data (using methods such as interquartile ranges), but the distributions of the variables produced a significant amount of data points that would've been considered outliers with this methodology.
3. **Missing Values**: I first sorted the dataframe by date and park ID, then used the [msno package](https://github.com/ResidentMario/missingno) to visually examine missing values.  After seeing fairly regular trends of missing values, I plotted histograms of the missing values by park IDs to see if I could fill them by linearly interpolating.  After seeing that certain park IDs were completely missing some values, I built random forest models to impute them.  This is a brute-force method that is CPU intensive, but was a trade-off for the limited time frame. 
4. **Feature Engineering**: This was almost non-existant in this competition due to the anonymity of the data.  I used daily and weekly averages of the individual variables in both the end model and missing value imputation models.
5. **Model Building**: I initially started by creating three models using random forests, gradient boosted trees, and AdaBoost.  The gradient boosted trees model outperformed the other two, so I stuck with that and scrapped the other two.
6. **Hyperparameter Tuning**: This was my first time using gradient boosted trees, so I took a trial-and-error approach by adjusting various parameters and running them through cross validation to see how differently they performed.  I found that just adjusting the number of trees and max depth obtained the best results in this situation.
7. **Validation**: I used both a holdout cross-validation and k-folds (with 10 folds) to check for overfitting.  The hackathon also had a "solution checker" for your predicted values (specifically for the first two years of the test set - the final score of the competition was on the full five years of the test set, so it is very important to not overfit) that provided a score, which I used in combination with the cross validation results.

Here is the annotated code for my final submission:


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import cross_val_score, train_test_split, KFold
from sklearn.preprocessing import Imputer

# For exploratory data analysis
import missingno as msno  # Visualizes missing values

%matplotlib inline
```


```python
df = pd.read_csv('crime.csv')  # Training set - ignore the name

df_test = pd.read_csv('Test_pyI9Owa.csv')  # Testing set
```

## Exploratory Data Analysis


```python
df.describe()
```




<div style="overflow-x:auto;">
<style>
table {
        margin-left: auto;
        margin-right: auto;
        border: none;
        border-collapse: collapse;
        border-spacing: 0;
        color: @rendered_html_border_color;
        font-size: 12px;
        table-layout: fixed;
    }
    thead {
        border-bottom: 1px solid @rendered_html_border_color;
        vertical-align: bottom;
    }
    tr, th, td {
        text-align: right;
        vertical-align: middle;
        padding: 0.5em 0.5em;
        line-height: normal;
        white-space: normal;
        max-width: none;
        border: none;
    }
    th {
        font-weight: bold;
    }
    tbody tr:nth-child(odd) {
        background: #f5f5f5;
    }
    tbody tr:hover {
        background: rgba(66, 165, 245, 0.2);
    }
    * + table {margin-top: 1em;}

    p {text-align: left;}
* + p {margin-top: 1em;}

td, th {
    text-align: center;
    padding: 8px;
}
</style>
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Park_ID</th>
      <th>Direction_Of_Wind</th>
      <th>Average_Breeze_Speed</th>
      <th>Max_Breeze_Speed</th>
      <th>Min_Breeze_Speed</th>
      <th>Var1</th>
      <th>Average_Atmospheric_Pressure</th>
      <th>Max_Atmospheric_Pressure</th>
      <th>Min_Atmospheric_Pressure</th>
      <th>Min_Ambient_Pollution</th>
      <th>Max_Ambient_Pollution</th>
      <th>Average_Moisture_In_Park</th>
      <th>Max_Moisture_In_Park</th>
      <th>Min_Moisture_In_Park</th>
      <th>Location_Type</th>
      <th>Footfall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.145390e+05</td>
      <td>114539.000000</td>
      <td>110608.000000</td>
      <td>110608.000000</td>
      <td>110603.000000</td>
      <td>110605.000000</td>
      <td>106257.000000</td>
      <td>74344.000000</td>
      <td>74344.000000</td>
      <td>74344.000000</td>
      <td>82894.000000</td>
      <td>82894.000000</td>
      <td>114499.000000</td>
      <td>114499.000000</td>
      <td>114499.000000</td>
      <td>114539.000000</td>
      <td>114539.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.517595e+06</td>
      <td>25.582596</td>
      <td>179.587146</td>
      <td>34.255340</td>
      <td>51.704297</td>
      <td>17.282553</td>
      <td>18.802545</td>
      <td>8331.545949</td>
      <td>8356.053468</td>
      <td>8305.692510</td>
      <td>162.806138</td>
      <td>306.555698</td>
      <td>248.008970</td>
      <td>283.917082</td>
      <td>202.355331</td>
      <td>2.630720</td>
      <td>1204.217192</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.189083e+05</td>
      <td>8.090592</td>
      <td>85.362934</td>
      <td>17.440065</td>
      <td>22.068301</td>
      <td>14.421844</td>
      <td>38.269851</td>
      <td>80.943971</td>
      <td>76.032983</td>
      <td>87.172258</td>
      <td>90.869627</td>
      <td>38.188020</td>
      <td>28.898084</td>
      <td>15.637930</td>
      <td>46.365728</td>
      <td>0.967435</td>
      <td>248.385651</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.311712e+06</td>
      <td>12.000000</td>
      <td>1.000000</td>
      <td>3.040000</td>
      <td>7.600000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7982.000000</td>
      <td>8037.000000</td>
      <td>7890.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>102.000000</td>
      <td>141.000000</td>
      <td>48.000000</td>
      <td>1.000000</td>
      <td>310.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.414820e+06</td>
      <td>18.000000</td>
      <td>111.000000</td>
      <td>22.040000</td>
      <td>38.000000</td>
      <td>7.600000</td>
      <td>0.000000</td>
      <td>8283.000000</td>
      <td>8311.000000</td>
      <td>8252.000000</td>
      <td>80.000000</td>
      <td>288.000000</td>
      <td>231.000000</td>
      <td>279.000000</td>
      <td>171.000000</td>
      <td>2.000000</td>
      <td>1026.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.517039e+06</td>
      <td>26.000000</td>
      <td>196.000000</td>
      <td>30.400000</td>
      <td>45.600000</td>
      <td>15.200000</td>
      <td>0.830000</td>
      <td>8335.000000</td>
      <td>8358.000000</td>
      <td>8311.000000</td>
      <td>180.000000</td>
      <td>316.000000</td>
      <td>252.000000</td>
      <td>288.000000</td>
      <td>207.000000</td>
      <td>3.000000</td>
      <td>1216.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.619624e+06</td>
      <td>33.000000</td>
      <td>239.000000</td>
      <td>42.560000</td>
      <td>60.800000</td>
      <td>22.800000</td>
      <td>21.580000</td>
      <td>8382.000000</td>
      <td>8406.000000</td>
      <td>8362.000000</td>
      <td>244.000000</td>
      <td>336.000000</td>
      <td>270.000000</td>
      <td>294.000000</td>
      <td>237.000000</td>
      <td>3.000000</td>
      <td>1402.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.725639e+06</td>
      <td>39.000000</td>
      <td>360.000000</td>
      <td>154.280000</td>
      <td>212.800000</td>
      <td>129.200000</td>
      <td>1181.090000</td>
      <td>8588.000000</td>
      <td>8601.000000</td>
      <td>8571.000000</td>
      <td>348.000000</td>
      <td>356.000000</td>
      <td>300.000000</td>
      <td>300.000000</td>
      <td>300.000000</td>
      <td>4.000000</td>
      <td>1925.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.head()
```




<div style="overflow-x:auto;">
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Park_ID</th>
      <th>Date</th>
      <th>Direction_Of_Wind</th>
      <th>Average_Breeze_Speed</th>
      <th>Max_Breeze_Speed</th>
      <th>Min_Breeze_Speed</th>
      <th>Var1</th>
      <th>Average_Atmospheric_Pressure</th>
      <th>Max_Atmospheric_Pressure</th>
      <th>Min_Atmospheric_Pressure</th>
      <th>Min_Ambient_Pollution</th>
      <th>Max_Ambient_Pollution</th>
      <th>Average_Moisture_In_Park</th>
      <th>Max_Moisture_In_Park</th>
      <th>Min_Moisture_In_Park</th>
      <th>Location_Type</th>
      <th>Footfall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3311712</td>
      <td>12</td>
      <td>01-09-1990</td>
      <td>194.0</td>
      <td>37.24</td>
      <td>60.8</td>
      <td>15.2</td>
      <td>92.1300</td>
      <td>8225.0</td>
      <td>8259.0</td>
      <td>8211.0</td>
      <td>92.0</td>
      <td>304.0</td>
      <td>255.0</td>
      <td>288.0</td>
      <td>222.0</td>
      <td>3</td>
      <td>1406</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3311812</td>
      <td>12</td>
      <td>02-09-1990</td>
      <td>285.0</td>
      <td>32.68</td>
      <td>60.8</td>
      <td>7.6</td>
      <td>14.1100</td>
      <td>8232.0</td>
      <td>8280.0</td>
      <td>8205.0</td>
      <td>172.0</td>
      <td>332.0</td>
      <td>252.0</td>
      <td>297.0</td>
      <td>204.0</td>
      <td>3</td>
      <td>1409</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3311912</td>
      <td>12</td>
      <td>03-09-1990</td>
      <td>319.0</td>
      <td>43.32</td>
      <td>60.8</td>
      <td>15.2</td>
      <td>35.6900</td>
      <td>8321.0</td>
      <td>8355.0</td>
      <td>8283.0</td>
      <td>236.0</td>
      <td>292.0</td>
      <td>219.0</td>
      <td>279.0</td>
      <td>165.0</td>
      <td>3</td>
      <td>1386</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3312012</td>
      <td>12</td>
      <td>04-09-1990</td>
      <td>297.0</td>
      <td>25.84</td>
      <td>38.0</td>
      <td>7.6</td>
      <td>0.0249</td>
      <td>8379.0</td>
      <td>8396.0</td>
      <td>8358.0</td>
      <td>272.0</td>
      <td>324.0</td>
      <td>225.0</td>
      <td>261.0</td>
      <td>192.0</td>
      <td>3</td>
      <td>1365</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3312112</td>
      <td>12</td>
      <td>05-09-1990</td>
      <td>207.0</td>
      <td>28.88</td>
      <td>45.6</td>
      <td>7.6</td>
      <td>0.8300</td>
      <td>8372.0</td>
      <td>8393.0</td>
      <td>8335.0</td>
      <td>236.0</td>
      <td>332.0</td>
      <td>234.0</td>
      <td>273.0</td>
      <td>183.0</td>
      <td>3</td>
      <td>1413</td>
    </tr>
  </tbody>
</table>
</div>



### Outliers

We'll do our outlier detection visually with box plots.  Rather than determining outliers mathematically (such as using the interquartile range), we'll simply look for any points that aren't contiguous.


```python
df_box = df.drop(['ID', 'Park_ID', 'Average_Atmospheric_Pressure', 'Max_Atmospheric_Pressure'
                  , 'Min_Atmospheric_Pressure', 'Footfall', 'Date'], axis = 1)
plt.figure(figsize = (20,10))
sns.boxplot(data=df_box)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1774dfe6630>



<img src="https://github.com/JeffMacaluso/JeffMacaluso.github.io/blob/master/_posts/Hackathon_files/Hackathon_8_1.png?raw=true">


Var1 seems to potentially have outliers, but since it is undefined, it is difficult to determine if these are anomalies or noisy/incorrect data.  We'll leave them for now.


```python
df_box = df[['Average_Atmospheric_Pressure', 'Max_Atmospheric_Pressure'
                  , 'Min_Atmospheric_Pressure']]
plt.figure(figsize = (20,10))
sns.boxplot(data=df_box)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1774f62a518>



<img src="https://github.com/JeffMacaluso/JeffMacaluso.github.io/blob/master/_posts/Hackathon_files/Hackathon_10_1.png?raw=true">


Max atmospheric pressure (and by result, average atmospheric pressure) have a few non-contiguous values, but they don't seem egregious enough to deal with for the time being.


```python
# Converting date field to datetime and extracting date components
df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = pd.DatetimeIndex(df['Date']).year
df['Month'] = pd.DatetimeIndex(df['Date']).month
df['Day'] = pd.DatetimeIndex(df['Date']).day
df['Week'] = pd.DatetimeIndex(df['Date']).week
df['WeekDay'] = pd.DatetimeIndex(df['Date']).dayofweek


# Repeating for the test set
df_test['Date'] = pd.to_datetime(df_test['Date'])

df_test['Year'] = pd.DatetimeIndex(df_test['Date']).year
df_test['Month'] = pd.DatetimeIndex(df_test['Date']).month
df_test['Day'] = pd.DatetimeIndex(df_test['Date']).day
df_test['Week'] = pd.DatetimeIndex(df_test['Date']).week
df_test['WeekDay'] = pd.DatetimeIndex(df_test['Date']).dayofweek


# Lastly, combining to use for building models to predict missing predictors 
df_full = df.append(df_test)
```

### Missing Values

Since this is ultimately a time series problem, we'll begin with sorting the values.  Then, I'm going to use a [useful package](https://github.com/ResidentMario/missingno) for visualizing missing values.


```python
# Sorting by date and park
df = df.sort_values(['Date', 'Park_ID'], ascending=[1, 1])
df_full = df_full.sort_values(['Date', 'Park_ID'], ascending=[1, 1])
```


```python
# Visualizing missing values
msno.matrix(df_full)
```

<img src="https://github.com/JeffMacaluso/JeffMacaluso.github.io/blob/master/_posts/Hackathon_files/Hackathon_15_0.png?raw=true">



```python
# Checking which Park IDs missing values occur in
plt.subplot(221)
df_full[df_full['Direction_Of_Wind'].isnull() == True]['Park_ID'].hist()
plt.subplot(222)
df_full[df_full['Average_Breeze_Speed'].isnull() == True]['Park_ID'].hist()
plt.subplot(223)
df_full[df_full['Max_Breeze_Speed'].isnull() == True]['Park_ID'].hist()
plt.subplot(224)
df_full[df_full['Min_Breeze_Speed'].isnull() == True]['Park_ID'].hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1774f091550>



<img src="https://github.com/JeffMacaluso/JeffMacaluso.github.io/blob/master/_posts/Hackathon_files/Hackathon_16_1.png?raw=true">



```python
df_full[df_full['Var1'].isnull() == True]['Park_ID'].hist()
plt.title('Var1 Missing Park IDs')
```




    <matplotlib.text.Text at 0x1774f6d1470>



<img src="https://github.com/JeffMacaluso/JeffMacaluso.github.io/blob/master/_posts/Hackathon_files/Hackathon_17_1.png?raw=true">



```python
plt.subplot(221)
df_full[df_full['Average_Atmospheric_Pressure'].isnull() == True]['Park_ID'].hist()
plt.subplot(222)
df_full[df_full['Max_Atmospheric_Pressure'].isnull() == True]['Park_ID'].hist()
plt.subplot(223)
df_full[df_full['Min_Atmospheric_Pressure'].isnull() == True]['Park_ID'].hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1774f78fb70>



<img src="https://github.com/JeffMacaluso/JeffMacaluso.github.io/blob/master/_posts/Hackathon_files/Hackathon_18_1.png?raw=true">



```python
plt.subplot(221)
df_full[df_full['Max_Ambient_Pollution'].isnull() == True]['Park_ID'].hist()
plt.subplot(222)
df_full[df_full['Min_Ambient_Pollution'].isnull() == True]['Park_ID'].hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1774e0245c0>



<img src="https://github.com/JeffMacaluso/JeffMacaluso.github.io/blob/master/_posts/Hackathon_files/Hackathon_19_1.png?raw=true">


We can see here that most missing values are re-occurring in the same parks.  This means we can't interpolate our missing values, and filling with the mean/median/mode is to over-generalized, so we should build models to predict our missing values.

Msno has a heatmap that shows the co-occurrance of missing values, which will be helpful in determining how to construct our models.


```python
# Co-occurrence of missing values
msno.heatmap(df)
```

<img src="https://github.com/JeffMacaluso/JeffMacaluso.github.io/blob/master/_posts/Hackathon_files/Hackathon_21_0.png?raw=true">


## Feature Engineering

#### Daily & Weekly Averages
Before building models to predict the missing values, we'll begin with calculating daily and weekly averages across all parks to assist wth our predictions.

There is a lot of repetition here due to re-running the same code for both the training and testing set.

*Training Set*


```python
# Gathering the daily averages of predictors

# Wind
avg_daily_breeze = df['Average_Breeze_Speed'].groupby(df['Date']).mean().to_frame().reset_index()  # Group by day
avg_daily_breeze.columns = ['Date', 'Avg_Daily_Breeze']  # Renaming the columns for the join
df = df.merge(avg_daily_breeze, how = 'left')  # Joining onto the original dataframe

max_daily_breeze = df['Max_Breeze_Speed'].groupby(df['Date']).mean().to_frame().reset_index()
max_daily_breeze.columns = ['Date', 'Max_Daily_Breeze']
df = df.merge(max_daily_breeze, how = 'left')

min_daily_breeze = df['Min_Breeze_Speed'].groupby(df['Date']).mean().to_frame().reset_index()
min_daily_breeze.columns = ['Date', 'Min_Daily_Breeze']
df = df.merge(min_daily_breeze, how = 'left')


# Var1
var1_daily = df['Var1'].groupby(df['Date']).mean().to_frame().reset_index()
var1_daily.columns = ['Date', 'Var1_Daily']
df = df.merge(var1_daily, how = 'left')


# Atmosphere & Pollution
avg_daily_atmo = df['Average_Atmospheric_Pressure'].groupby(df['Date']).mean().to_frame().reset_index()
avg_daily_atmo.columns = ['Date', 'Avg_Daily_Atmosphere']
df = df.merge(avg_daily_atmo, how = 'left')

max_daily_atmo = df['Max_Atmospheric_Pressure'].groupby(df['Date']).mean().to_frame().reset_index()
max_daily_atmo.columns = ['Date', 'Max_Daily_Atmosphere']
df = df.merge(max_daily_atmo, how = 'left')

min_daily_atmo = df['Min_Atmospheric_Pressure'].groupby(df['Date']).mean().to_frame().reset_index()
min_daily_atmo.columns = ['Date', 'Min_Daily_Atmosphere']
df = df.merge(min_daily_atmo, how = 'left')

max_daily_pollution = df['Max_Ambient_Pollution'].groupby(df['Date']).mean().to_frame().reset_index()
max_daily_pollution.columns = ['Date', 'Max_Daily_Pollution']
df = df.merge(max_daily_pollution, how = 'left')

min_daily_pollution = df['Min_Ambient_Pollution'].groupby(df['Date']).mean().to_frame().reset_index()
min_daily_pollution.columns = ['Date', 'Min_Daily_Pollution']
df = df.merge(min_daily_pollution, how = 'left')


# Moisture
avg_daily_moisture = df['Average_Moisture_In_Park'].groupby(df['Date']).mean().to_frame().reset_index()
avg_daily_moisture.columns = ['Date', 'Avg_Daily_moisture']
df = df.merge(avg_daily_moisture, how = 'left')

max_daily_moisture = df['Max_Moisture_In_Park'].groupby(df['Date']).mean().to_frame().reset_index()
max_daily_moisture.columns = ['Date', 'Max_Daily_moisture']
df = df.merge(max_daily_moisture, how = 'left')

min_daily_moisture = df['Min_Moisture_In_Park'].groupby(df['Date']).mean().to_frame().reset_index()
min_daily_moisture.columns = ['Date', 'Min_Daily_moisture']
df = df.merge(min_daily_moisture, how = 'left')
```


```python
# Repeating with weekly averages of predictors

# Wind
avg_weekly_breeze = df['Average_Breeze_Speed'].groupby((df['Year'], df['Week'])).mean().to_frame().reset_index()
avg_weekly_breeze.columns = ['Year', 'Week', 'Avg_Weekly_Breeze']
df = df.merge(avg_weekly_breeze, how = 'left')

max_weekly_breeze = df['Max_Breeze_Speed'].groupby((df['Year'], df['Week'])).mean().to_frame().reset_index()
max_weekly_breeze.columns = ['Year', 'Week', 'Max_Weekly_Breeze']
df = df.merge(max_weekly_breeze, how = 'left')

min_weekly_breeze = df['Min_Breeze_Speed'].groupby((df['Year'], df['Week'])).mean().to_frame().reset_index()
min_weekly_breeze.columns = ['Year', 'Week', 'Min_Weekly_Breeze']
df = df.merge(min_weekly_breeze, how = 'left')


# Var 1
var1_weekly = df['Var1'].groupby((df['Year'], df['Week'])).mean().to_frame().reset_index()
var1_weekly.columns = ['Year', 'Week', 'Var1_Weekly']
df = df.merge(var1_weekly, how = 'left')


# Atmosphere & Pollution
avg_weekly_atmo = df['Average_Atmospheric_Pressure'].groupby((df['Year'], df['Week'])).mean().to_frame().reset_index()
avg_weekly_atmo.columns = ['Year', 'Week', 'Avg_Weekly_Atmosphere']
df = df.merge(avg_weekly_atmo, how = 'left')

max_weekly_atmo = df['Max_Atmospheric_Pressure'].groupby((df['Year'], df['Week'])).mean().to_frame().reset_index()
max_weekly_atmo.columns = ['Year', 'Week', 'Max_Weekly_Atmosphere']
df = df.merge(max_weekly_atmo, how = 'left')

min_weekly_atmo = df['Min_Atmospheric_Pressure'].groupby((df['Year'], df['Week'])).mean().to_frame().reset_index()
min_weekly_atmo.columns = ['Year', 'Week', 'Min_Weekly_Atmosphere']
df = df.merge(min_weekly_atmo, how = 'left')

max_weekly_pollution = df['Max_Ambient_Pollution'].groupby((df['Year'], df['Week'])).mean().to_frame().reset_index()
max_weekly_pollution.columns = ['Year', 'Week', 'Max_Weekly_Pollution']
df = df.merge(max_weekly_pollution, how = 'left')

min_weekly_pollution = df['Min_Ambient_Pollution'].groupby((df['Year'], df['Week'])).mean().to_frame().reset_index()
min_weekly_pollution.columns = ['Year', 'Week', 'Min_Weekly_Pollution']
df = df.merge(min_weekly_pollution, how = 'left')


# Moisture
avg_weekly_moisture = df['Average_Moisture_In_Park'].groupby((df['Year'], df['Week'])).mean().to_frame().reset_index()
avg_weekly_moisture.columns = ['Year', 'Week', 'Avg_Weekly_Moisture']
df = df.merge(avg_weekly_moisture, how = 'left')

max_weekly_moisture = df['Max_Moisture_In_Park'].groupby((df['Year'], df['Week'])).mean().to_frame().reset_index()
max_weekly_moisture.columns = ['Year', 'Week', 'Max_Weekly_Moisture']
df = df.merge(max_weekly_moisture, how = 'left')

min_weekly_moisture = df['Min_Moisture_In_Park'].groupby((df['Year'], df['Week'])).mean().to_frame().reset_index()
min_weekly_moisture.columns = ['Year', 'Week', 'Min_Weekly_Moisture']
df = df.merge(min_weekly_moisture, how = 'left')
```

*Testing Set*


```python
# Gathering the daily averages of predictors

# Wind
avg_daily_breeze = df_test['Average_Breeze_Speed'].groupby(df_test['Date']).mean().to_frame().reset_index()
avg_daily_breeze.columns = ['Date', 'Avg_Daily_Breeze']
df_test = df_test.merge(avg_daily_breeze, how = 'left')

max_daily_breeze = df_test['Max_Breeze_Speed'].groupby(df_test['Date']).mean().to_frame().reset_index()
max_daily_breeze.columns = ['Date', 'Max_Daily_Breeze']
df_test = df_test.merge(max_daily_breeze, how = 'left')

min_daily_breeze = df_test['Min_Breeze_Speed'].groupby(df_test['Date']).mean().to_frame().reset_index()
min_daily_breeze.columns = ['Date', 'Min_Daily_Breeze']
df_test = df_test.merge(min_daily_breeze, how = 'left')


# Var1
var1_daily = df_test['Var1'].groupby(df_test['Date']).mean().to_frame().reset_index()
var1_daily.columns = ['Date', 'Var1_Daily']
df_test = df_test.merge(var1_daily, how = 'left')


# Atmosphere & Pollution
avg_daily_atmo = df_test['Average_Atmospheric_Pressure'].groupby(df_test['Date']).mean().to_frame().reset_index()
avg_daily_atmo.columns = ['Date', 'Avg_Daily_Atmosphere']
df_test = df_test.merge(avg_daily_atmo, how = 'left')
                        
max_daily_atmo = df_test['Max_Atmospheric_Pressure'].groupby(df_test['Date']).mean().to_frame().reset_index()
max_daily_atmo.columns = ['Date', 'Max_Daily_Atmosphere']
df_test = df_test.merge(max_daily_atmo, how = 'left')
                        
min_daily_atmo = df_test['Min_Atmospheric_Pressure'].groupby(df_test['Date']).mean().to_frame().reset_index()
min_daily_atmo.columns = ['Date', 'Min_Daily_Atmosphere']
df_test = df_test.merge(min_daily_atmo, how = 'left')
                        
max_daily_pollution = df_test['Max_Ambient_Pollution'].groupby(df_test['Date']).mean().to_frame().reset_index()
max_daily_pollution.columns = ['Date', 'Max_Daily_Pollution']
df_test = df_test.merge(max_daily_pollution, how = 'left')
                        
min_daily_pollution = df_test['Min_Ambient_Pollution'].groupby(df_test['Date']).mean().to_frame().reset_index()
min_daily_pollution.columns = ['Date', 'Min_Daily_Pollution']
df_test = df_test.merge(min_daily_pollution, how = 'left')


# Moisture
avg_daily_moisture = df_test['Average_Moisture_In_Park'].groupby(df_test['Date']).mean().to_frame().reset_index()
avg_daily_moisture.columns = ['Date', 'Avg_Daily_moisture']
df_test = df_test.merge(avg_daily_moisture, how = 'left')
                        
max_daily_moisture = df_test['Max_Moisture_In_Park'].groupby(df_test['Date']).mean().to_frame().reset_index()
max_daily_moisture.columns = ['Date', 'Max_Daily_moisture']
df_test = df_test.merge(max_daily_moisture, how = 'left')
                        
min_daily_moisture = df_test['Min_Moisture_In_Park'].groupby(df_test['Date']).mean().to_frame().reset_index()
min_daily_moisture.columns = ['Date', 'Min_Daily_moisture']
df_test = df_test.merge(min_daily_moisture, how = 'left')
```


```python
# Repeating with weekly averages of predictors

# Wind
avg_weekly_breeze = df_test['Average_Breeze_Speed'].groupby((df_test['Year'], df_test['Week'])).mean().to_frame().reset_index()
avg_weekly_breeze.columns = ['Year', 'Week', 'Avg_Weekly_Breeze']
df_test = df_test.merge(avg_weekly_breeze, how = 'left')

max_weekly_breeze = df_test['Max_Breeze_Speed'].groupby((df_test['Year'], df_test['Week'])).mean().to_frame().reset_index()
max_weekly_breeze.columns = ['Year', 'Week', 'Max_Weekly_Breeze']
df_test = df_test.merge(max_weekly_breeze, how = 'left')

min_weekly_breeze = df_test['Min_Breeze_Speed'].groupby((df_test['Year'], df_test['Week'])).mean().to_frame().reset_index()
min_weekly_breeze.columns = ['Year', 'Week', 'Min_Weekly_Breeze']
df_test = df_test.merge(min_weekly_breeze, how = 'left')


# Var 1
var1_weekly = df_test['Var1'].groupby((df_test['Year'], df_test['Week'])).mean().to_frame().reset_index()
var1_weekly.columns = ['Year', 'Week', 'Var1_Weekly']
df_test = df_test.merge(var1_weekly, how = 'left')


# Atmosphere & Pollution
avg_weekly_atmo = df_test['Average_Atmospheric_Pressure'].groupby((df_test['Year'], df_test['Week'])).mean().to_frame().reset_index()
avg_weekly_atmo.columns = ['Year', 'Week', 'Avg_Weekly_Atmosphere']
df_test = df_test.merge(avg_weekly_atmo, how = 'left')

max_weekly_atmo = df_test['Max_Atmospheric_Pressure'].groupby((df_test['Year'], df_test['Week'])).mean().to_frame().reset_index()
max_weekly_atmo.columns = ['Year', 'Week', 'Max_Weekly_Atmosphere']
df_test = df_test.merge(max_weekly_atmo, how = 'left')

min_weekly_atmo = df_test['Min_Atmospheric_Pressure'].groupby((df_test['Year'], df_test['Week'])).mean().to_frame().reset_index()
min_weekly_atmo.columns = ['Year', 'Week', 'Min_Weekly_Atmosphere']
df_test = df_test.merge(min_weekly_atmo, how = 'left')

max_weekly_pollution = df_test['Max_Ambient_Pollution'].groupby((df_test['Year'], df_test['Week'])).mean().to_frame().reset_index()
max_weekly_pollution.columns = ['Year', 'Week', 'Max_Weekly_Pollution']
df_test = df_test.merge(max_weekly_pollution, how = 'left')

min_weekly_pollution = df_test['Min_Ambient_Pollution'].groupby((df_test['Year'], df_test['Week'])).mean().to_frame().reset_index()
min_weekly_pollution.columns = ['Year', 'Week', 'Min_Weekly_Pollution']
df_test = df_test.merge(min_weekly_pollution, how = 'left')


# Moisture
avg_weekly_moisture = df_test['Average_Moisture_In_Park'].groupby((df_test['Year'], df_test['Week'])).mean().to_frame().reset_index()
avg_weekly_moisture.columns = ['Year', 'Week', 'Avg_Weekly_Moisture']
df_test = df_test.merge(avg_weekly_moisture, how = 'left')

max_weekly_moisture = df_test['Max_Moisture_In_Park'].groupby((df_test['Year'], df_test['Week'])).mean().to_frame().reset_index()
max_weekly_moisture.columns = ['Year', 'Week', 'Max_Weekly_Moisture']
df_test = df_test.merge(max_weekly_moisture, how = 'left')

min_weekly_moisture = df_test['Min_Moisture_In_Park'].groupby((df_test['Year'], df_test['Week'])).mean().to_frame().reset_index()
min_weekly_moisture.columns = ['Year', 'Week', 'Min_Weekly_Moisture']
df_test = df_test.merge(min_weekly_moisture, how = 'left')
```


```python
df_full = df.append(df_test)
```

## Handling Missing Values

- Using random forests for all missing value prediction for imputation
- For values with missing average, minimum, and maximum values, will first predict the average, then use that in predicting the minimum, then use both in predicting the maximum.

This section is relatively lengthy, and I used alot of copying/pasting.  There are better ways to handle this for something that would be used for production, but it got the job done for this application.

#### Average Atmospheric Pressure


```python
X = df_full[df_full['Average_Atmospheric_Pressure'].isnull() == False].drop(['ID', 'Footfall', 'Date', 'Year', 'Average_Atmospheric_Pressure'
                                                              ,'Max_Atmospheric_Pressure'
                                                              ,'Min_Atmospheric_Pressure'
                                                              ,'Min_Ambient_Pollution'
                                                              ,'Max_Ambient_Pollution'], axis = 1)

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X = imp.fit_transform(X)

y = df_full['Average_Atmospheric_Pressure'].dropna()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

rfr_avg_atmosphere = RandomForestRegressor(n_estimators = 150, n_jobs = 4)
rfr_avg_atmosphere.fit(X_train, y_train)
```




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               n_estimators=150, n_jobs=4, oob_score=False, random_state=None,
               verbose=0, warm_start=False)




```python
np.mean(cross_val_score(rfr_avg_atmosphere, X_test, y_test))
```




    0.99347389326468905




```python
# Predicting the missing values and filling them in the dataframe
# Training data
X_avg_atmosphere = df[df['Average_Atmospheric_Pressure'].isnull() == True].drop(['ID', 'Footfall', 'Date', 'Year', 'Average_Atmospheric_Pressure'
                                                              ,'Max_Atmospheric_Pressure'
                                                              ,'Min_Atmospheric_Pressure'
                                                              ,'Min_Ambient_Pollution'
                                                              ,'Max_Ambient_Pollution'], axis = 1)

X_avg_atmosphere = imp.fit_transform(X_avg_atmosphere)

avg_atmosphere_prediction = rfr_avg_atmosphere.predict(X_avg_atmosphere)
avg_atmosphere_prediction = pd.DataFrame({'ID':df.ix[(df['Average_Atmospheric_Pressure'].isnull() == True)]['ID']
                                          ,'avg_atmo_predict':avg_atmosphere_prediction})

df = df.merge(avg_atmosphere_prediction, how = 'left', on = 'ID')

df.Average_Atmospheric_Pressure.fillna(df.avg_atmo_predict, inplace=True)
del df['avg_atmo_predict']
```


```python
# Predicting the missing values and filling them in the dataframe
# Test data
X_avg_atmosphere = df_test[df_test['Average_Atmospheric_Pressure'].isnull() == True].drop(['ID', 'Date', 'Average_Atmospheric_Pressure', 'Year'
                                                              ,'Max_Atmospheric_Pressure'
                                                              ,'Min_Atmospheric_Pressure'
                                                              ,'Min_Ambient_Pollution'
                                                              ,'Max_Ambient_Pollution'], axis = 1)

X_avg_atmosphere = imp.fit_transform(X_avg_atmosphere)

avg_atmosphere_prediction = rfr_avg_atmosphere.predict(X_avg_atmosphere)
avg_atmosphere_prediction = pd.DataFrame({'ID':df_test.ix[(df_test['Average_Atmospheric_Pressure'].isnull() == True)]['ID']
                                          ,'avg_atmo_predict':avg_atmosphere_prediction})

df_test = df_test.merge(avg_atmosphere_prediction, how = 'left', on = 'ID')

df_test.Average_Atmospheric_Pressure.fillna(df_test.avg_atmo_predict, inplace=True)
del df_test['avg_atmo_predict']
```

#### Max Atmospheric Pressure


```python
X = df_full[df_full['Max_Atmospheric_Pressure'].isnull() == False].drop(['ID', 'Footfall', 'Date', 'Year'
                                                              ,'Max_Atmospheric_Pressure'
                                                              ,'Min_Atmospheric_Pressure'
                                                              ,'Min_Ambient_Pollution'
                                                              ,'Max_Ambient_Pollution'], axis = 1)

X = imp.fit_transform(X)

y = df_full['Max_Atmospheric_Pressure'].dropna()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

rfr_max_atmosphere = RandomForestRegressor(n_estimators = 150, n_jobs = 4)
rfr_max_atmosphere.fit(X_train, y_train)
```




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               n_estimators=150, n_jobs=4, oob_score=False, random_state=None,
               verbose=0, warm_start=False)




```python
np.mean(cross_val_score(rfr_max_atmosphere, X_test, y_test))
```




    0.9954164885643183




```python
# Predicting the missing values and filling them in the dataframe
# Training data
X_max_atmosphere = df[df['Max_Atmospheric_Pressure'].isnull() == True].drop(['ID', 'Footfall', 'Date', 'Year'
                                                              ,'Max_Atmospheric_Pressure'
                                                              ,'Min_Atmospheric_Pressure'
                                                              ,'Min_Ambient_Pollution'
                                                              ,'Max_Ambient_Pollution'], axis = 1)

X_max_atmosphere = imp.fit_transform(X_max_atmosphere)

max_atmosphere_prediction = rfr_max_atmosphere.predict(X_max_atmosphere)
max_atmosphere_prediction = pd.DataFrame({'ID':df.ix[(df['Max_Atmospheric_Pressure'].isnull() == True)]['ID']
                                          ,'max_atmo_predict':max_atmosphere_prediction})

df = df.merge(max_atmosphere_prediction, how = 'left', on = 'ID')

df.Max_Atmospheric_Pressure.fillna(df.max_atmo_predict, inplace=True)
del df['max_atmo_predict']
```


```python
# Predicting the missing values and filling them in the dataframe
# Test data
X_max_atmosphere = df_test[df_test['Max_Atmospheric_Pressure'].isnull() == True].drop(['ID', 'Date', 'Year'
                                                              ,'Max_Atmospheric_Pressure'
                                                              ,'Min_Atmospheric_Pressure'
                                                              ,'Min_Ambient_Pollution'
                                                              ,'Max_Ambient_Pollution'], axis = 1)

X_max_atmosphere = imp.fit_transform(X_max_atmosphere)

max_atmosphere_prediction = rfr_max_atmosphere.predict(X_max_atmosphere)
max_atmosphere_prediction = pd.DataFrame({'ID':df_test.ix[(df_test['Max_Atmospheric_Pressure'].isnull() == True)]['ID']
                                          ,'max_atmo_predict':max_atmosphere_prediction})

df_test = df_test.merge(max_atmosphere_prediction, how = 'left', on = 'ID')

df_test.Max_Atmospheric_Pressure.fillna(df_test.max_atmo_predict, inplace=True)
del df_test['max_atmo_predict']
```

#### Min Atmospheric Pressure


```python
X = df_full[df_full['Min_Atmospheric_Pressure'].isnull() == False].drop(['ID', 'Footfall', 'Date', 'Year'
                                                              ,'Min_Atmospheric_Pressure'
                                                              ,'Max_Ambient_Pollution'
                                                              ,'Min_Ambient_Pollution'], axis = 1)

X = imp.fit_transform(X)

y = df_full['Min_Atmospheric_Pressure'].dropna()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

rfr_min_atmosphere = RandomForestRegressor(n_estimators = 150, n_jobs = 4)
rfr_min_atmosphere.fit(X_train, y_train)
```




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               n_estimators=150, n_jobs=4, oob_score=False, random_state=None,
               verbose=0, warm_start=False)




```python
np.mean(cross_val_score(rfr_min_atmosphere, X_test, y_test))
```




    0.99499363701433363




```python
# Predicting the missing values and filling them in the dataframe
# Training data
X_min_atmosphere = df[df['Min_Atmospheric_Pressure'].isnull() == True].drop(['ID', 'Footfall', 'Date', 'Year'
                                                              ,'Min_Atmospheric_Pressure'
                                                              ,'Min_Ambient_Pollution'
                                                              ,'Max_Ambient_Pollution'], axis = 1)

X_min_atmosphere = imp.fit_transform(X_min_atmosphere)

min_atmosphere_prediction = rfr_min_atmosphere.predict(X_min_atmosphere)
min_atmosphere_prediction = pd.DataFrame({'ID':df.ix[(df['Min_Atmospheric_Pressure'].isnull() == True)]['ID']
                                          ,'min_atmo_predict':min_atmosphere_prediction})

df = df.merge(min_atmosphere_prediction, how = 'left', on = 'ID')

df.Min_Atmospheric_Pressure.fillna(df.min_atmo_predict, inplace=True)
del df['min_atmo_predict']
```


```python
# Predicting the missing values and filling them in the dataframe
# Test data
X_min_atmosphere = df_test[df_test['Min_Atmospheric_Pressure'].isnull() == True].drop(['ID', 'Date', 'Year'
                                                              ,'Min_Atmospheric_Pressure'
                                                              ,'Min_Ambient_Pollution'
                                                              ,'Max_Ambient_Pollution'], axis = 1)

X_min_atmosphere = imp.fit_transform(X_min_atmosphere)

min_atmosphere_prediction = rfr_min_atmosphere.predict(X_min_atmosphere)
min_atmosphere_prediction = pd.DataFrame({'ID':df_test.ix[(df_test['Min_Atmospheric_Pressure'].isnull() == True)]['ID']
                                          ,'min_atmo_predict':min_atmosphere_prediction})

df_test = df_test.merge(min_atmosphere_prediction, how = 'left', on = 'ID')

df_test.Min_Atmospheric_Pressure.fillna(df_test.min_atmo_predict, inplace=True)
del df_test['min_atmo_predict']
```

#### Max Ambient Pollution


```python
X = df_full[df_full['Max_Ambient_Pollution'].isnull() == False].drop(['ID', 'Footfall', 'Date', 'Year'
                                                              ,'Min_Ambient_Pollution'
                                                              ,'Max_Ambient_Pollution'], axis = 1)

X = imp.fit_transform(X)

y = df_full['Max_Ambient_Pollution'].dropna()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

rfr_max_pollution = RandomForestRegressor(n_estimators = 150, n_jobs = 4)
rfr_max_pollution.fit(X_train, y_train)
```




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               n_estimators=150, n_jobs=4, oob_score=False, random_state=None,
               verbose=0, warm_start=False)




```python
np.mean(cross_val_score(rfr_max_pollution, X_test, y_test))
```




    0.80003476426720599




```python
# Predicting the missing values and filling them in the dataframe
# Training data
X_max_pollution = df[df['Max_Ambient_Pollution'].isnull() == True].drop(['ID', 'Footfall', 'Date', 'Year'
                                                              ,'Min_Ambient_Pollution'
                                                              ,'Max_Ambient_Pollution'], axis = 1)

X_max_pollution = imp.fit_transform(X_max_pollution)

max_pollution_prediction = rfr_max_pollution.predict(X_max_pollution)
max_pollution_prediction = pd.DataFrame({'ID':df.ix[(df['Max_Ambient_Pollution'].isnull() == True)]['ID']
                                          ,'max_pollution_predict':max_pollution_prediction})

df = df.merge(max_pollution_prediction, how = 'left', on = 'ID')

df.Max_Ambient_Pollution.fillna(df.max_pollution_predict, inplace=True)
del df['max_pollution_predict']
```


```python
# Predicting the missing values and filling them in the dataframe
# Testing data
X_max_pollution = df_test[df_test['Max_Ambient_Pollution'].isnull() == True].drop(['ID', 'Date', 'Year'
                                                              ,'Min_Ambient_Pollution'
                                                              ,'Max_Ambient_Pollution'], axis = 1)

X_max_pollution = imp.fit_transform(X_max_pollution)

max_pollution_prediction = rfr_max_pollution.predict(X_max_pollution)
max_pollution_prediction = pd.DataFrame({'ID':df_test.ix[(df_test['Max_Ambient_Pollution'].isnull() == True)]['ID']
                                          ,'max_pollution_predict':max_pollution_prediction})

df_test = df_test.merge(max_pollution_prediction, how = 'left', on = 'ID')

df_test.Max_Ambient_Pollution.fillna(df_test.max_pollution_predict, inplace=True)
del df_test['max_pollution_predict']
```

#### Min Ambient Pollution


```python
X = df_full[df_full['Min_Ambient_Pollution'].isnull() == False].drop(['ID', 'Footfall', 'Date', 'Year'
                                                              ,'Min_Ambient_Pollution'], axis = 1)

X = imp.fit_transform(X)

y = df_full['Min_Ambient_Pollution'].dropna()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

rfr_min_pollution = RandomForestRegressor(n_estimators = 150, n_jobs = 4)
rfr_min_pollution.fit(X_train, y_train)
```




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               n_estimators=150, n_jobs=4, oob_score=False, random_state=None,
               verbose=0, warm_start=False)




```python
np.mean(cross_val_score(rfr_min_pollution, X_test, y_test))
```




    0.75850550015000273




```python
# Predicting the missing values and filling them in the dataframe
# Training data
X_min_pollution = df[df['Min_Ambient_Pollution'].isnull() == True].drop(['ID', 'Footfall', 'Date', 'Year'
                                                              ,'Min_Ambient_Pollution'], axis = 1)

X_min_pollution = imp.fit_transform(X_min_pollution)

min_pollution_prediction = rfr_min_pollution.predict(X_min_pollution)
min_pollution_prediction = pd.DataFrame({'ID':df.ix[(df['Min_Ambient_Pollution'].isnull() == True)]['ID']
                                          ,'min_pollution_predict':min_pollution_prediction})

df = df.merge(min_pollution_prediction, how = 'left', on = 'ID')

df.Min_Ambient_Pollution.fillna(df.min_pollution_predict, inplace=True)
del df['min_pollution_predict']
```


```python
# Predicting the missing values and filling them in the dataframe
# Testing data
X_min_pollution = df_test[df_test['Min_Ambient_Pollution'].isnull() == True].drop(['ID', 'Date', 'Year'
                                                              ,'Min_Ambient_Pollution'], axis = 1)

X_min_pollution = imp.fit_transform(X_min_pollution)

min_pollution_prediction = rfr_min_pollution.predict(X_min_pollution)
min_pollution_prediction = pd.DataFrame({'ID':df_test.ix[(df_test['Min_Ambient_Pollution'].isnull() == True)]['ID']
                                          ,'min_pollution_predict':min_pollution_prediction})

df_test = df_test.merge(min_pollution_prediction, how = 'left', on = 'ID')

df_test.Min_Ambient_Pollution.fillna(df_test.min_pollution_predict, inplace=True)
del df_test['min_pollution_predict']
```

#### Average Breeze Speed


```python
X = df_full[df_full['Average_Breeze_Speed'].isnull() == False].drop(['ID', 'Footfall', 'Date', 'Year'
                                                              ,'Average_Breeze_Speed'
                                                              , 'Max_Breeze_Speed'
                                                              , 'Min_Breeze_Speed'
                                                              , 'Direction_Of_Wind'], axis = 1)

X = imp.fit_transform(X)

y = df_full['Average_Breeze_Speed'].dropna()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

rfr_avg_breeze = RandomForestRegressor(n_estimators = 150, n_jobs = 4)
rfr_avg_breeze.fit(X_train, y_train)
```




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               n_estimators=150, n_jobs=4, oob_score=False, random_state=None,
               verbose=0, warm_start=False)




```python
np.mean(cross_val_score(rfr_avg_breeze, X_test, y_test))
```




    0.91409857146873208




```python
# Predicting the missing values and filling them in the dataframe
# Training data
X_avg_breeze = df[df['Average_Breeze_Speed'].isnull() == True].drop(['ID', 'Footfall', 'Date', 'Year'
                                                              ,'Average_Breeze_Speed'
                                                              , 'Max_Breeze_Speed'
                                                              , 'Min_Breeze_Speed'
                                                              , 'Direction_Of_Wind'], axis = 1)

X_avg_breeze = imp.fit_transform(X_avg_breeze)

avg_breeze_prediction = rfr_avg_breeze.predict(X_avg_breeze)
avg_breeze_prediction = pd.DataFrame({'ID':df.ix[(df['Average_Breeze_Speed'].isnull() == True)]['ID']
                                          ,'avg_breeze_predict':avg_breeze_prediction})

df = df.merge(avg_breeze_prediction, how = 'left', on = 'ID')

df.Average_Breeze_Speed.fillna(df.avg_breeze_predict, inplace=True)
del df['avg_breeze_predict']
```


```python
# Predicting the missing values and filling them in the dataframe
# Testing data
X_avg_breeze = df_test[df_test['Average_Breeze_Speed'].isnull() == True].drop(['ID', 'Date', 'Year'
                                                              ,'Average_Breeze_Speed'
                                                              , 'Max_Breeze_Speed'
                                                              , 'Min_Breeze_Speed'
                                                              , 'Direction_Of_Wind'], axis = 1)

X_avg_breeze = imp.fit_transform(X_avg_breeze)

avg_breeze_prediction = rfr_avg_breeze.predict(X_avg_breeze)
avg_breeze_prediction = pd.DataFrame({'ID':df_test.ix[(df_test['Average_Breeze_Speed'].isnull() == True)]['ID']
                                          ,'avg_breeze_predict':avg_breeze_prediction})

df_test = df_test.merge(avg_breeze_prediction, how = 'left', on = 'ID')

df_test.Average_Breeze_Speed.fillna(df_test.avg_breeze_predict, inplace=True)
del df_test['avg_breeze_predict']
```

#### Max Breeze Speed


```python
X = df_full[df_full['Max_Breeze_Speed'].isnull() == False].drop(['ID', 'Footfall', 'Date', 'Year'
                                                              , 'Max_Breeze_Speed'
                                                              , 'Min_Breeze_Speed'
                                                              , 'Direction_Of_Wind'], axis = 1)

X = imp.fit_transform(X)

y = df_full['Max_Breeze_Speed'].dropna()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

rfr_max_breeze = RandomForestRegressor(n_estimators = 150, n_jobs = 4)
rfr_max_breeze.fit(X_train, y_train)
```




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               n_estimators=150, n_jobs=4, oob_score=False, random_state=None,
               verbose=0, warm_start=False)




```python
np.mean(cross_val_score(rfr_max_breeze, X_test, y_test))
```




    0.92781368606209191




```python
# Predicting the missing values and filling them in the dataframe
# Training data
X_max_breeze = df[df['Max_Breeze_Speed'].isnull() == True].drop(['ID', 'Footfall', 'Date', 'Year'
                                                              , 'Max_Breeze_Speed'
                                                              , 'Min_Breeze_Speed'
                                                              , 'Direction_Of_Wind'], axis = 1)

X_max_breeze = imp.fit_transform(X_max_breeze)

max_breeze_prediction = rfr_max_breeze.predict(X_max_breeze)
max_breeze_prediction = pd.DataFrame({'ID':df.ix[(df['Max_Breeze_Speed'].isnull() == True)]['ID']
                                          ,'max_breeze_predict':max_breeze_prediction})

df = df.merge(max_breeze_prediction, how = 'left', on = 'ID')

df.Max_Breeze_Speed.fillna(df.max_breeze_predict, inplace=True)
del df['max_breeze_predict']
```


```python
# Predicting the missing values and filling them in the dataframe
# Testing data
X_max_breeze = df_test[df_test['Max_Breeze_Speed'].isnull() == True].drop(['ID', 'Date', 'Year'
                                                              , 'Max_Breeze_Speed'
                                                              , 'Min_Breeze_Speed'
                                                              , 'Direction_Of_Wind'], axis = 1)

X_max_breeze = imp.fit_transform(X_max_breeze)

max_breeze_prediction = rfr_max_breeze.predict(X_max_breeze)
max_breeze_prediction = pd.DataFrame({'ID':df_test.ix[(df_test['Max_Breeze_Speed'].isnull() == True)]['ID']
                                          ,'max_breeze_predict':max_breeze_prediction})

df_test = df_test.merge(max_breeze_prediction, how = 'left', on = 'ID')

df_test.Max_Breeze_Speed.fillna(df_test.max_breeze_predict, inplace=True)
del df_test['max_breeze_predict']
```

#### Min Breeze Speed


```python
X = df_full[df_full['Min_Breeze_Speed'].isnull() == False].drop(['ID', 'Footfall', 'Date', 'Year'
                                                              , 'Min_Breeze_Speed'
                                                              , 'Direction_Of_Wind'], axis = 1)

X = imp.fit_transform(X)

y = df_full['Min_Breeze_Speed'].dropna()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

rfr_min_breeze = RandomForestRegressor(n_estimators = 150, n_jobs = 4)
rfr_min_breeze.fit(X_train, y_train)
```




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               n_estimators=150, n_jobs=4, oob_score=False, random_state=None,
               verbose=0, warm_start=False)




```python
np.mean(cross_val_score(rfr_min_breeze, X_test, y_test))
```




    0.88292032860270131




```python
# Predicting the missing values and filling them in the dataframe
# Training data
X_min_breeze = df[df['Min_Breeze_Speed'].isnull() == True].drop(['ID', 'Footfall', 'Date', 'Year'
                                                              , 'Min_Breeze_Speed'
                                                              , 'Direction_Of_Wind'], axis = 1)

X_min_breeze = imp.fit_transform(X_min_breeze)

min_breeze_prediction = rfr_min_breeze.predict(X_min_breeze)
min_breeze_prediction = pd.DataFrame({'ID':df.ix[(df['Min_Breeze_Speed'].isnull() == True)]['ID']
                                          ,'min_breeze_predict':min_breeze_prediction})

df = df.merge(min_breeze_prediction, how = 'left', on = 'ID')

df.Min_Breeze_Speed.fillna(df.min_breeze_predict, inplace=True)
del df['min_breeze_predict']
```


```python
# Predicting the missing values and filling them in the dataframe
# Testing data
X_min_breeze = df_test[df_test['Min_Breeze_Speed'].isnull() == True].drop(['ID', 'Date', 'Year'
                                                              , 'Min_Breeze_Speed'
                                                              , 'Direction_Of_Wind'], axis = 1)

X_min_breeze = imp.fit_transform(X_min_breeze)

min_breeze_prediction = rfr_min_breeze.predict(X_min_breeze)
min_breeze_prediction = pd.DataFrame({'ID':df_test.ix[(df_test['Min_Breeze_Speed'].isnull() == True)]['ID']
                                          ,'min_breeze_predict':min_breeze_prediction})

df_test = df_test.merge(min_breeze_prediction, how = 'left', on = 'ID')

df_test.Min_Breeze_Speed.fillna(df_test.min_breeze_predict, inplace=True)
del df_test['min_breeze_predict']
```

#### Average Moisture

79 missing values, causing high error on test set


```python
X = df_full[df_full['Average_Moisture_In_Park'].isnull() == False].drop(['ID', 'Footfall', 'Date', 'Year'
                                                              , 'Average_Moisture_In_Park'
                                                              , 'Min_Moisture_In_Park'
                                                              , 'Max_Moisture_In_Park'
                                                              , 'Direction_Of_Wind'], axis = 1)

X = imp.fit_transform(X)

y = df_full['Average_Moisture_In_Park'].dropna()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

rfr_avg_moisture = RandomForestRegressor(n_estimators = 150, n_jobs = 4)
rfr_avg_moisture.fit(X_train, y_train)
```




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               n_estimators=150, n_jobs=4, oob_score=False, random_state=None,
               verbose=0, warm_start=False)




```python
np.mean(cross_val_score(rfr_avg_moisture, X_test, y_test))
```




    0.87629324767915051




```python
# Predicting the missing values and filling them in the dataframe
# Training data
X_avg_moisture = df[df['Average_Moisture_In_Park'].isnull() == True].drop(['ID', 'Footfall', 'Date', 'Year'
                                                              , 'Average_Moisture_In_Park'
                                                              , 'Min_Moisture_In_Park'
                                                              , 'Max_Moisture_In_Park'
                                                              , 'Direction_Of_Wind'], axis = 1)

X_avg_moisture = imp.fit_transform(X_avg_moisture)

avg_moisture_prediction = rfr_avg_moisture.predict(X_avg_moisture)
avg_moisture_prediction = pd.DataFrame({'ID':df.ix[(df['Average_Moisture_In_Park'].isnull() == True)]['ID']
                                          ,'avg_moisture_predict':avg_moisture_prediction})

df = df.merge(avg_moisture_prediction, how = 'left', on = 'ID')

df.Average_Moisture_In_Park.fillna(df.avg_moisture_predict, inplace=True)
del df['avg_moisture_predict']
```


```python
# Predicting the missing values and filling them in the dataframe
# Testing data
X_avg_moisture = df_test[df_test['Average_Moisture_In_Park'].isnull() == True].drop(['ID', 'Date', 'Year'
                                                              , 'Average_Moisture_In_Park'
                                                              , 'Min_Moisture_In_Park'
                                                              , 'Max_Moisture_In_Park'
                                                              , 'Direction_Of_Wind'], axis = 1)

X_avg_moisture = imp.fit_transform(X_avg_moisture)

avg_moisture_prediction = rfr_avg_moisture.predict(X_avg_moisture)
avg_moisture_prediction = pd.DataFrame({'ID':df_test.ix[(df_test['Average_Moisture_In_Park'].isnull() == True)]['ID']
                                          ,'avg_moisture_predict':avg_moisture_prediction})

df_test = df_test.merge(avg_moisture_prediction, how = 'left', on = 'ID')

df_test.Average_Moisture_In_Park.fillna(df_test.avg_moisture_predict, inplace=True)
del df_test['avg_moisture_predict']
```

#### Min Moisture


```python
X = df_full[df_full['Min_Moisture_In_Park'].isnull() == False].drop(['ID', 'Footfall', 'Date', 'Year'
                                                              , 'Min_Moisture_In_Park'
                                                              , 'Max_Moisture_In_Park'
                                                              , 'Direction_Of_Wind'], axis = 1)

X = imp.fit_transform(X)

y = df_full['Min_Moisture_In_Park'].dropna()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

rfr_min_moisture = RandomForestRegressor(n_estimators = 150, n_jobs = 4)
rfr_min_moisture.fit(X_train, y_train)
```




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               n_estimators=150, n_jobs=4, oob_score=False, random_state=None,
               verbose=0, warm_start=False)




```python
np.mean(cross_val_score(rfr_min_moisture, X_test, y_test))
```




    0.93591464075660369




```python
# Predicting the missing values and filling them in the dataframe
# Training data
X_min_moisture = df[df['Min_Moisture_In_Park'].isnull() == True].drop(['ID', 'Footfall', 'Date', 'Year'
                                                              , 'Min_Moisture_In_Park'
                                                              , 'Max_Moisture_In_Park'
                                                              , 'Direction_Of_Wind'], axis = 1)

X_min_moisture = imp.fit_transform(X_min_moisture)

min_moisture_prediction = rfr_min_moisture.predict(X_min_moisture)
min_moisture_prediction = pd.DataFrame({'ID':df.ix[(df['Min_Moisture_In_Park'].isnull() == True)]['ID']
                                          ,'min_moisture_predict':min_moisture_prediction})

df = df.merge(min_moisture_prediction, how = 'left', on = 'ID')

df.Min_Moisture_In_Park.fillna(df.min_moisture_predict, inplace=True)
del df['min_moisture_predict']
```


```python
# Predicting the missing values and filling them in the dataframe
# Testing data
X_min_moisture = df_test[df_test['Min_Moisture_In_Park'].isnull() == True].drop(['ID', 'Date', 'Year'
                                                              , 'Min_Moisture_In_Park'
                                                              , 'Max_Moisture_In_Park'
                                                              , 'Direction_Of_Wind'], axis = 1)

X_min_moisture = imp.fit_transform(X_min_moisture)

min_moisture_prediction = rfr_min_moisture.predict(X_min_moisture)
min_moisture_prediction = pd.DataFrame({'ID':df_test.ix[(df_test['Min_Moisture_In_Park'].isnull() == True)]['ID']
                                          ,'min_moisture_predict':min_moisture_prediction})

df_test = df_test.merge(min_moisture_prediction, how = 'left', on = 'ID')

df_test.Min_Moisture_In_Park.fillna(df_test.min_moisture_predict, inplace=True)
del df_test['min_moisture_predict']
```

#### Max Moisture


```python
X = df_full[df_full['Max_Moisture_In_Park'].isnull() == False].drop(['ID', 'Footfall', 'Date', 'Year'
                                                              , 'Max_Moisture_In_Park'
                                                              , 'Direction_Of_Wind'], axis = 1)

X = imp.fit_transform(X)

y = df_full['Max_Moisture_In_Park'].dropna()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

rfr_max_moisture = RandomForestRegressor(n_estimators = 150, n_jobs = 4)
rfr_max_moisture.fit(X_train, y_train)
```




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               n_estimators=150, n_jobs=4, oob_score=False, random_state=None,
               verbose=0, warm_start=False)




```python
np.mean(cross_val_score(rfr_max_moisture, X_test, y_test))
```




    0.8239425796574974




```python
# Predicting the missing values and filling them in the dataframe
# Training data
X_max_moisture = df[df['Max_Moisture_In_Park'].isnull() == True].drop(['ID', 'Footfall', 'Date', 'Year'
                                                              , 'Max_Moisture_In_Park'
                                                              , 'Direction_Of_Wind'], axis = 1)

X_max_moisture = imp.fit_transform(X_max_moisture)

max_moisture_prediction = rfr_max_moisture.predict(X_max_moisture)
max_moisture_prediction = pd.DataFrame({'ID':df.ix[(df['Max_Moisture_In_Park'].isnull() == True)]['ID']
                                          ,'max_moisture_predict':max_moisture_prediction})

df = df.merge(max_moisture_prediction, how = 'left', on = 'ID')

df.Max_Moisture_In_Park.fillna(df.max_moisture_predict, inplace=True)
del df['max_moisture_predict']
```


```python
# Predicting the missing values and filling them in the dataframe
# Testing data
X_max_moisture = df_test[df_test['Max_Moisture_In_Park'].isnull() == True].drop(['ID', 'Date', 'Year'
                                                              , 'Max_Moisture_In_Park'
                                                              , 'Direction_Of_Wind'], axis = 1)

X_max_moisture = imp.fit_transform(X_max_moisture)

max_moisture_prediction = rfr_max_moisture.predict(X_max_moisture)
max_moisture_prediction = pd.DataFrame({'ID':df_test.ix[(df_test['Max_Moisture_In_Park'].isnull() == True)]['ID']
                                          ,'max_moisture_predict':max_moisture_prediction})

df_test = df_test.merge(max_moisture_prediction, how = 'left', on = 'ID')

df_test.Max_Moisture_In_Park.fillna(df_test.max_moisture_predict, inplace=True)
del df_test['max_moisture_predict']
```

#### Var1


```python
X = df_full[df_full['Var1'].isnull() == False].drop(['ID', 'Footfall', 'Date', 'Year'
                                                    , 'Var1'
                                                    ], axis = 1)

X = imp.fit_transform(X)

y = df_full['Var1'].dropna()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

rfr_var1 = RandomForestRegressor(n_estimators = 150, n_jobs = 4)
rfr_var1.fit(X_train, y_train)
```




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               n_estimators=150, n_jobs=4, oob_score=False, random_state=None,
               verbose=0, warm_start=False)




```python
np.mean(cross_val_score(rfr_var1, X_test, y_test))
```




    0.61193989909532365




```python
# Predicting the missing values and filling them in the dataframe
# Training data
X_var1 = df[df['Var1'].isnull() == True].drop(['ID', 'Footfall', 'Date', 'Year'
                                                              , 'Var1'
                                                              ], axis = 1)

X_var1 = imp.fit_transform(X_var1)

var1_prediction = rfr_var1.predict(X_var1)
var1_prediction = pd.DataFrame({'ID':df.ix[(df['Var1'].isnull() == True)]['ID']
                                          ,'var1_predict':var1_prediction})

df = df.merge(var1_prediction, how = 'left', on = 'ID')

df.Var1.fillna(df.var1_predict, inplace=True)
del df['var1_predict']
```


```python
# Predicting the missing values and filling them in the dataframe
# Testing data
X_var1 = df_test[df_test['Var1'].isnull() == True].drop(['ID', 'Date', 'Year'
                                                              , 'Var1'
                                                              ], axis = 1)

X_var1 = imp.fit_transform(X_var1)

var1_prediction = rfr_var1.predict(X_var1)
var1_prediction = pd.DataFrame({'ID':df_test.ix[(df_test['Var1'].isnull() == True)]['ID']
                                          ,'var1_predict':var1_prediction})

df_test = df_test.merge(var1_prediction, how = 'left', on = 'ID')

df_test.Var1.fillna(df_test.var1_predict, inplace=True)
del df_test['var1_predict']
```

#### Direction of Wind


```python
X = df_full[df_full['Direction_Of_Wind'].isnull() == False].drop(['ID', 'Footfall', 'Date', 'Year'
                                                    , 'Direction_Of_Wind'
                                                    ], axis = 1)

X = imp.fit_transform(X)

y = df_full['Direction_Of_Wind'].dropna()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

rfr_wind_dir = RandomForestRegressor(n_estimators = 150, n_jobs = 4)
rfr_wind_dir.fit(X_train, y_train)
```




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               n_estimators=150, n_jobs=4, oob_score=False, random_state=None,
               verbose=0, warm_start=False)




```python
np.mean(cross_val_score(rfr_wind_dir, X_test, y_test))
```




    0.67839725370662685




```python
# Predicting the missing values and filling them in the dataframe
# Training data
X_wind_dir = df[df['Direction_Of_Wind'].isnull() == True].drop(['ID', 'Footfall', 'Date', 'Year'
                                                              , 'Direction_Of_Wind'
                                                              ], axis = 1)

X_wind_dir = imp.fit_transform(X_wind_dir)

wind_dir_prediction = rfr_wind_dir.predict(X_wind_dir)
wind_dir_prediction = pd.DataFrame({'ID':df.ix[(df['Direction_Of_Wind'].isnull() == True)]['ID']
                                          ,'wind_dir_predict':wind_dir_prediction})

df = df.merge(wind_dir_prediction, how = 'left', on = 'ID')

df.Direction_Of_Wind.fillna(df.wind_dir_predict, inplace=True)
del df['wind_dir_predict']
```


```python
# Predicting the missing values and filling them in the dataframe
# Testing data
X_wind_dir = df_test[df_test['Direction_Of_Wind'].isnull() == True].drop(['ID', 'Date', 'Year'
                                                              , 'Direction_Of_Wind'
                                                              ], axis = 1)

X_wind_dir = imp.fit_transform(X_wind_dir)

wind_dir_prediction = rfr_wind_dir.predict(X_wind_dir)
wind_dir_prediction = pd.DataFrame({'ID':df_test.ix[(df_test['Direction_Of_Wind'].isnull() == True)]['ID']
                                          ,'wind_dir_predict':wind_dir_prediction})

df_test = df_test.merge(wind_dir_prediction, how = 'left', on = 'ID')

df_test.Direction_Of_Wind.fillna(df_test.wind_dir_predict, inplace=True)
del df_test['wind_dir_predict']
```

Checking for all missing values being accounted for:


```python
df_missing_check = df.append(df_test)
df_missing_check = df_missing_check.sort_values(['Date', 'Park_ID'], ascending=[1, 1])
msno.matrix(df_missing_check)
```

<img src="https://github.com/JeffMacaluso/JeffMacaluso.github.io/blob/master/_posts/Hackathon_files/Hackathon_98_0.png?raw=true">


## Model Building

### Gradient Boosted Trees

- Max depth of 5 is most effective

- Outperformed both random forests and AdaBoost


```python
X = df.drop(['ID', 'Footfall', 'Date', 'Year'
             , 'Location_Type', 'Average_Atmospheric_Pressure', 'Max_Atmospheric_Pressure'
             , 'Var1', 'Max_Ambient_Pollution', 'Min_Atmospheric_Pressure'
             , 'Max_Breeze_Speed', 'Min_Breeze_Speed', 'Min_Ambient_Pollution'
             , 'Max_Moisture_In_Park'
            ], axis = 1)
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
X = imp.fit_transform(X)

y = df['Footfall']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
```


```python
gbr = GradientBoostingRegressor(n_estimators = 300
                               , max_depth = 5
                               )
gbr.fit(X_train, y_train)
```




    GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.1, loss='ls',
                 max_depth=5, max_features=None, max_leaf_nodes=None,
                 min_samples_leaf=1, min_samples_split=2,
                 min_weight_fraction_leaf=0.0, n_estimators=300,
                 presort='auto', random_state=None, subsample=1.0, verbose=0,
                 warm_start=False)




```python
# Regular cross validation
np.mean(cross_val_score(gbr, X_test, y_test, n_jobs = 3))
```




    0.95762687239052224




```python
# K-fold cross validation

k_fold = KFold(len(y), n_folds=10, shuffle=True, random_state=0)
cross_val_score(gbr, X, y, cv=k_fold, n_jobs=3)
```

    C:\Anaconda3\lib\site-packages\sklearn\externals\joblib\hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    C:\Anaconda3\lib\site-packages\sklearn\externals\joblib\hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    C:\Anaconda3\lib\site-packages\sklearn\externals\joblib\hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    C:\Anaconda3\lib\site-packages\sklearn\externals\joblib\hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    C:\Anaconda3\lib\site-packages\sklearn\externals\joblib\hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    C:\Anaconda3\lib\site-packages\sklearn\externals\joblib\hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    C:\Anaconda3\lib\site-packages\sklearn\externals\joblib\hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    C:\Anaconda3\lib\site-packages\sklearn\externals\joblib\hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    C:\Anaconda3\lib\site-packages\sklearn\externals\joblib\hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    C:\Anaconda3\lib\site-packages\sklearn\externals\joblib\hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    




    array([ 0.96254909,  0.96338802,  0.96328219,  0.96139247,  0.96212329,
            0.9647377 ,  0.96253464,  0.96374171,  0.96351884,  0.96228093])



Examining the distance of predictions from the actual, and looking for common characteristics among the parts with the biggest differences.


```python
# Plot of error over time
y_pred = gbr.predict(X_test)

cv_error = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Error': (y_test - y_pred)})
cv_error = pd.merge(df, cv_error, left_index=True, right_index=True)

error_plot = cv_error[['Date', 'Error']]
error_plot = error_plot.set_index('Date')
error_plot.plot(figsize = (20,10))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1774e13a518>



<img src="https://github.com/JeffMacaluso/JeffMacaluso.github.io/blob/master/_posts/Hackathon_files/Hackathon_106_1.png?raw=true">



```python
cv_error.sort_values('Error').head()
```




<div style="overflow-x:auto;">
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Park_ID</th>
      <th>Date</th>
      <th>Direction_Of_Wind</th>
      <th>Average_Breeze_Speed</th>
      <th>Max_Breeze_Speed</th>
      <th>Min_Breeze_Speed</th>
      <th>Var1</th>
      <th>Average_Atmospheric_Pressure</th>
      <th>Max_Atmospheric_Pressure</th>
      <th>...</th>
      <th>Max_Weekly_Atmosphere</th>
      <th>Min_Weekly_Atmosphere</th>
      <th>Max_Weekly_Pollution</th>
      <th>Min_Weekly_Pollution</th>
      <th>Avg_Weekly_Moisture</th>
      <th>Max_Weekly_Moisture</th>
      <th>Min_Weekly_Moisture</th>
      <th>Actual</th>
      <th>Error</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>95947</th>
      <td>3686224</td>
      <td>24</td>
      <td>2000-02-12</td>
      <td>178.000000</td>
      <td>16.720000</td>
      <td>38.000000</td>
      <td>0.000000</td>
      <td>1.66</td>
      <td>8300.000000</td>
      <td>8317.000000</td>
      <td>...</td>
      <td>8339.769841</td>
      <td>8302.928571</td>
      <td>318.951724</td>
      <td>153.875862</td>
      <td>246.730159</td>
      <td>284.904762</td>
      <td>200.650794</td>
      <td>528</td>
      <td>-250.086489</td>
      <td>778.086489</td>
    </tr>
    <tr>
      <th>22191</th>
      <td>3388817</td>
      <td>17</td>
      <td>1992-11-10</td>
      <td>138.026667</td>
      <td>78.072267</td>
      <td>97.077333</td>
      <td>58.773333</td>
      <td>0.00</td>
      <td>8005.033333</td>
      <td>8043.626667</td>
      <td>...</td>
      <td>8285.809524</td>
      <td>8245.206349</td>
      <td>291.910448</td>
      <td>153.671642</td>
      <td>254.158163</td>
      <td>278.739796</td>
      <td>225.091837</td>
      <td>1011</td>
      <td>-235.739930</td>
      <td>1246.739930</td>
    </tr>
    <tr>
      <th>18749</th>
      <td>3388419</td>
      <td>19</td>
      <td>1992-07-10</td>
      <td>83.000000</td>
      <td>31.160000</td>
      <td>45.600000</td>
      <td>15.200000</td>
      <td>0.00</td>
      <td>8341.000000</td>
      <td>8355.000000</td>
      <td>...</td>
      <td>8341.722222</td>
      <td>8287.119048</td>
      <td>324.059701</td>
      <td>160.686567</td>
      <td>240.734694</td>
      <td>281.693878</td>
      <td>186.000000</td>
      <td>1035</td>
      <td>-228.127802</td>
      <td>1263.127802</td>
    </tr>
    <tr>
      <th>18751</th>
      <td>3388421</td>
      <td>21</td>
      <td>1992-07-10</td>
      <td>81.000000</td>
      <td>27.360000</td>
      <td>45.600000</td>
      <td>15.200000</td>
      <td>0.00</td>
      <td>8348.000000</td>
      <td>8358.000000</td>
      <td>...</td>
      <td>8341.722222</td>
      <td>8287.119048</td>
      <td>324.059701</td>
      <td>160.686567</td>
      <td>240.734694</td>
      <td>281.693878</td>
      <td>186.000000</td>
      <td>999</td>
      <td>-227.231848</td>
      <td>1226.231848</td>
    </tr>
    <tr>
      <th>22247</th>
      <td>3394917</td>
      <td>17</td>
      <td>1992-11-12</td>
      <td>138.166667</td>
      <td>78.330667</td>
      <td>97.330667</td>
      <td>59.026667</td>
      <td>0.00</td>
      <td>8005.033333</td>
      <td>8043.626667</td>
      <td>...</td>
      <td>8285.809524</td>
      <td>8245.206349</td>
      <td>291.910448</td>
      <td>153.671642</td>
      <td>254.158163</td>
      <td>278.739796</td>
      <td>225.091837</td>
      <td>652</td>
      <td>-222.944937</td>
      <td>874.944937</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 50 columns</p>
</div>




```python
cv_error.sort_values('Error').tail()
```




<div style="overflow-x:auto;">
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Park_ID</th>
      <th>Date</th>
      <th>Direction_Of_Wind</th>
      <th>Average_Breeze_Speed</th>
      <th>Max_Breeze_Speed</th>
      <th>Min_Breeze_Speed</th>
      <th>Var1</th>
      <th>Average_Atmospheric_Pressure</th>
      <th>Max_Atmospheric_Pressure</th>
      <th>...</th>
      <th>Max_Weekly_Atmosphere</th>
      <th>Min_Weekly_Atmosphere</th>
      <th>Max_Weekly_Pollution</th>
      <th>Min_Weekly_Pollution</th>
      <th>Avg_Weekly_Moisture</th>
      <th>Max_Weekly_Moisture</th>
      <th>Min_Weekly_Moisture</th>
      <th>Actual</th>
      <th>Error</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>69981</th>
      <td>3562439</td>
      <td>39</td>
      <td>1997-07-13</td>
      <td>204.000000</td>
      <td>25.080000</td>
      <td>45.600000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8000.260000</td>
      <td>8042.993333</td>
      <td>...</td>
      <td>8361.619048</td>
      <td>8293.492063</td>
      <td>315.517730</td>
      <td>194.496454</td>
      <td>250.790816</td>
      <td>287.770408</td>
      <td>208.040816</td>
      <td>1703</td>
      <td>182.894997</td>
      <td>1520.105003</td>
    </tr>
    <tr>
      <th>3429</th>
      <td>3335939</td>
      <td>39</td>
      <td>1991-01-05</td>
      <td>20.000000</td>
      <td>36.480000</td>
      <td>53.200000</td>
      <td>22.800000</td>
      <td>0.000000</td>
      <td>8000.400000</td>
      <td>8042.313333</td>
      <td>...</td>
      <td>8362.590278</td>
      <td>8287.125000</td>
      <td>299.838926</td>
      <td>111.087248</td>
      <td>265.745455</td>
      <td>290.850000</td>
      <td>225.940909</td>
      <td>1323</td>
      <td>184.154605</td>
      <td>1138.845395</td>
    </tr>
    <tr>
      <th>26230</th>
      <td>3403224</td>
      <td>24</td>
      <td>1993-04-03</td>
      <td>159.000000</td>
      <td>26.600000</td>
      <td>38.000000</td>
      <td>15.200000</td>
      <td>0.000000</td>
      <td>8341.000000</td>
      <td>8358.000000</td>
      <td>...</td>
      <td>8381.301587</td>
      <td>8309.087302</td>
      <td>303.657143</td>
      <td>147.142857</td>
      <td>252.214286</td>
      <td>287.311224</td>
      <td>201.443878</td>
      <td>1230</td>
      <td>195.827119</td>
      <td>1034.172881</td>
    </tr>
    <tr>
      <th>11167</th>
      <td>3352417</td>
      <td>17</td>
      <td>1991-10-13</td>
      <td>140.313333</td>
      <td>78.072267</td>
      <td>97.026667</td>
      <td>59.077333</td>
      <td>0.000000</td>
      <td>8005.033333</td>
      <td>8044.040000</td>
      <td>...</td>
      <td>8408.166667</td>
      <td>8359.476190</td>
      <td>309.082707</td>
      <td>173.593985</td>
      <td>252.872449</td>
      <td>284.448980</td>
      <td>204.811224</td>
      <td>1536</td>
      <td>203.969139</td>
      <td>1332.030861</td>
    </tr>
    <tr>
      <th>48807</th>
      <td>3486833</td>
      <td>33</td>
      <td>1995-06-18</td>
      <td>67.000000</td>
      <td>21.280000</td>
      <td>38.000000</td>
      <td>0.000000</td>
      <td>192.631933</td>
      <td>8420.000000</td>
      <td>8441.000000</td>
      <td>...</td>
      <td>8362.158730</td>
      <td>8313.317460</td>
      <td>317.285714</td>
      <td>182.400000</td>
      <td>232.301020</td>
      <td>282.260204</td>
      <td>176.326531</td>
      <td>1663</td>
      <td>206.940381</td>
      <td>1456.059619</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 50 columns</p>
</div>



Viewing variable importance - used to discard unnecessary variables


```python
# Plotting feature importance

gbr_labels = df_test.drop(['ID', 'Date', 'Year'
             , 'Location_Type', 'Average_Atmospheric_Pressure', 'Max_Atmospheric_Pressure'
             , 'Var1', 'Max_Ambient_Pollution', 'Min_Atmospheric_Pressure'
             , 'Max_Breeze_Speed', 'Min_Breeze_Speed', 'Min_Ambient_Pollution'
             , 'Max_Moisture_In_Park'
                            ], axis = 1)

feature_importance = gbr.feature_importances_

feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(10,15))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, gbr_labels.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
```

<img src="https://github.com/JeffMacaluso/JeffMacaluso.github.io/blob/master/_posts/Hackathon_files/Hackathon_110_0.png?raw=true">


## Outputting Predictions


```python
# Predicting for submission
X_submission = df_test.drop(['ID', 'Date', 'Year'
             , 'Location_Type', 'Average_Atmospheric_Pressure', 'Max_Atmospheric_Pressure'
             , 'Var1', 'Max_Ambient_Pollution', 'Min_Atmospheric_Pressure'
             , 'Max_Breeze_Speed', 'Min_Breeze_Speed', 'Min_Ambient_Pollution'
             , 'Max_Moisture_In_Park'
                            ], axis = 1)
X_submission = imp.fit_transform(X_submission)
y_submission = gbr.predict(X_submission)
```


```python
# Exporting results
df_submission = pd.DataFrame(y_submission, index = df_test['ID'])
df_submission.columns = ['Footfall']
df_submission.to_csv('test.csv')
```


```python
# Saving train/test with imputed null values
df.to_csv('train_imputed.csv')
df_test.to_csv('test_imputed.csv')
```
