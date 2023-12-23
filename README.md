# Ski Resorts Price Analysis and Prediction

![screenshots1](images/ski.jpeg?raw=true)

Welcome to ML_Ski_ResortsPrice_Predictor, an open-source data analysis and machine learning project! This repository hosts a comprehensive analysis of a real-world dataset, showcasing the entire data science pipeline from data exploration to model building.
## Overview of the Dataset
#### The dataset includes various features of ski resorts across Europe, such as elevation points, slope lengths, lift counts, and amenities like snowparks and night skiing. It provides a comprehensive view of these resorts' characteristics.
###### Resort: Name of the ski resort.
###### Country: Country where the resort is located.
###### HighestPoint: The highest point of the resort in meters.
###### LowestPoint: The lowest point of the resort in meters.
###### DayPassPriceAdult: Price of an adult day pass in Euros.
###### BeginnerSlope: Length of beginner slopes in kilometers.
###### IntermediateSlope: Length of intermediate slopes in kilometers.
###### DifficultSlope: Length of difficult slopes in kilometers.
###### TotalSlope: Total length of slopes in kilometers.
###### Snowparks: Availability of snowparks (Yes/No).
###### NightSki: Availability of night skiing (Yes/No).
###### SurfaceLifts: Number of surface lifts.
###### ChairLifts: Number of chairlifts.
###### GondolaLifts: Number of gondola lifts.
###### TotalLifts: Total number of lifts.
###### LiftCapacity: Lift capacity per hour.
###### SnowCannons: Number of snow cannons.


## Modules/Libraries
* Scikit-learn
* MatPlotLib
* Statistics
* Numpy
* Pandas
* Seaborn
* Scipy
## Univariate Analysis / Skewness / Distributions
### Univariate Analysis:

Numeric Variables: Showed varied distributions. Features like slope lengths and lift counts were right-skewed, indicating a prevalence of smaller resorts. Prices and elevation points had a more balanced distribution.
Categorical Variables: The presence of snowparks and night skiing varied among resorts. The dataset also covered a diverse range of European countries.


![screenshots1](images/skewcheckhistograms.png?raw=true)


![screenshots1](images/numbersofresortsnow.png?raw=true)

![screenshots1](images/averagenumberbynight.png?raw=true)

## Bivariate Analysis:

#### Relationships between features like total slope length and lift counts, and day pass prices and slope lengths were explored. Positive correlations were observed in some cases (e.g., total slope vs. total lifts), while other relationships were less clear, suggesting the influence of multiple factors.

![screenshots1](images/bivariatewithue.png?raw=true)

## Multivariate Analysis with 'Hue':

#### Provided deeper insights into how variables interacted across different categories, such as countries or amenities available.

![screenshots1](images/3dcharts.png?raw=true)

```python
# Preparing data for 3D plots
plot_data1 = ski_resorts_data[['TotalSlope', 'TotalLifts', 'DayPassPriceAdult']]
plot_data2 = ski_resorts_data[['HighestPoint', 'LowestPoint', 'TotalLifts']]
plot_data3 = ski_resorts_data[['BeginnerSlope', 'IntermediateSlope', 'DifficultSlope']]

# Creating 3D plots
fig = plt.figure(figsize=(18, 6))

# 3D Plot 1: Total Slope vs Total Lifts vs Day Pass Price
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.scatter(plot_data1['TotalSlope'], plot_data1['TotalLifts'], plot_data1['DayPassPriceAdult'], c='blue', marker='o')
ax1.set_xlabel('Total Slope')
ax1.set_ylabel('Total Lifts')
ax1.set_zlabel('Day Pass Price Adult')
ax1.set_title('Total Slope vs Total Lifts vs Day Pass Price')

# 3D Plot 2: Highest Point vs Lowest Point vs Total Lifts
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.scatter(plot_data2['HighestPoint'], plot_data2['LowestPoint'], plot_data2['TotalLifts'], c='red', marker='o')
ax2.set_xlabel('Highest Point')
ax2.set_ylabel('Lowest Point')
ax2.set_zlabel('Total Lifts')
ax2.set_title('Highest Point vs Lowest Point vs Total Lifts')

# 3D Plot 3: Beginner vs Intermediate vs Difficult Slopes
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.scatter(plot_data3['BeginnerSlope'], plot_data3['IntermediateSlope'], plot_data3['DifficultSlope'], c='green', marker='o')
ax3.set_xlabel('Beginner Slope')
ax3.set_ylabel('Intermediate Slope')
ax3.set_zlabel('Difficult Slope')
ax3.set_title('Beginner vs Intermediate vs Difficult Slopes')

plt.tight_layout()
plt.show()
```

## Preprocessing/Modeling
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



df = pd.read_csv("data/European_Ski_Resorts.csv")
# Copying the original dataset for transformations
ski_resorts_data = df.copy()

# filtering outiers out
ski_resorts_data['SlopeRange'] = ski_resorts_data['HighestPoint'] - ski_resorts_data['LowestPoint']
Q1 = ski_resorts_data['DayPassPriceAdult'].quantile(0.25)
Q3 = ski_resorts_data['DayPassPriceAdult'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
ski_resorts_data = ski_resorts_data[(ski_resorts_data['DayPassPriceAdult'] >= lower_bound) & 
                                    (ski_resorts_data['DayPassPriceAdult'] <= upper_bound)]
# One-hot encoding for categorical variables
ski_resorts_data_encoded = pd.get_dummies(ski_resorts_data, columns=['Country', 'Snowparks', 'NightSki'])
ski_resorts_data_encoded.drop(['Resort','Unnamed: 0',],axis=1)
X = ski_resorts_data_encoded.drop(['Resort','Unnamed: 0','DayPassPriceAdult'],axis=1)
y = ski_resorts_data_encoded['DayPassPriceAdult']
feature_names = ski_resorts_data_encoded.drop(['Resort', 'Unnamed: 0'], axis=1).columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
#### Applied standardization to  numeric features.
#### Employed one-hot encoding for categorical variables.
Split the data into training and testing sets.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



df = pd.read_csv("data/European_Ski_Resorts.csv")
# Copying the original dataset for transformations
ski_resorts_data = df.copy()

# filtering outiers out
ski_resorts_data['SlopeRange'] = ski_resorts_data['HighestPoint'] - ski_resorts_data['LowestPoint']
Q1 = ski_resorts_data['DayPassPriceAdult'].quantile(0.25)
Q3 = ski_resorts_data['DayPassPriceAdult'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
ski_resorts_data = ski_resorts_data[(ski_resorts_data['DayPassPriceAdult'] >= lower_bound) & 
                                    (ski_resorts_data['DayPassPriceAdult'] <= upper_bound)]
# One-hot encoding for categorical variables
ski_resorts_data_encoded = pd.get_dummies(ski_resorts_data, columns=['Country', 'Snowparks', 'NightSki'])
ski_resorts_data_encoded.drop(['Resort','Unnamed: 0',],axis=1)
X = ski_resorts_data_encoded.drop(['Resort','Unnamed: 0','DayPassPriceAdult'],axis=1)
y = ski_resorts_data_encoded['DayPassPriceAdult']
feature_names = ski_resorts_data_encoded.drop(['Resort', 'Unnamed: 0'], axis=1).columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred = linear_reg.predict(X_test)
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred))
Linear Regression MSE: 1.938075262191945e+27

```
![screenshots1](images/linear10.png?raw=true)

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
# Splitting Dataset into training and testing sets
#X = ski_resorts_data.drop(['DayPassPriceAdult', 'Resort', 'Unnamed: 0','Country'], axis=1)
#y = ski_resorts_data['DayPassPriceAdult']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_RfR = RandomForestRegressor()
model_RfR.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model_RfR.predict(X_test)

# Calculating MAE, MSE, and RMSE
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
Mean Absolute Error (MAE): 3.98337837837838
Mean Squared Error (MSE): 27.893631081081082
Root Mean Squared Error (RMSE): 5.281442140275805

```
![screenshots1](images/top10ridgeregression.png?raw=true)

```python
from sklearn.linear_model import ElasticNet

elastic_net_reg = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net_reg.fit(X_train, y_train)
y_pred = elastic_net_reg.predict(X_test)
print("Elastic Net Regression MSE:", mean_squared_error(y_test, y_pred))
Elastic Net Regression MSE: 30.854962299677354
# Extracting coefficients
elastic_net_coef = elastic_net_reg.coef_

# Plotting
plt.figure(figsize=(20, 10))
plt.barh(range(len(elastic_net_coef)), elastic_net_coef)
plt.yticks(range(len(feature_names)), feature_names)
plt.xlabel('Coefficient Value')
plt.title('Feature Importance in Elastic Net Model')
plt.show()
```
![screenshots1](images/elastic.png?raw=true)
## Regression Models:

Linear Regression: Performed poorly, indicating that the linear model assumptions might not hold for this dataset.
Random Forest and Gradient Boosting Regressors: Showed better performance, suggesting their ability to capture more complex, non-linear relationships in the data.
Metrics: MSE and RMSE provided insights into the average error of the models' predictions. R² score indicated how well the models explained the variance in the target variable.

## Key Takeaways
The dataset presents a diverse picture of European ski resorts, with a mix of different sizes, amenities, and price points.
Most resorts cater to casual or intermediate skiers, with fewer resorts offering extensive facilities for advanced skiers.
Machine learning models indicated that predicting certain aspects of ski resorts, such as day pass prices, can be complex and may be influenced by a range of factors.
The performance of different models highlighted the importance of choosing the right algorithm and properly preparing the data for analysis.
## Recommendations
Further analysis could be conducted by focusing on specific subsets of data (e.g., resorts in a particular country or with specific amenities).
For machine learning models, feature engineering and model tuning could potentially improve performance.
Considering the complexity of the dataset, exploring other machine learning algorithms or deep learning approaches might yield more insights.
