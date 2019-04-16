import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data
diabetes_y = diabetes.target
features = diabetes.feature_names

# Convert to Dataframe
df_X = pd.DataFrame(data=diabetes_X, columns=features)
print(df_X.head())

df_y = pd.DataFrame(data=diabetes_y, columns=['diabetes_indicator'])
print(df_y.head())

# Split the data into training/testing sets
X_train = df_X[:-20]
X_test = df_X[-20:]

# Split the targets into training/testing sets
y_train = df_y[:-20]
y_test = df_y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients of features')
print(features)
print(regr.coef_)

# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))
