import pandas as pd
from sklearn import datasets, linear_model

# Load the iris dataset
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
features = iris.feature_names

# Convert to Dataframe
df_X = pd.DataFrame(data=iris_X, columns=features)
print(df_X.head())

df_y = pd.DataFrame(data=iris_y, columns=['label'])
print(df_y.head())

# Split the data into training/testing sets
X_train = df_X[:-20]
X_test = df_X[-20:]

# Split the targets into training/testing sets
y_train = df_y[:-20]
y_test = df_y[-20:]

# Create logistic regression object
regr = linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial')

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)
