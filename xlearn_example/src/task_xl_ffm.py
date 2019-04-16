import pandas as pd
import xlearn as xl
from utils import _convert_to_ffm
from sklearn.model_selection import train_test_split
import config
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('data/train.csv')

cols = ['Education', 'ApplicantIncome', 'Loan_Status', 'Credit_History']
train_sub = train[cols]
train_sub['Credit_History'].fillna(0, inplace=True)
dict_ls = {'Y': 1, 'N': 0}
train_sub['Loan_Status'].replace(dict_ls, inplace=True)

train, test = train_test_split(train_sub, test_size=0.3, random_state=5)
print(f' train data: \n {train.head()}')

# # Add unseen data
# test = test.append({'Education': 'Unknown', 'ApplicantIncome': 1000, 'Loan_Status': 1, 'Credit_History': 1.0}, ignore_index=True)
# print(f' test data: \n {test.head()}')

# Initialise fields and variables encoder
encoder = {"currentcode": len(config.NUMERICAL_FEATURES),  # Unique index for each numerical field or categorical variables
           "catdict": {},  # Dictionary that stores numerical and categorical variables
           "catcodes": {}}  # Dictionary that stores index for each categorical variables per categorical field

encoder = _convert_to_ffm('data/', train, 'train', config.GOAL[0],
                          config.NUMERICAL_FEATURES,
                          config.CATEGORICAL_FEATURES,
                          config.ALL_FEATURES,
                          encoder)

encoder = _convert_to_ffm('data/', test, 'test', config.GOAL[0],
                          config.NUMERICAL_FEATURES,
                          config.CATEGORICAL_FEATURES,
                          config.ALL_FEATURES,
                          encoder)

ffm_model = xl.create_ffm()
ffm_model.setTrain("data/train_ffm.txt")
ffm_model.setValidate("data/test_ffm.txt")

param = {'task': 'binary',
         'lr': 0.2,
         'lambda': 0.002,
         'metric': 'auc',
         'nthread': config.NUM_THREADS}

# Start to train
ffm_model.fit(param, 'trained_models/model.out')

# Cross Validation
ffm_model.cv(param)

# Prediction task
ffm_model.setTest("data/test_ffm.txt")  # Test data
ffm_model.setSigmoid()  # Convert output to 0-1

# Start to predict
ffm_model.predict("trained_models/model.out", "output/predictions.txt")

# Online Learning (train new data based on pre-trained model)
ffm_model = xl.create_ffm()
ffm_model.setTrain("data/train_ffm.txt")
ffm_model.setValidate("data/test_ffm.txt")
ffm_model.setPreModel("trained_models/model.out")
param = {'task': 'binary', 'lr': 0.2, 'lambda': 0.002}
ffm_model.fit(param, "trained_models/model.out")
