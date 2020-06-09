from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from xgboost import XGBRegressor as xgb


train = joblib.load('data/processed/train.pkl')
test = joblib.load('data/processed/test.pkl')

# The model needs to be able to create a prognosis for max 12 days with 24h resolution.
# This equates to 288 hours ahead in the future, based on the latest measurement

# Create hold-out set to validate two different approaches
holdout = train.loc['2011-05-01':'2011-05-19']
train = train.loc[(train.index < '2011-04-30') | (train.index > '2011-05-20')]


kone_train = train.copy()
kone_train['k1'] = kone_train['count'].shift(1)
kone_train.dropna(inplace=True)

# neutral approach: no shift column
X_train, X_test, y_train, y_test = train_test_split(train.drop(['count'], axis='columns'), train['count'],
                                                    test_size=.3, random_state=42)

model = xgb(eta=0.4)

model.fit(X_train, y_train)

# k1 approach: input shift column

X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(kone_train.drop(['count'], axis='columns'), kone_train['count'],
                                                            test_size=.3, random_state=42)

kone_model = xgb(eta=0.4)
kone_model.fit(X_train_n.values, y_train_n.values)

# Validate and compare on holdout set
neutral_preds = model.predict(holdout.drop(['count'], axis='columns'))

# Advanced
kone_holdout = holdout.copy()
kone_holdout.drop(['count'], axis='columns', inplace=True)
index_range = kone_holdout.index
counter = 0
prediction = 0
predictions = []
import numpy as np

for idex in index_range:
    temp = kone_holdout.loc[idex]
    if counter == 0:
        temp = np.append(temp, 82)
    else:
        temp = np.append(temp, prediction)
    prediction = kone_model.predict(temp.reshape(1, -1))
    predictions.append(prediction)



# Doing some evaluation
import matplotlib.pyplot as plt
_ = plt.plot(y_test.values, 'bo')
_ = plt.plot(preds, 'r-')
plt.show()
