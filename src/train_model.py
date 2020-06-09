from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
#import lightgbm as lgb


train = joblib.load('data/processed/train.pkl')
test = joblib.load('data/processed/test.pkl')

# The model needs to be able to create a prognosis for max 12 days with 24h resolution.
# This equates to 288 hours ahead in the future, based on the latest measurement

train['k1'] = train['count'].shift(1)
train.dropna(inplace=True)

lgb_params = {
    'objective': 'regression',
    'boosting': 'gbdt',
    'verbose': 1,
    'metric': 'rmse'
}

X_train, X_test, y_train, y_test = train_test_split(train.drop(['count'], axis='columns'), train['count'], test_size=.3)

lgb_estimator = lgb.train(
    lgb_params,
    train_set=lgb.Dataset(X_train, y_train),
    valid_sets=lgb.Dataset(X_test, y_test),
    num_boost_round=500
)

preds = lgb_estimator.predict(X_test)

# Doing some evaluation
import matplotlib.pyplot as plt
_ = plt.plot(y_test.values, 'bo')
_ = plt.plot(preds, 'r-')
plt.show()