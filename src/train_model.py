from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import lightgbm as lgb


train = joblib.load('data/processed/train.pkl')
test = joblib.load('data/processed/test.pkl')

# The model needs to be able to create a prognosis for max 12 days with 24h resolution.
# This equates to 288 hours ahead in the future, based on the latest measurement

