import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Data description: train.csv contains calendar dates 1 - 20, and test has 21 - 28/30/31 depending on month
# If our strategy is rolling window, we know that we are predicting 10 days forward in the future at most.
# Maybe a column that holds "days in the future".

train_path = 'data/raw/train.csv'
test_path = 'data/raw/test.csv'

df = pd.read_csv(train_path)
test = pd.read_csv(test_path)


def set_dateindex(_df):
    _df['datetime'] = pd.to_datetime(_df.datetime)
    _df.index = _df['datetime']
    _df.drop(['datetime'], axis='columns', inplace=True)

    return _df


df = set_dateindex(df)
test = set_dateindex(test)

# Feature engineering
# Create dummy variables for spring, summer, fall & winter instead of int season
def dummy_season(_df):
    _df['spring'] = _df.season.apply(lambda x: 1 if x == 1 else 0)
    _df['summer'] = _df.season.apply(lambda x: 1 if x == 2 else 0)
    _df['fall'] = _df.season.apply(lambda x: 1 if x == 3 else 0)
    _df['winter'] = _df.season.apply(lambda x: 1 if x == 4 else 0)
    _df.drop(['season'], axis='columns', inplace=True)

    return _df

df = dummy_season(df)
test = dummy_season(test)

# Create "days before free day", meaning days before either a holiday or sat - sun arrives

# Weather -> 1 = good, 4 = really bad

def create_day_of_week_dummies(_df):
    _df['dow'] = _df.index.dayofweek
    _df['monday'] = _df.dow.apply(lambda x: 1 if x == 0 else 0)
    _df['tuesday'] = _df.dow.apply(lambda x: 1 if x == 1 else 0)
    _df['wednesday'] = _df.dow.apply(lambda x: 1 if x == 2 else 0)
    _df['thursday'] = _df.dow.apply(lambda x: 1 if x == 3 else 0)
    _df['friday'] = _df.dow.apply(lambda x: 1 if x == 4 else 0)
    _df['saturday'] = _df.dow.apply(lambda x: 1 if x == 5 else 0)
    _df['sunday'] = _df.dow.apply(lambda x: 1 if x == 6 else 0)
    _df.drop(['dow'], axis='columns', inplace=True)

    return _df

df = create_day_of_week_dummies(df)
test = create_day_of_week_dummies(test)



def create_hour_of_day(_df):
    _df['hod'] = _df.index.hour

    return _df

df = create_hour_of_day(df)
test = create_hour_of_day(test)


df.drop(['casual', 'registered'], axis='columns', inplace=True)

import joblib
joblib.dump(df, 'data/processed/train.pkl')
joblib.dump(test, 'data/processed/test.pkl')
