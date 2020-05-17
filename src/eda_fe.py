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

def set_dateindex(df):
    df['datetime'] = pd.to_datetime(df.datetime)
    df.index = df['datetime']
    df.drop(['datetime'], axis='columns', inplace=True)

    return df

df = set_dateindex(df)
test = set_dateindex(test)

# Feature engineering
# Create dummy variables for spring, summer, fall & winter instead of int season
df['spring'] = df.season.apply(lambda x: 1 if x == 1 else 0)
df['summer'] = df.season.apply(lambda x: 1 if x == 2 else 0)
df['fall'] = df.season.apply(lambda x: 1 if x == 3 else 0)
df['winter'] = df.season.apply(lambda x: 1 if x == 4 else 0)
df.drop(['season'], axis='columns', inplace=True)

# Create "days before free day", meaning days before either a holiday or sat - sun arrives

# Weather -> 1 = good, 4 = really bad



# Some EDA of total daily count
tf = df['count'].resample('D').sum()
_ = plt.plot(tf)
