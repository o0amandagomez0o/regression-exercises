import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from env import host, user, password

'''
*------------------*
|                  |
|     ACQUIRE      |
|                  |
*------------------*
'''
def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

    
    


def get_zillow():
    '''
    This function reads in the zillow data from the Codeup db
    and returns a pandas DataFrame with only requested columns: 
    - square feet of the home 
    - number of bedrooms
    - number of bathrooms
    '''
    
    sql_query = """
    select parcelid, calculatedfinishedsquarefeet, bedroomcnt, bathroomcnt, taxvaluedollarcnt 
    from properties_2017 
    join predictions_2017 using(parcelid) 
    where transactiondate between "2017-05-01" and "2017-06-30";
    """
    return pd.read_sql(sql_query, get_connection('zillow'))





def zillow_df():
    '''
    This function reads in the zillow data from the Codeup db
    and returns a pandas DataFrame with only requested columns: 
    - square feet of the home 
    - number of bedrooms
    - number of bathrooms
    '''
    
    sql_query = """
    select * 
    from properties_2017 
    join predictions_2017 using(parcelid) 
    where transactiondate between "2017-05-01" and "2017-06-30";
    """
    return pd.read_sql(sql_query, get_connection('zillow'))




'''
*------------------*
|                  |
|     PREPARE      |
|                  |
*------------------*
'''
def clean_zillow(df):
    '''
    clean _zillow will take in a dataframe, clean the data by renaming columns, establishing parcelid as index, drops all null/NaN rows, as well as removes tax_value & square_feet outliers.
    '''
    
    df = df.rename(columns={"bedroomcnt": "bedrooms", "bathroomcnt": "bathrooms", "calculatedfinishedsquarefeet": "square_feet", "taxamount": "taxes", "taxvaluedollarcnt": "tax_value"})
    df = df.set_index("parcelid")
    df = df.dropna()
    
    upper_bound, lower_bound = outlier(df, "tax_value")
    
    df = df[df.tax_value < upper_bound]
    
    upper_bound1, lower_bound1 = outlier(df, "square_feet")
    
    df = df[df.square_feet < upper_bound1]
    
    return df
    
    

    
def outlier(df, feature):
    '''
    outlier will take in a dataframe's feature:
    - calculate it's 1st & 3rd quartiles,
    - use their difference to calculate the IQR
    - then apply to calculate upper and lower bounds
    '''
    q1 = df[feature].quantile(.25)
    q3 = df[feature].quantile(.75)
    
    iqr = q3 - q1
    
    multiplier = 1.5
    upper_bound = q3 + (multiplier * iqr)
    lower_bound = q1 - (multiplier * iqr)
    
    return upper_bound, lower_bound
 
    
    
    
    
def split_zillow(df):
    """
    split_zillow will take one argument(df) and 
    run clean_zillow to remove NaNs/outliers, rename columns, and reset index to `parcelid`
    then split our data into 20/80, 
    then split the 80% into 30/70
    
    perform a train, validate, test split
    
    return: the three split pandas dataframes-train/validate/test
    """  
    
    train_validate, test = train_test_split(df, test_size=0.2, random_state=3210)
    train, validate = train_test_split(train_validate, train_size=0.7, random_state=3210)
    return train, validate, test





def wrangle_zillow():
    '''
    wrangle_zillow will: 
    - read in zillow dataset for transaction dates between 05/2017-06/2017 as a pandas DataFrame,
    - clean the data
    - split the data
    return: the three split pandas dataframes-train/validate/test
    '''
    
    df = clean_zillow(get_zillow())
    return split_zillow(df)




'''
*------------------*
|                  |
|     EXPLORE      |
|                  |
*------------------*
'''
def plot_variable_pairs(df):
    '''
    plot_variable_pairs will take in a dataframe and create a pair grid with 
    '''
    g = sns.PairGrid(df)
    g.map_diag(plt.hist)
    g.map_offdiag(sns.regplot)
    return g





def plot_categorical_and_continous_vars(train, target, cat_var):

 


        
