import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from env import host, user, password




def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

    
    


def get_telco():
    '''
    This function reads in the telco data from the Codeup db
    and returns a pandas DataFrame with all columns.
    '''
    
    sql_query = 'select * from customers where contract_type_id = 3;'
    return pd.read_sql(sql_query, get_connection('telco_churn'))






def clean_telco(df):
    '''
    clean_telco will take in a dataframe, cleans the data by converting total_charges from object to int, replacing new customer total_charges from Nan to zero, as well as sets customer_id to index.
    '''
    
    features = ['customer_id', 'monthly_charges', 'tenure', 'total_charges']
    df = df[features]
    
    df.total_charges = pd.to_numeric(df.total_charges, errors='coerce').astype('float64')
    df = df.fillna(0)
    
    df = df.set_index("customer_id")
   
    return df





def split_telco(df):
    """
    train_validate_test will take one argument(df) and 
    run clean_telco to remove/rename/encode columns
    then split our data into 20/80, 
    then split the 80% into 30/70
    
    perform a train, validate, test split
    
    return: the three split pandas dataframes-train/validate/test
    """  
    
    train_validate, test = train_test_split(df, test_size=0.2, random_state=3210)
    train, validate = train_test_split(train_validate, train_size=0.7, random_state=3210)
    return train, validate, test





def prepare_telco():
    '''
    prepare_telco will: 
    - read in telco dataset for 2 year contracted customer as a pandas DataFrame,
    - clean the data
    - split the data
    return: the three split pandas dataframes-train/validate/test
    '''
    
    df = clean_telco(get_telco())
    return split_telco(df)





def scaled_telco():
    