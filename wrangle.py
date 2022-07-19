import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from env import host, user, password

def get_connection(db, user=user, host=host,password=password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


def get_zillow_data():
    filename = 'zillow_values.csv'
    
    # Acquire data from csv if exists
    if os.path.exists(filename):
        print ('Using cached csv')
        return pd.read_csv(filename, index_col=0)
    else:
        return get_new_zillow_data()
    
def get_new_zillow_data():
    """Returns a dataframe of all 2017 properties that are Single Family Residential"""
    query = '''
            SELECT parcelid AS parcel,
            bedroomcnt AS bedrooms,
            bathroomcnt AS bathrooms,
            calculatedfinishedsquarefeet AS square_feet,
            garagecarcnt as garage,
            poolcnt as pool,
            lotsizesquarefeet as lot_size,
            regionidzip as zip,
            yearbuilt AS year_built,
            assessmentyear as year_last_assessed,
            taxvaluedollarcnt AS tax_value,
            taxamount AS tax_amount,
            fips as fed_code
            FROM properties_2017
            JOIN predictions_2017 USING (parcelid)
            JOIN propertylandusetype USING (propertylandusetypeid)
            WHERE transactiondate IS NOT NULL
            AND propertylandusedesc IN("Single Family Residential","Inferred Single Family Residential")
            AND transactiondate BETWEEN '2017-01-01' and '2017-12-31';
            '''
    #Export to csv
    df = pd.read_sql(query, get_connection('zillow'))
    df.to_csv('zillow_values.csv', index = False)

    return df

def add_info(df):
        # Make Column for age of home
        df['age'] = 2017-df.year_built
        return df

def fill_na(df):
        # Update garage and pool NaN values to 0
        df.pool=df.pool.fillna(0)
        df.garage= df.garage.fillna(0)
        df=df.dropna()
        return df
    
def optimize_types(df):
    # Convert some columns to integers
    # fips, yearbuilt, and bedrooms can be integers
    df["garage"] = df["garage"].astype(int)
    df["pool"] = df["pool"].astype(int)
    df["fed_code"] = df["fed_code"].astype(int)
    df["bedrooms"] = df["bedrooms"].astype(int)
    df["bathrooms"] = df["bathrooms"].astype(int)
    df["tax_value"] = df["tax_value"].astype(int)
    df["square_feet"] = df["square_feet"].astype(int)
    df["year_built"] = df["year_built"].astype(int)
    df["zip"] = df["zip"].astype(int)
    return df
      
def replace_nan(df):
    df = df.replace(r'^\s*$', np.nan, regex=True)
    return df       
        
def handle_outliers(df):
    df=df[df.bathrooms <=6]
    df=df[df.bedrooms <=6]  
    df = df[df.tax_value < 2_000_000]
    return df

def wrangle_zillow():
    """
    Acquires Zillow data
    Handles nulls
    optimizes or fixes data types
    handles outliers w/ manual logic
    returns a clean dataframe
    """
    df = get_zillow_data()
    df = fill_na(df)
    df = optimize_types(df)
    df = replace_nan(df)
    df = add_info(df)
    df = handle_outliers(df)
    df.to_csv('zillow_values.csv')

    return df



def split_data(df, train_size_vs_train_test = 0.8, train_size_vs_train_val = 0.7, random_state = 123):
    """Splits the inputted dataframe into 3 datasets for train, validate and test (in that order).
    Can specific as arguments the percentage of the train/val set vs test (default 0.8) and the percentage of the
    train size vs train/val (default 0.7). Default values results in following:
    Train: 0.56
    Validate: 0.24
    Test: 0.2"""
    train_val, test = train_test_split(df, train_size=train_size_vs_train_test, random_state=123)
    train, validate = train_test_split(train_val, train_size=train_size_vs_train_val, random_state=123)
    
    train_size = train_size_vs_train_test*train_size_vs_train_val
    test_size = 1 - train_size_vs_train_test
    validate_size = 1-test_size-train_size
    
    print(f"Data split as follows: Train {train_size:.2%}, Validate {validate_size:.2%}, Test {test_size:.2%}")
    
    return train, validate, test

def scale_data(train, validate, test, features_to_scale):
    """Scales data using MinMax Scaler. 
    Accepts train, validate, and test datasets as inputs as well as a list of the features to scale. 
    Returns dataframe with scaled values added on as columns"""
    
    # Fit the scaler to train data only
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(train[features_to_scale])
    
    # Generate a list of the new column names with _scaled added on
    scaled_columns = [col+"_scaled" for col in features_to_scale]
    
    # Transform the separate datasets using the scaler learned from train
    scaled_train = scaler.transform(train[features_to_scale])
    scaled_validate = scaler.transform(validate[features_to_scale])
    scaled_test = scaler.transform(test[features_to_scale])
    
    # Concatenate the scaled data to the original unscaled data
    train_scaled = pd.concat([train, pd.DataFrame(scaled_train,index=train.index, columns = scaled_columns)],axis=1)
    validate_scaled = pd.concat([validate, pd.DataFrame(scaled_validate,index=validate.index, columns =
                                                        scaled_columns)],axis=1)
    test_scaled = pd.concat([test, pd.DataFrame(scaled_test,index=test.index, columns = scaled_columns)],axis=1)

    return train_scaled, validate_scaled, test_scaled

def data_split(df, target):
    '''
    This function drops the customer_id column and then splits a dataframe into 
    train, validate, and test in order to explore the data and to create and validate models. 
    It takes in a dataframe and contains an integer for setting a seed for replication. 
    Test is 20% of the original dataset. The remaining 80% of the dataset is 
    divided between valiidate and train, with validate being .30*.80= 24% of 
    the original dataset, and train being .70*.80= 56% of the original dataset. 
    The function returns, train, validate and test dataframes. 
    '''
    
    train, test = train_test_split(df, test_size = .2, random_state=123)   
    train, validate = train_test_split(train, test_size=.3, random_state=123)


    print(f'train -> {train.shape}')
    print(f'validate -> {validate.shape}')
    print(f'test -> {test.shape}')
    
    return train, validate, test

## TODO Encode categorical variables (and FIPS is a category so Fips to string to one-hot-encoding
## TODO Scale numeric columns
## TODO How to handle 0 bedroom, 0 bathroom homes? Drop them? How many? They're probably clerical nulls