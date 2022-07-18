import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from env import host, user, password, get_db_url


def wrangle_zillow():
    filename = 'zillow_values.csv'
    # Acquire data from csv if exists
    if os.path.exists(filename):
        print ('Using cached csv')
        return pd.read_csv(filename, index_col=0)
    # Acquire data from database if no CSV exists
    else:
        query = '''
            SELECT parcelid AS parcel,
            bedroomcnt AS bedrooms,
            bathroomcnt AS bathrooms,
            calculatedfinishedsquarefeet AS square_feet,
            taxvaluedollarcnt AS tax_value,
            yearbuilt AS year,
            taxamount AS tax_amount,
            fips as fed_code
            FROM properties_2017
            JOIN propertylandusetype USING (propertylandusetypeid)
            WHERE propertylandusedesc IN("Single Family Residential","Inferred Single Family Residential");
            '''
        df = pd.read_sql(query, get_db_url('zillow'))
        df.tocsv('zillow_values.csv', index = False)
    
    return df



