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
            garagecarcnt as garage,
            poolcnt as pool,
            airconditioningtypeid as ac,
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
        # Replace white space values with NaN values.
        df = df.replace(r'^\s*$', np.nan, regex=True)
        
        # Update garage and pool NaN values to 0
        df.pool=df.pool.fillna(0)
        df.garage= df.garage.fillna(0)
        
        # Make Column for age of home
        df['age'] = 2017-df.year_built
        
        # Convert column datatype
        convert_dict = {'fed_code': object,
                        'year_built': int,
                        'year_assessed':int
                       }
        df=df.astype(convert_dict)
                                
       #Export to csv
        df = pd.read_sql(query, get_connection('zillow'))
        df.to_csv('zillow_values.csv', index = False)
    
    return df



