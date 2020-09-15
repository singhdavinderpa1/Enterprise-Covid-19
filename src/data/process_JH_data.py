import pandas as pd
import numpy as np

from datetime import datetime


def processed_relational_data():
    ''' Transformes the John Hopkins COVID-19 data into a relational data set with primary key( index in terms of pandas data frame)

    '''

    data_path = '../../data/raw/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    pd_raw = pd.read_csv(data_path)

    #renaming the country/region columns to country and province/state column to state for the ease of accessing it
    pd_data_base = pd_raw.rename(columns={'Country/Region': 'country',
                                          'Province/State': 'state'})
    #filling the state value as no whereever the value was NAN
    pd_data_base['state'] = pd_data_base['state'].fillna('no')

    #removing the data that is not required for modelling
    pd_data_base = pd_data_base.drop(['Lat', 'Long'], axis=1)


    pd_relational_model = pd_data_base.set_index(['state', 'country']) \
        .T \
        .stack(level=[0, 1]) \
        .reset_index() \
        .rename(columns={'level_0': 'date',
                         0: 'confirmed'},
                )

    pd_relational_model['date'] = pd_relational_model.date.astype('datetime64[ns]')

    #saving relational model to the csv file
    pd_relational_model.to_csv('../../data/processed/COVID_relational_confirmed.csv', sep=';', index=False)
    print(pd_relational_model.tail())
    print(' Number of rows stored: ' + str(pd_relational_model.shape[0]))


if __name__ == '__main__':
    processed_relational_data()
