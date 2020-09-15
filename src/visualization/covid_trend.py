import pandas as pd
import numpy as np
import subprocess
import dash
dash.__version__
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from sklearn import linear_model
reg = linear_model.LinearRegression(fit_intercept=True)
import os

print(os.getcwd())
from src.data.get_data import get_johns_hopkins
from src.data.process_JH_data import processed_relational_data
from src.data.get_pd_large import pd_result_large
from src.features.build_features import *

get_johns_hopkins()
processed_relational_data()
pd_result_large()

print(os.getcwd())
df_large = pd.read_csv('../../data/processed/COVID_final_set.csv', sep=';')

fig = go.Figure()

app = dash.Dash()
app.layout = html.Div([

    dcc.Markdown('''
    #  #  Applied Data Science on COVID-19 John Hopkins data
    

    '''),

    dcc.Markdown('''
    ## Multi-Select Country list for graph visualization
    
    '''),

    dcc.Dropdown(
        id='country_drop_down',
        options=[{'label': each, 'value': each} for each in df_large['country'].unique()],
        value=['India', 'Japan', 'Germany'],  # which are pre-selected
        multi=True
    ),

    dcc.Markdown('''
        ## Select Timeline of confirmed COVID-19 cases (Filtered, doubling rate)
        #### 1: Processed Relational data
        #### 2: Data filtered with Savgol filter
        #### 3: Time doubling rate data
        ##### 4: Time doubling rate data filtered with Savgol filter
        '''),

    dcc.Dropdown(
        id='doubling_time',
        options=[
            {'label': 'Timeline Confirmed Cases', 'value': 'confirmed'},
            {'label': 'Timeline Confirmed Filtered Cases', 'value': 'confirmed_filtered'},
            {'label': 'Timeline Doubling Rate', 'value': 'confirmed_DR'},
            {'label': 'Timeline Doubling Rate Filtered', 'value': 'confirmed_filtered_DR'},
        ],
        value='confirmed',
        multi=False
    ),

    dcc.Graph(figure=fig, id='main_window_slope')
])


@app.callback(
    Output('main_window_slope', 'figure'),
    [Input('country_drop_down', 'value'),
     Input('doubling_time', 'value')])
def update_figure(country_list, show_doubling):
    if 'doubling_rate' in show_doubling:
        my_yaxis = {'type': "log",
                    'title': 'Approximated doubling rate over 3 days (larger numbers are better #stayathome)'
                    }
    else:
        my_yaxis = {'type': "log",
                    'title': 'Confirmed infected people (source johns hopkins csse, log-scale)'
                    }

    traces = []
    for each in country_list:

        df_plot = df_large[df_large['country'] == each]

        # extracting the needed columns based upon user selection
        if show_doubling == 'doubling_rate_filtered':
            df_plot = df_plot[
                ['state', 'country', 'confirmed', 'confirmed_filtered', 'confirmed_DR', 'confirmed_filtered_DR',
                 'date']].groupby(['country', 'date']).agg(np.mean).reset_index()
        else:
            df_plot = df_plot[
                ['state', 'country', 'confirmed', 'confirmed_filtered', 'confirmed_DR', 'confirmed_filtered_DR',
                 'date']].groupby(['country', 'date']).agg(np.sum).reset_index()
        # print(show_doubling)

        traces.append(dict(x=df_plot.date,
                           y=df_plot[show_doubling],
                           mode='markers+lines',
                           opacity=0.9,
                           name=each
                           )
                      )

    return {
        'data': traces,
        'layout': dict(
            width=1280,
            height=720,

            xaxis={'title': 'Timeline',
                   'tickangle': -45,
                   'nticks': 20,
                   'tickfont': dict(size=14, color="#7f7f7f"),
                   },

            yaxis=my_yaxis
        )
    }


def get_doubling_time_via_regression(in_array):
    ''' Use a linear regression to approximate the doubling rate

        Parameters:
        ----------
        in_array : pandas.series

        Returns:
        ----------
        Doubling rate: double
    '''

    y = np.array(in_array)
    X = np.arange(-1, 2).reshape(-1, 1)

    assert len(in_array) == 3
    reg.fit(X, y)
    intercept = reg.intercept_
    slope = reg.coef_

    return intercept / slope


def savgol_filter(df_input, column='confirmed', window=5):
    ''' Savgol Filter which can be used in groupby apply function (data structure kept)

        parameters:
        ----------
        df_input : pandas.series
        column : str
        window : int
            used data points to calculate the filter result

        Returns:
        ----------
        df_result: pd.DataFrame
            the index of the df_input has to be preserved in result
    '''

    degree = 1
    df_result = df_input

    filter_in = df_input[column].fillna(0)  # attention with the neutral element here

    result = signal.savgol_filter(np.array(filter_in),
                                  window,  # window size used for filtering
                                  1)
    df_result[str(column + '_filtered')] = result
    return df_result


def rolling_reg(df_input, col='confirmed'):
    ''' Rolling Regression to approximate the doubling time'

        Parameters:
        ----------
        df_input: pd.DataFrame
        col: str
            defines the used column
        Returns:
        ----------
        result: pd.DataFrame
    '''
    days_back = 3
    result = df_input[col].rolling(
        window=days_back,
        min_periods=days_back).apply(get_doubling_time_via_regression, raw=False)

    return result


def calculate_filtered_data(df_input, filter_column='confirmed'):
    '''  Calculate savgol filter and return merged data frame

        Parameters:
        ----------
        df_input: pd.DataFrame
        filter_on: str
            defines the used column
        Returns:
        ----------
        df_output: pd.DataFrame
            the result will be joined as a new column on the input data frame
    '''

    must_contain = set(['state', 'country', filter_column])
    assert must_contain.issubset(set(df_input.columns)), ' Erro in calc_filtered_data not all columns in data frame'

    df_output = df_input.copy()  # we need a copy here otherwise the filter_on column will be overwritten

    pd_filtered_result = df_output[['state', 'country', filter_column]].groupby(['state', 'country']).apply(
        savgol_filter)  # .reset_index()

    # print('--+++ after group by apply')
    # print(pd_filtered_result[pd_filtered_result['country']=='Germany'].tail())

    # df_output=pd.merge(df_output,pd_filtered_result[['index',str(filter_on+'_filtered')]],on=['index'],how='left')
    df_output = pd.merge(df_output, pd_filtered_result[[str(filter_column + '_filtered')]], left_index=True,
                         right_index=True, how='left')
    # print(df_output[df_output['country']=='Germany'].tail())
    return df_output.copy()


def calculate_doubling_rate(df_input, filter_column='confirmed'):
    ''' Calculate approximated doubling rate and return merged data frame

        Parameters:
        ----------
        df_input: pd.DataFrame
        filter_column: str
            defines the used column
        Returns:
        ----------
        df_output: pd.DataFrame
            the result will be joined as a new column on the input data frame
    '''

    must_contain = set(['state', 'country', filter_column])
    assert must_contain.issubset(set(df_input.columns)), ' Erro in calc_filtered_data not all columns in data frame'

    pd_DR_result = df_input.groupby(['state', 'country']).apply(rolling_reg, filter_column).reset_index()

    pd_DR_result = pd_DR_result.rename(columns={filter_column: filter_column + '_DR',
                                                'level_2': 'index'})

    # we do the merge on the index of our big table and on the index column after groupby
    df_output = pd.merge(df_input, pd_DR_result[['index', str(filter_column + '_DR')]], left_index=True,
                         right_on=['index'],
                         how='left')
    df_output = df_output.drop(columns=['index'])

    return df_output


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
