import numpy as np
import pandas as pd


def process_data():
    # read in data
    df = pd.read_csv('datasets/lead_convert.csv', index_col=0)

    # improve column name
    df = df.rename(
        columns={
            'lead_ip_country_code': 'lead_country_of_registration'})

    # filling the missing values with the mode
    df['lead_utm_medium'] = df['lead_utm_medium'].fillna(
        df['lead_utm_medium'].mode()[0])

    # Creating converted from conversion revenue
    df['converted'] = np.repeat(df.conversion_revenue.values, 1)
    df['converted'] = pd.Series(
        np.where(
            df['converted'].values == 0,
            0,
            1),
        df.index)

    # creating time since registraion from hour of registration
    df['Time since registration'] = pd.cut(df['hours_since_registration'],
                                           [0, 17, 190, 805, 9490],
                                           labels=[
                                            'Under one day',
                                            'A day to a week',
                                            'A week to a month',
                                            'A month to over a year'],
                                           right=False, include_lowest=True)

    # creating weekday and weekend from day of week
    df['lead_weekday_of_registration'] = np.where(
        (df['lead_weekday_of_registration']) < 5, 'weekday', 'weekend')

    # creating lead time of registraion from hour of registration
    df['lead_time_of_registration'] = pd.cut(df['lead_hour_of_registration'],
                                        [0, 5, 9, 14, 20, 24], labels=['Dawn',
                                        'Morning', 'Afternoon', 'Evening',
                                    'Night'], right=False, include_lowest=True)

    # dropping unneccesary columns
    drop_cols = ['lead_hour_of_registration', 'redirect_hour',
        'redirect_weekday', 'redirect_month_day', 'hours_since_last_revenue',
            'conversion_revenue', 'hours_since_registration',
            'different_redirect_sources']
    df = df.drop(columns=drop_cols)

    return df
