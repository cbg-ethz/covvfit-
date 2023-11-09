"""utilities to preprocess relative abundances"""
import pandas as pd
import numpy as np


# variants = [
#     'B.1.1.7', 'B.1.351', 'P.1', 'undetermined',
#     'B.1.617.2', 'BA.1', 'BA.2', 'BA.4', 'BA.5', 'BA.2.75',
#     'BQ.1.1', 'XBB.1.5', 'XBB.1.9', 'XBB.1.16', 'XBB.2.3', 'EG.5', "BA.2.86"
# ]

# variants2 = [
#     'BA.4', 'BA.5', 'BA.2.75',
#     'BQ.1.1', 'XBB.1.5', 'XBB.1.9', 'XBB.1.16', 'XBB.2.3', 'EG.5', "BA.2.86"
# ]


# variants3 = [
#     'BA.2.75', 'BA.5', 'BQ.1.1',
#     'XBB.1.5', 'XBB.1.9', 'XBB.1.16', 'XBB.2.3', 'EG.5',
#     'BA.2.86',
# ]

# variants4 = [
#     'XBB.1.5', 'XBB.1.9', 'XBB.1.16', 'XBB.2.3', 'EG.5', "BA.2.86"
# ]

# variants5 = [
# #     'B.1.1.7', 'B.1.351', 'P.1', 'undetermined',
#     'B.1.617.2', 'BA.1', 'BA.2', 'BA.4', 'BA.5', 'BA.2.75',
#     'BQ.1.1', 'XBB.1.5', 'XBB.1.9', 'XBB.1.16', 'XBB.2.3', 'EG.5', "BA.2.86"
# ]

# cities = ['Lugano (TI)', 'Zürich (ZH)', 'Chur (GR)', 'Altenrhein (SG)',
#        'Laupen (BE)', 'Genève (GE)', 'Basel (BS)', 'Porrentruy (JU)',
#        'Lausanne (VD)', 'Bern (BE)', 'Luzern (LU)', 'Solothurn (SO)',
#        'Neuchâtel (NE)', 'Schwyz (SZ)']

def load_data(file):
	wwdat = pd.read_csv(file)
	wwdat = wwdat.rename(columns={wwdat.columns[0]: 'time'})
	return wwdat


def preprocess_df(
	df,
	cities,
	variants, 
	undertermined_thresh=0.01, 
	zero_date='2023-01-01', 
	date_min=None, 
	date_max=None,
	):
	# Convert the 'time' column to datetime
	df['time'] = pd.to_datetime(df['time'])

	# Remove days with too high undetermined
	df = df[df['undetermined'] < undertermined_thresh]

	# Subset the 'BQ.1.1' column
	df = df[['time', 'city'] + variants]

	# Subset only the specified cities
	df = df[df['city'].isin(cities)]

	# Create a new column which is the difference in days between zero_date and the date
	df['days_from'] = (df['time'] - pd.to_datetime(zero_date)).dt.days

	# Subset dates 
	if date_min is not None:
		df = df[df['time'] >= pd.to_datetime(date_min)]
	if date_max is not None:
		df = df[df['time'] < pd.to_datetime(date_max)]


	return df


def make_data_list(df, cities, variants):
	ts_lst = [df[(df.city == city)].days_from.values for city in cities]
	ys_lst = [df[(df.city == city)][variants].values.T for city in cities]

	return (ts_lst, ys_lst)


