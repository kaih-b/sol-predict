import pandas as pd
import numpy as np


df = pd.read_csv('players_master.csv')

# print(df.head()) # First 5 rows of the dataframe
# print(df.tail()) # Last 5 rows of the dataframe

# print(df.iloc[0:6, 3]) # First 5 rows of the fourth column in the dataframe
# print(df["PLAYER_NAME"].head()) # First 5 rows of the PLAYER_NAME column in the dataframe

# print(df[["PLAYER_NAME", "BASE_AGE", "BASE_FGM"]].head()) # First 5 rows of the PLAYER_NAME, BASE_AGE, and BASE_FGM columns in the dataframe

# numeric_df = df.select_dtypes(include="number") # Selects all numeric columns from the dataframe
# cat_df = df.select_dtypes(exclude="number") # Selects all non-numeric columns from the dataframe

# df.iloc[0:5, 0:3] = np.nan # Replaces the first 5 rows and the first 3 columns with NaN values
# df.loc[0:4, ["PLAYER_NAME", "BASE_AGE"]] = np.nan # Replaces the first 5 rows and the PLAYER_NAME and BASE_AGE columns with NaN values

# boolean comparisons work as with numpy arrays
# df_good = df[df['BASE_FGM'] > 10] # Selects rows where BASE_FGM is greater than 10
# player_list = []
# df_incl = df[df['PLAYER_NAME'].isin(player_list)] # Selects rows where PLAYER_NAME is in the player_list list
# df_high = df[df["BASE_FGM"] > df["BASE_FGM"].mean()] # Selects rows where BASE_FGM is greater than the mean of BASE_FGM
# df_hi, df_lo = low, high = df["BASE_FGM"].quantile([0.05, 0.95]) # Selects rows where BASE_FGM is within the 5th and 95th percentiles

# df_age_fgm = df[df["BASE_FGM"] > 10].sort_values("PLAYER_AGE", ascending=False) # Selects rows where BASE_FGM is greater than 10 and sorts the dataframe by PLAYER_AGE in descending order

# matching_rows = df[df['BASE_PTS'] == 28.9] # Selects rows where BASE_PTS is equal to 28.9 (Giannis Antetokounmpo)
# idx = matching_rows.index[0] # Gets the index of the first (only) row that matches the condition
# print(df.loc[idx, "PLAYER_NAME"])  # Prints the PLAYER_NAME of the row with index idx

# print(df.groupby('BASE_TEAM_ABBREVIATION')['BASE_PTS'].max().sort_values(ascending=False).head()) # Groups the dataframe by BASE_TEAM_ABBREVIATION, finds the maximum BASE_PTS for each team, sorts the results in descending order, and prints the top 5 teams with the highest BASE_PTS

# print(df['BASE_PTS'].describe()) # Prints the summary statistics for the BASE_PTS column