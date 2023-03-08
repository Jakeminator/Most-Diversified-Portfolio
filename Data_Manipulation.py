import pandas as pd
import openpyxl
from pathlib import Path

#Define input and outpu paths --> Remember to name the correct CSV file and the name of the Excel outpuy
input_path = Path('/Users/jacobhenrichsen/iCloud Drive (arkiv) - 1/Documents/CBS/Master/3. Semester/Thesis/Code/Data/USA all themes daily data.csv')
output_path = Path('/Users/jacobhenrichsen/iCloud Drive (arkiv) - 1/Documents/CBS/Master/3. Semester/Thesis/Code/Data/Manipulated data/Data.xlsx')
Kenneth_french = Path('/Users/jacobhenrichsen/iCloud Drive (arkiv) - 1/Documents/CBS/Master/3. Semester/Thesis/Code/Data/Manipulated data/Kenneth_French_data.xlsx')
output_path2 = Path('/Users/jacobhenrichsen/iCloud Drive (arkiv) - 1/Documents/CBS/Master/3. Semester/Thesis/Code/Data/Manipulated data/Merged data.xlsx')

# Read in the CSV file
df = pd.read_csv(input_path)

# Sort the data by the date column
df = df.sort_values('date')

# Pivot the data so that each stock has its own column
df = df.loc[:, ['name', 'date', 'ret']]
df = df.pivot(index='date', columns='name', values='ret')

# Convert the date column to a date format that Excel recognizes
df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
df.index.name = 'date'
df.reset_index(inplace=True)
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d').dt.strftime('%d/%m/%Y')

# Print the resulting dataframe
print(df)

# Save the dataframe as an Excel file
df.to_excel(output_path, index=False)

# Print a message to confirm that the file was saved
print("File saved as Excel'")

#Begin the data mergin process
# read the first Excel file into a dataframe
df1 = pd.read_excel(output_path)

# read the second Excel file into a dataframe
df2 = pd.read_excel(Kenneth_french)

# merge the two dataframes based on the date column
merged_df = pd.merge(df1, df2, on='date', how='inner')

# save the merged dataframe to a new Excel file
merged_df.to_excel(output_path2, index=False)