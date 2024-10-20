import pandas as pd

def data_cleaning(df):
   df.drop(columns=['profile_id'], inplace=True,errors='ignore') 
   df.replace([''], pd.NA, inplace=True)
   numeric_columns = ['u_q', 'coolant', 'u_d', 'motor_speed', 'i_d', 'i_q', 'ambient', 'pm']
   for column in numeric_columns:
      if column in df.columns:
         df[column] = pd.to_numeric(df[column], errors='coerce')
   Not_Nan=['u_d','i_d','i_q','motor_speed']

   for mean_column in Not_Nan:
      if column in df.columns:
         mean_value = df[mean_column].mean()
         df[mean_column].fillna(mean_value, inplace=True)

   return df