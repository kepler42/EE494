import pandas as pd
import numpy as np 
import os, glob

 
list_of_files = glob.glob(os.path.join("/home/kepler42/EE494/EE494/EE494_Data/", "EE494_data_*.csv"))

df_from_each_file = (pd.read_csv(f, sep=',',header=None) for f in list_of_files)


df_merged   = pd.concat(df_from_each_file, ignore_index=True)
df_merged.to_csv( "merged3.csv")
