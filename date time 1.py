import pandas as pd
df=pd.read_csv("C:/Users/gshik/Desktop/at1.csv")
date=input("Enter date- dd/mm/yyyy")
df[date]='A'  #add new date column
df.to_csv("C:/Users/gshik/Desktop/at1.csv",index=False)
df