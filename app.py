import pandas as pd

data = {
    'Name': ['Rahul', 'Suresh', 'Gowtam'],
    'Age': [18,25,33]
}

df = pd.DataFrame(data)
print(df)

df.to_csv('data/data.csv', index=False)