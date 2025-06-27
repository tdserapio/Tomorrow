import pandas as pd 
  
# sample dataframe 
df = pd.DataFrame({'A': ['foo', 'bar', 'g2g', 'g2g', 'g2g', 
                         'bar', 'bar', 'foo', 'bar'], 
                  'B': ['a', 'b', 'a', 'b', 'b', 'b', 'a', 'a', 'b'] }) 

# frequency count of column A 
count = df['A'].value_counts() 
print("Absolute Frequencies:")
print(count)
print("-"*10)
print("Relative Frequencies:")
print(count / len(df) * 100)