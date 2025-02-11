# Machine Learning: Using the Iris Dataset!  
Code:  
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris

# Load the data
df = load_iris()

#print(df)

dataset = pd.DataFrame(df.data, columns=df.feature_names)
dataset['label'] = df.target

print(dataset.head())
```

Today we will be beginning to use the Iris Dataset to classify which species of Iris(or label) using the dimensions of a input flower.  

Here's df.head:
```
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  label
0                5.1               3.5                1.4               0.2      0
1                4.9               3.0                1.4               0.2      0
2                4.7               3.2                1.3               0.2      0
3                4.6               3.1                1.5               0.2      0
4                5.0               3.6                1.4               0.2      0
```

We will be covering more in the future classes.  
