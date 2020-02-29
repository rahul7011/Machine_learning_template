# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values    #only the independent one i.e,0 till n-1
y = dataset.iloc[:, 3].values     #only the dependent i.e, the last one!


#Taking care of the missing data!
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer=imputer.fit(X[:,1:3])       #to avoid the dataleakage(testing vs train data)
X[:,1:3]=imputer.transform(X[:,1:3])

#Encoding categorical data(categorized data)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])

# =============================================================================
#       OLD CODE NOT IN SUPPORT NOW
# onehotencoder=OneHotEncoder(categorical_features=[0])
# X=onehotencoder.fit_transform(X).toarray()
# =============================================================================

    #dummy encoding
transformer = ColumnTransformer([('onehotencoder', OneHotEncoder(),[0])],remainder='passthrough')
X = np.array(transformer.fit_transform(X), dtype=np.float)
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling(-1 to 1)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# =============================================================================
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train)
# =============================================================================

