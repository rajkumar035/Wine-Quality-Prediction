#import libararies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn import metrics
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

wine_data = pd.read_csv('winequalityN.csv')
wine_data.head()
wine_dataa = wine_data.rename(columns={'fixed acidity': 'fixed_acidity', 'volatile acidity': 'volatile_acidity', 'citric acid': 'citric_acid','residual sugar': 'residual_sugar', 'free sulfur dioxide': 'free_sulfur_dioxide', 'total sulfur dioxide':'total_sulfur_dioxide'})
wine_dataa.head()
wine_dataa.quality.value_counts()
wine_dataa['pH'].unique()
wine_dataa.count()
wine_dataa.info()
wine_dataa.isnull().any()
wine_dataa.describe()

zeromask = np.zeros_like(wine_dataa.corr()) #creates an array of zeros
triangle_indices = np.triu_indices_from(zeromask)
zeromask[triangle_indices] = True

plt.figure(figsize=(16,10))
sns.heatmap(wine_dataa.corr(), mask=zeromask, annot=True, annot_kws={'size': 14})
sns.set_style('whitegrid')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

quality = wine_dataa['quality']
features = wine_dataa.drop(['quality'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(features, quality, test_size=0.2, random_state = 10)

#% of training set
print('Train: ', len(X_train)/len(features))

# % of test data
X_test.shape[0]/features.shape[0]
regr = LinearRegression()
regr.fit(X_train, y_train)
print('Training data r-squared:', regr.score(X_train, y_train))
print('Test data r-squared:', regr.score(X_test, y_test))

print('Intercept', regr.intercept_)
pd.DataFrame(data=regr.coef_, index=X_train.columns, columns=['coef'])

X_incl_const = sm.add_constant(X_train)
model = sm.OLS(y_train, X_incl_const)
results = model.fit()

pd.DataFrame({'coef': results.params, 'p values': round(results.pvalues, 3)})
print(results.summary())

vif = [variance_inflation_factor(exog=X_incl_const.values, exog_idx=i) for i in range(X_incl_const.shape[1])]  #empty list
pd.DataFrame({'coef_name': X_incl_const.columns,'vif':np.around(vif, 2)})

X_incl_const = sm.add_constant(X_train) 
model = sm.OLS(y_train, X_incl_const) 
results = model.fit() 

org_coef = pd.DataFrame({'coef': results.params, 'p values': round(results.pvalues, 3)})
print('BIC is', results.bic)
print('r-squared is', results.rsquared)

full_normal_mse = round(results.mse_resid, 3)
full_normal_rmse = round(np.sqrt(full_normal_mse))
full_normal_rsquared = round(results.rsquared, 3)

print("Full Normal Mean Square Error", full_normal_mse)
print("Full Normal Root Mean Square Error", full_normal_rmse)
print("Full Normal R-Squared", full_normal_rsquared)

wine_table = []

for i in wine_dataa.quality:
    if i <= 4:
        wine_table.append(0) #Table Wine
    elif i >= 7:
        wine_table.append(2) #Fine Wine
    else:
        wine_table.append(1) #Premium Wine
        
wine_dataa['label'] = wine_table
wine_dataa[:20]

# Import train_test_split function

X=wine_dataa.drop(['label','quality'], axis=1)# Features
y=wine_dataa['label']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 

rfc=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets 
rfc.fit(X_train,y_train)

#Perform prediction on the test set
y_pred=rfc.predict(X_test)

print("Classification Report:",classification_report(y_test, y_pred))
print("Confusion Metrics :",confusion_matrix(y_test, y_pred))
print('F1 Score: ', f1_score(y_test, y_pred, average='micro'))
print('Precision Score:', metrics.precision_score(y_test, y_pred, average="micro"))
print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))

y_test, y_pred
df = pd.DataFrame(list(zip(y_test, y_pred)), columns =['Y_test', 'Y_pred']) 
print(df[:20])
