import pandas as pd
from sklearn.cross_validation import cross_val_score
import numpy as np
data=pd.read_csv(r"C:\Users\acer\Desktop\Advertising.csv",index_col=0)
data.head()
data.tail()
# =============================================================================
# Here TV,radio and newspapers are used as features object and sales as response object.
#since response is continuous its a regression problem
# =========================Visualizing the confidence Band====================================================
import seaborn as sns
sns.pairplot(data,x_vars=["TV","radio","newspaper"],y_vars="sales",size=7,aspect=0.7,kind="reg")
feature_cols=["TV","radio","newspaper"]#Modelling process
X=data[feature_cols]
X.head()
y=data["sales"]
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)#default is 25%
from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(X_train,y_train)
zip(feature_cols,linreg.coef_)

#1 from sklearn import metrics
#1 import numpy as np
#1 y_pred=linreg.predict(X_test)

root_mean_sq=np.sqrt(-(cross_val_score(linreg,X,y,cv=10,scoring='mean_squared_error'))).mean()
#print((scores2))
#inv_score=-scores2
#rmse=np.sqrt(inv_score)
#all the above shit is simplified into root_mean_sq
print(root_mean_sq)
# model excluding newspaper is betaa
#1 print(np.sqrt(metrics.mean_squared_error(y_pred,y_test)))
#here in case of regression lower values of RMS,MS,MAe aae betta 