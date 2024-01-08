import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
pd.set_option('display.expand_frame_repr', False)
churn_df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(churn_df.head())
print(churn_df.info())
#
print(churn_df.columns.values)
print(churn_df.dtypes)
#mask = churn_df['TotalCharges'].apply(isinstance, args=(str, ))
#print("mask:",  churn_df.loc[mask]['TotalCharges'])
churn_df.TotalCharges = pd.to_numeric(churn_df.TotalCharges, errors='coerce')
#print(churn_df.isnull().sum())

#Remove missing values
churn_df.dropna(inplace = True)
corr_matrix = churn_df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True)
plt.show()
#Drop customer IDs
df2 = churn_df.iloc[:,1:]
#Convert Churn to numeric
df2['Churn'].replace(to_replace='Yes', value=1, inplace=True)
df2['Churn'].replace(to_replace='No',  value=0, inplace=True)

#handle categorical variables
df_dummies = pd.get_dummies(df2, drop_first=True)
print(df_dummies.head())

plt.figure(figsize=(15,8))
df_dummies.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')
plt.show()

figure = (churn_df['gender'].value_counts()*100.0 /len(churn_df)).plot(kind='bar',
                                                                           stacked=True)
figure.yaxis.set_major_formatter(mtick.PercentFormatter())

plt.show()

figure = (churn_df['SeniorCitizen'].value_counts()*100.0 /len(churn_df))\
.plot.pie(autopct='%.1f%%', labels = ['No', 'Yes'])
figure.yaxis.set_major_formatter(mtick.PercentFormatter())

plt.show()

figure = sns.distplot(churn_df['tenure'], hist=True, kde=False,
             bins=int(180/5), color = 'darkblue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
figure.set_ylabel('# of Clients')
figure.set_xlabel('Tenure (months)')
figure.set_title('# of Clients by their tenure')

plt.show()

figure = churn_df['Contract'].value_counts().plot(kind = 'bar',rot = 0, width = 0.3)
figure.set_ylabel('# of Clients')
figure.set_title('# of Clients by Contract Type')

plt.show()

fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, sharey = True, figsize = (20,6))

figure = sns.distplot(churn_df[churn_df['Contract']=='Month-to-month']['tenure'],
                   hist=True, kde=False,
                   bins=int(180/5), color = 'turquoise',
                   hist_kws={'edgecolor':'black'},
                   kde_kws={'linewidth': 4},
                 ax=ax1)
figure.set_ylabel('# of Clients')
figure.set_xlabel('Tenure (months)')
figure.set_title('Month to Month Contract')

figure = sns.distplot(churn_df[churn_df['Contract']=='One year']['tenure'],
                   hist=True, kde=False,
                   bins=int(180/5), color = 'steelblue',
                   hist_kws={'edgecolor':'black'},
                   kde_kws={'linewidth': 4},
                 ax=ax2)
figure.set_xlabel('Tenure (months)',size = 14)
figure.set_title('One Year Contract',size = 14)

figure = sns.distplot(churn_df[churn_df['Contract']=='Two year']['tenure'],
                   hist=True, kde=False,
                   bins=int(180/5), color = 'darkblue',
                   hist_kws={'edgecolor':'black'},
                   kde_kws={'linewidth': 4},
                 ax=ax3)

figure.set_xlabel('Tenure (months)')
figure.set_title('Two Year Contract')
plt.show()

services = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
for i, item in enumerate(services):
    if i < 3:
        figure = churn_df[item].value_counts().plot(kind='bar', ax=axes[i, 0], rot=0)

    elif i >= 3 and i < 6:
        figure = churn_df[item].value_counts().plot(kind='bar', ax=axes[i - 3, 1], rot=0)

    elif i < 9:
        figure = churn_df[item].value_counts().plot(kind='bar', ax=axes[i - 6, 2], rot=0)
    figure.set_title(item)
plt.show()

churn_df[['MonthlyCharges', 'TotalCharges']].plot.scatter(x = 'MonthlyCharges',
                                                              y='TotalCharges')

plt.show()

colors = ['#4D3425','#E4512B']
figure = (churn_df['Churn'].value_counts()*100.0 /len(churn_df)).plot(kind='bar',
                                                                           stacked = True,
                                                                          rot = 0,
                                                                          color = colors,
                                                                         figsize = (8,6))
figure.yaxis.set_major_formatter(mtick.PercentFormatter())
figure.set_ylabel('% Clients',size = 14)
figure.set_xlabel('Churn',size = 14)
figure.set_title('Churn Rate', size = 14)

plt.show()

sns.boxplot(x = churn_df.Churn, y = churn_df.tenure)

plt.show()

colors = ['#4D3425','#E4512B']
contract_churn = churn_df.groupby(['Contract','Churn']).size().unstack()

figure = (contract_churn.T*100.0 / contract_churn.T.sum()).T.plot(kind='bar',
                                                                width = 0.3,
                                                                stacked = True,
                                                                rot = 0,
                                                                figsize = (10,6),
                                                                color = colors)
figure.yaxis.set_major_formatter(mtick.PercentFormatter())
figure.legend(loc='best',prop={'size':14},title = 'Churn')
figure.set_ylabel('% Clients',size = 14)

plt.show()

seniority_churn = churn_df.groupby(['SeniorCitizen','Churn']).size().unstack()

figure = (seniority_churn.T*100.0 / seniority_churn.T.sum()).T.plot(kind='bar',
                                                                width = 0.2,
                                                                stacked = True,
                                                                rot = 0,
                                                                figsize = (8,6),
                                                                color = colors)
figure.yaxis.set_major_formatter(mtick.PercentFormatter())
figure.legend(loc='center',prop={'size':14},title = 'Churn')
figure.set_ylabel('% Clients')

plt.show()

figure = sns.kdeplot(churn_df.MonthlyCharges[(churn_df["Churn"] == 'No') ],
                color="Red", shade = True)
figure = sns.kdeplot(churn_df.MonthlyCharges[(churn_df["Churn"] == 'Yes') ],
                ax =figure, color="Blue", shade= True)
figure.legend(["Not Churn","Churn"],loc='upper right')
figure.set_ylabel('Density')
figure.set_xlabel('Monthly Charges')

plt.show()

figure = sns.kdeplot(churn_df.TotalCharges[(churn_df["Churn"] == 'No') ],
                color="Red", shade = True)
figure = sns.kdeplot(churn_df.TotalCharges[(churn_df["Churn"] == 'Yes') ],
                ax =figure, color="Blue", shade= True)
figure.legend(["Not Churn","Churn"],loc='upper right')
figure.set_ylabel('Density')
figure.set_xlabel('Total Charges')

plt.show()

t = df_dummies['Churn'].values
X = df_dummies.drop(columns = ['Churn'])
corr_matrix = X.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, t_train, t_test = train_test_split(X, t, train_size=0.7, test_size=0.3, random_state=100)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, make_scorer
logreg = linear_model.LogisticRegression(C=1e5)
grid={"C":np.logspace(-3,3,7)}
print(np.logspace(-3,3,7))
# f1 = make_scorer(f1_score , average='weighted')
logreg_cv=GridSearchCV(logreg,grid,cv=5, scoring='f1')

logreg_cv.fit(X_train,t_train)

print("tuned hyperparameters :(best parameters) ",logreg_cv.best_params_)
print("Accuracy :",logreg_cv.best_score_)

Y_hat_test = logreg_cv.predict(X_test)
print("Accuracy score on the test set: ", accuracy_score(t_test, Y_hat_test))
print("F1 score on the test set: ", f1_score(t_test, Y_hat_test))

# Check for the VIF values
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(X_train, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
print(vif)
to_delete_idx = vif['VIF'] > 4

X_train = np.delete(X_train, [12], axis=1) # Es. delete features 12 with VIF=inf
X_test = np.delete(X_test, [12], axis=1)

logreg = linear_model.LogisticRegression(C=1e5)
grid={"C":np.logspace(-3,3,7)}

logreg_cv=GridSearchCV(logreg,grid,cv=5, scoring='accuracy')

logreg_cv.fit(X_train,t_train)

print("tuned hyperparameters :(best parameters) ",logreg_cv.best_params_)
print("Accuracy :",logreg_cv.best_score_)

t_hat_test = logreg_cv.predict(X_test)
print("Accuracy score on the test set: ", accuracy_score(t_test, t_hat_test))
print("F1 score on the test set: ", f1_score(t_test, t_hat_test))

vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(X_train, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
print(vif)

cm = confusion_matrix(t_test, t_hat_test)
print(cm)
print(t_test[0:50])
print(t_hat_test[0:50])
sns.heatmap(cm, annot=True,fmt='d')
plt.show()
