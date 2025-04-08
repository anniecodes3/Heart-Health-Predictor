#!/usr/bin/env python
# coding: utf-8

# # Heart Health Prediction

# #### Importing Data form heart.csv

# In[114]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
df=pd.read_csv("C:/Users/anush/Downloads/heart.csv")
df.head() 


# # Data Cleansing 

# #### Listing Null Values form data

# In[80]:


print(df.isnull().sum())


# ###### Droping Null values because Null values are very less as compared to data size

# In[81]:


df=df.dropna()


# # Data Visualization 

# ##### Give the info of Data Type

# In[46]:


df.info()


# ###### Describing the whole Data

# In[47]:


df.describe()


# In[82]:


get_ipython().system('pip install -U ydata-profiling')


# ##### Pandas Profiling Provide the Report of data set including Corelation

# In[85]:


from ydata_profiling import ProfileReport
print(ProfileReport)


# In[96]:


print(df.columns)


# In[97]:


df = df.drop(columns=['2'])


# In[104]:


profile = ProfileReport(df)
profile.to_notebook_iframe()


# In[127]:


d=df['Target'].value_counts()
print(d)


# ##### Heart Diseases Ratio in Dataset
# ###### Blue Graph indicate no heart desease and Orange Graph show Heart desease

# In[149]:


import seaborn as sns
import matplotlib.pyplot as plt

def plotTarget():
    plt.figure(figsize=(5, 2)) 
    ax = sns.countplot(x='Target', data=df, palette=['blue', 'orange']) 
    target_counts = df['Target'].value_counts().sort_index()  

    for i, p in enumerate(ax.patches):
        count = target_counts.iloc[i]  
        x = p.get_x() + p.get_width() / 2.0  
        y = p.get_height() + (0.02 * df.shape[0])  
        label = f'{(count / df.shape[0]) * 100:.2f}%'  
        ax.text(x, y, label, ha='center', fontsize=10, fontweight='bold', color='black')  

plotTarget()
plt.show()


# ##### Checking the corelation of data

# In[116]:


df.dtypes


# In[134]:


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

df['ChestPain'] = encoder.fit_transform(df['ChestPain'])
df['Thal'] = encoder.fit_transform(df['Thal'])


# In[135]:


df.corr()


# ##### Select Age as most dependent data on label 
# ###### Disease Probability Bar Plot

# In[156]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

fig_age, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 8))

sns.kdeplot(
    data=df[df['Target'] == 0], x="Age", fill=True, ax=axes[0], label='Disease false', color='steelblue'
)
sns.kdeplot(
    data=df[df['Target'] == 1], x="Age", fill=True, ax=axes[0], label='Disease true', color='coral'
)

axes[0].set(xlabel='Age', ylabel='Density')
axes[0].legend(title='Disease')

avg = df[["Age", "Target"]].groupby(['Age'], as_index=False).mean()

palette = sns.color_palette("Spectral", len(avg))  # Can also try "husl", "viridis", etc.
sns.barplot(x='Age', y='Target', data=avg, ax=axes[1], palette=palette)

axes[1].set(xlabel='Age', ylabel='Disease probability')

plt.tight_layout()
plt.show()


# ###### Checking For Categorical Data

# In[161]:


chest_pain_types = ['Typical', 'Asymptomatic', 'Nonanginal']
df['ChestPain'] = df[chest_pain_types].idxmax(axis=1)
print(df['ChestPain'].value_counts())


# In[166]:


df['Thal'] = df['Thal_Normal'].apply(lambda x: 'Normal' if x == 1 else 'Other')
print(df['Thal'].value_counts())


# ##### Ploting Function For Categorical Data " Chest Pain" && "Thalassemia"
# ##### Ploting Function For Continoius Data

# In[177]:


import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Define your lists based on your actual df.columns
category = [
    ('ChestPain', ['typical', 'nontypical', 'nonanginal', 'asymptomatic']),
    ('Thal', ['fixed', 'normal', 'reversable'])
]

continuous = [
    ('Age', 'Age in year'),
    ('Sex', '1 for Male 0 for Female'),
    ('RestBP', 'BP in Rest State'),
    ('Chol', 'serum cholestoral in mg/d'),
    ('Fbs', 'Fasting blood glucose'),
    ('MaxHR', 'Max Heart Rate'),
    ('ExAng', 'Exercise Induced Angina'),
    ('Oldpeak', 'ST depression by exercise relative to rest'),
    ('Ca', '# major vessels (0-3) colored by flourosopy')
]

# ✅ Plot functions
def plotCategorial(attribute, labels, ax_index):
    sns.countplot(x=attribute, data=df, ax=axes[ax_index][0])
    sns.countplot(x='Target', hue=attribute, data=df, ax=axes[ax_index][1])
    
    avg = df[[attribute, 'Target']].groupby([attribute], as_index=False).mean()
    sns.barplot(x=attribute, y='Target', hue=attribute, data=avg, ax=axes[ax_index][2])
    
    legend1 = axes[ax_index][1].get_legend()
    legend2 = axes[ax_index][2].get_legend()
    
    if legend1:
        for t, l in zip(legend1.texts, labels):
            t.set_text(l)
    if legend2:
        for t, l in zip(legend2.texts, labels):
            t.set_text(l)

def plotContinuous(attribute, xlabel, ax_index):
    sns.histplot(df[attribute], ax=axes[ax_index][0], kde=True)
    axes[ax_index][0].set(xlabel=xlabel, ylabel='Density')
    sns.violinplot(x='Target', y=attribute, data=df, ax=axes[ax_index][1])

def plotGrid(isCategorial):
    if isCategorial:
        [plotCategorial(x[0], x[1], i) for i, x in enumerate(category)]
    else:
        [plotContinuous(x[0], x[1], i) for i, x in enumerate(continuous)]


# ##### Categorical Plot

# In[174]:


fig_categorial, axes = plt.subplots(nrows=len(category), ncols=3, figsize=(18, 5 * len(category)))
plotGrid(isCategorial=True)


# ##### Continuous Plot

# In[175]:


fig_continuous, axes = plt.subplots(nrows=len(continuous), ncols=2, figsize=(15, 5 * len(continuous)))
plotGrid(isCategorial=False)


# In[179]:


# ✅ Save the plots
fig_categorial.savefig("categorial_plots.png", bbox_inches='tight')
fig_continuous.savefig("continuous_plots.png", bbox_inches='tight')

# ✅ Create HTML report
html = """
<html>
<head>
    <title>Heart Health Data EDA Report</title>
</head>
<body style="font-family: Arial; text-align: center;">
    <h1 style="color:#2c3e50;">Heart Health Data - EDA Report</h1>

    <h2 style="color:#34495e;">Categorical Variables</h2>
    <img src="categorial_plots.png" width="95%" /><br><br>

    <h2 style="color:#34495e;">Continuous Variables</h2>
    <img src="continuous_plots.png" width="95%" />
</body>
</html>
"""

# ✅ Save HTML file
with open("EDA_Report.html", "w", encoding='utf-8') as file:
    file.write(html)

print("✅ HTML report saved as EDA_Report.html")

# ✅ Try to open in browser (if you're not in JupyterLab)
import os
import webbrowser

file_path = os.path.abspath("EDA_Report.html")
webbrowser.open("file://" + file_path)


# ##### Creating Dummy

# In[144]:


#dummy for chest Pain
chestpain_dummy = pd.get_dummies(df.loc[:,'ChestPain'])
chestpain_dummy.rename(columns={1: 'Typical', 2: 'Asymptomatic',3: 'Nonanginal', 4: 'Nontypical'}, inplace=True)
#dummy for RestECG
restecg_dummy = pd.get_dummies(df.loc[:,'RestECG'])
restecg_dummy.rename(columns={0: 'Normal_restECG', 1: 'Wave_abnormal_restECG',2:'Ventricular_ht_restECG'},inplace=True)
#dummy for Slope
slope_dummy = pd.get_dummies(df['Slope'])
slope_dummy.rename(columns={1: 'Slope_upsloping', 2:'Slope_flat',3: 'Slope_downsloping'},inplace=True)
#dummy for Thal
thal_dummy = pd.get_dummies(df['Thal'])
thal_dummy.rename(columns={3: 'Thal_Normal', 6: 'Thal_fixed',7: 'Thal_reversible'}, inplace=True)
#concatination in data frame
df = pd.concat([df,chestpain_dummy, restecg_dummy, slope_dummy, thal_dummy], axis=1)
#droping Column because their dummies are created
df.drop(['ChestPain','RestECG', 'Slope', 'Thal'], axis=1, inplace=True)


# ##### Checking the No object will left 

# In[150]:


df.columns


# In[151]:


# Drop the unnamed (duplicate) dummy columns for Thal
df.drop(columns=[0, 1, 2], inplace=True)


# In[152]:


print(df.columns)


# In[154]:


df.info()


# ##### Checking dataset

# In[180]:


df.head()


# ##### Selecting label means selected Column to predict in df_X and input column in df_y

# In[66]:


df_X= df.loc[:, df.columns != 'Target']
df_y= df.loc[:, df.columns == 'Target']


# # Model Training 

# #### Recursive Feature Elimination

# In[181]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder

df_selected_X = df[['Age', 'Sex', 'RestBP', 'Chol', 'Fbs', 'MaxHR', 'ExAng', 'Oldpeak', 'Ca', 
                    'Typical', 'Asymptomatic', 'Nonanginal',  # ChestPain categories
                    'Normal_restECG', 'Wave_abnormal_restECG', 'Ventricular_ht_restECG',  # RestECG categories
                    'Slope_upsloping', 'Slope_flat', 'Slope_downsloping']]
df_selected_y = df['Target']

print(df_selected_X.dtypes)
print(df_selected_X.head())

categorical_cols = df_selected_X.select_dtypes(include=['object']).columns
print(f"Categorical columns: {list(categorical_cols)}")

for col in categorical_cols:
    if df_selected_X[col].nunique() == 2:  
        le = LabelEncoder()
        df_selected_X[col] = le.fit_transform(df_selected_X[col])
    else:  
        df_selected_X = pd.get_dummies(df_selected_X, columns=[col], drop_first=True)

df_selected_X = df_selected_X.apply(pd.to_numeric, errors='coerce')
df_selected_X.fillna(df_selected_X.median(), inplace=True)

df_selected_X = df_selected_X.astype(int)

df_selected_X = sm.add_constant(df_selected_X)

lm = sm.Logit(df_selected_y, df_selected_X)
result = lm.fit()

print(result.summary())


# ##### Spliting Values into test and training Dataset in the ratio 0.75:0.25

# In[182]:


from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test=train_test_split(df_selected_X,df_selected_y, test_size = 0.25, random_state =0)
columns = X_train.columns


# ##### Calculating Accuracy Function of the Models

# In[190]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def cal_accuracy(y_test, y_predict): 
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_predict)) 
    print("\nClassification Report:\n", classification_report(y_test, y_predict))
    print(f"\nAccuracy: {accuracy_score(y_test, y_predict) * 100:.3f}%")


# In[188]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def cal_accuracy(y_test, y_predict): 
    cm = confusion_matrix(y_test, y_predict)
    
    print("\nConfusion Matrix:\n", cm) 
    print("\nClassification Report:\n", classification_report(y_test, y_predict))
    print(f"\nAccuracy: {accuracy_score(y_test, y_predict) * 100:.3f}%")
    
    # Heatmap visualization
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()


# # Logistic Regression

# In[187]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

# Update Logistic Regression model
lr = LogisticRegression(max_iter=1000, solver='newton-cg')
lr.fit(X_train, y_train)

y_predict = lr.predict(X_test)

print(f"Accuracy of Test Dataset: {lr.score(X_test, y_test):.3f}")
print(f"Accuracy of Train Dataset: {lr.score(X_train, y_train):.3f}")


# ##### Vale Prediction for Test dataset for Logistic Regression

# In[186]:


print("Predicted values:") 
print(y_predict)
cal_accuracy(y_test, y_predict)


# # Support Vector Machine

# In[72]:


from sklearn import svm
svm_linear = svm.SVC(kernel='linear')
svm_linear.fit(X_train,y_train)
warnings.simplefilter('ignore')
print(f"Accuracy of Test Dataset: {svm_linear.score(X_test,y_test):0.3f}")
print(f"Accuracy of Train Dataset: {svm_linear.score(X_train,y_train):0.3f}")


# ##### Vale Prediction for Test dataset for SVM

# In[73]:


print("Predicted values:") 
print(y_predict)
cal_accuracy(y_test, y_predict)


# # Decision Tree

# In[74]:


from sklearn.tree import DecisionTreeClassifier
gini = DecisionTreeClassifier(criterion = "gini", random_state =100,max_depth=3, min_samples_leaf=5)
gini.fit(X_train, y_train)
warnings.simplefilter('ignore')
print(f"Accuracy of Test Dataset: {gini.score(X_test,y_test):0.3f}")
print(f"Accuracy of Train Dataset: {gini.score(X_train,y_train):0.3f}")


# ##### Vale Prediction for Test dataset for Decision Tree

# In[75]:


y_predict=gini.predict(X_test) 
print("Predicted values:\n")
print(y_predict) 
cal_accuracy(y_test, y_predict)


# # Random Forest

# In[228]:


from sklearn.ensemble import RandomForestClassifier

forest=RandomForestClassifier(n_estimators=100)
forest.fit(X_train,y_train)

warnings.simplefilter('ignore')
print(f"Accuracy of Test Dataset: {forest.score(X_test,y_test):0.3f}")
print(f"Accuracy of Train Dataset: {forest.score(X_train,y_train):0.3f}")


# ##### Over Fitting Issue
# ##### Vale Prediction for Test dataset for Rondom Forest

# In[230]:


print("Predicted values:\n")
print(y_predict)
cal_accuracy(y_test, y_predict, 'Random Forest')


# # Cross Validation For Models

# In[248]:


import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.simplefilter('ignore')
X_train, X_test, y_train, y_test = train_test_split(df_selected_X, df_selected_y, test_size=0.25, random_state=0)

X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

def cal_accuracy(y_true, y_pred, model_name):
    print(f"\n===== {model_name} =====")
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print(f"\nAccuracy: {accuracy_score(y_true, y_pred) * 100:.3f}%")

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
def cross_validate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
    print(f"Cross-Validation Accuracies for each fold: {np.round(scores * 100, 2)}")
    print(f"\nMean CV Accuracy: {np.mean(scores) * 100:.2f}%")
    print(f"Standard Deviation: {np.std(scores) * 100:.2f}%\n")
    return scores
lr = LogisticRegression(max_iter=1000, solver='newton-cg')
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
cal_accuracy(y_test, y_pred_lr, "Logistic Regression")
cross_validate_model(lr, df_selected_X, df_selected_y)

svm_linear = svm.SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
y_pred_svm = svm_linear.predict(X_test)
cal_accuracy(y_test, y_pred_svm, "Support Vector Machine")
cross_validate_model(svm_linear, df_selected_X, df_selected_y)

dt = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
cal_accuracy(y_test, y_pred_dt, "Decision Tree")
cross_validate_model(dt, df_selected_X, df_selected_y)

rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
cal_accuracy(y_test, y_pred_rf, "Random Forest")
cross_validate_model(rf, df_selected_X, df_selected_y)


# In[247]:


import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

y_pred = [lr.predict(X_test), svm_linear.predict(X_test), dt.predict(X_test), rf.predict(X_test)]
models = ['Logistic Regression', 'SVM', 'Decision Tree', 'Random Forest']
metrics = {
    'Test Accuracy': [lr.score(X_test, y_test), svm_linear.score(X_test, y_test), dt.score(X_test, y_test), rf.score(X_test, y_test)],
    'Precision': [precision_score(y_test, y) for y in y_pred],
    'Recall': [recall_score(y_test, y) for y in y_pred],
    'F1-Score': [f1_score(y_test, y) for y in y_pred]
}
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for ax, (metric, values) in zip(axes.ravel(), metrics.items()):
    ax.bar(models, values, color='skyblue')
    ax.set_title(metric)
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric)

plt.tight_layout()
plt.show()


# ### Best Model for Dataset is Linear Regression
