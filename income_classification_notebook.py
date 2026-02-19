#!/usr/bin/env python
# coding: utf-8

# # Income Classification
# 

# In[1]:


"""
Income Classification Script
Predict whether income >50K or <=50K using demographic and employment data.
All plots use seaborn for consistent styling.
"""
# =========================================================================
# Import Python Libraries
#
# ==========================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,confusion_matrix,
                             f1_score, roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')
sns.set_style('ticks')
sns.set_palette('Set2')
sns.set_context('poster')
plt.rc('font', size=14, weight='bold')
plt.rc('axes', labelsize=14, titlesize=14, labelweight='bold')
plt.rc('legend', fontsize=14, title_fontsize=16)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')
# Create output directory
output_dirs = ['plots_classification_outputs', 'models_classification_outputs']
for dir in output_dirs:
    os.makedirs(dir, exist_ok=True)




# ## Load Census Bureau Dataset
# 

# In[2]:


# =============================================================================
# Load and Explore Data
# =============================================================================
# Load the dataset columns
col_filename = 'census-bureau.columns'
# Load column names
with open(col_filename, 'r') as f:
    column_list = [line.strip() for line in f.readlines()]
print(column_list)

# Load the dataset
filename = 'census-bureau.data'
data = pd.read_csv(filename, header=None, names=column_list)
#data.duplicated()
# Remove duplicates
#data.drop_duplicates(inplace=True)

class_features =  ['age', 'class of worker', 'detailed industry recode', 'detailed occupation recode', 'education', 'wage per hour',
  'race', 'hispanic origin', 'sex', 'full or part time employment stat', 'capital gains', 'capital losses', 'dividends from stocks', 'tax filer stat',
 'detailed household and family stat', 'detailed household summary in household',  'country of birth self',
 'citizenship', 'own business or self employed', 'veterans benefits', 'weeks worked in year', 'weight', 'year', 'label']
df = data.drop_duplicates(ignore_index=True)

df['label'] = df['label'].replace({'- 50000.':'<=50K', '50000+.':'>50K'})
# Display basic info
print("Data loaded. Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
# print("\nFirst 5 rows:")
# print(df.head())


# In[3]:


# Dataset Summary
print("\nDataset Summary:")
print(df.info())

# Dataset Summary Statistics
print("\nSummary Statistics:")
print(df.describe(include='all'))


# Clean target
# Map label to binary
df['income'] = df['label'].apply(lambda x: 1 if x.strip() == '>50K' else 0)
print(df['income'].value_counts(normalize=True))

# Replace '?' , 'Not in universe or children', 'Not in universe under 1 year old', and 'Not in universe' with NaN for easier handling
na_values = ['?', 'Not in universe',
              'Not in universe or children', 'Not in universe under 1 year old']
df.replace(na_values, np.nan, inplace=True)


# In[4]:


df.describe()


# In[5]:


print("\nMissing Values:\n", df.isnull().sum())

# Plotting missing values
plt.figure(figsize=(8, 6))
sns.countplot(x=df.isnull().sum(), palette='Set2')
plt.title('Missing Values Count')
plt.ylabel("Number of Columns")
plt.savefig(f'{output_dirs[0]}/missing_distribution.png')
plt.close()


# ## Exploratory Data Analysis
# ### Univariate Analysis

# In[6]:


print("\n=== Exploratory Data Analysis ===")
def histogram_boxplot(data, feature, figsize=(10, 8), kde=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (15,10))
    kde: whether to show the density curve (default False)
    bins: number of bins for histogram (default None)
    """
    fig, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  # boxplot will be created and a triangle will indicate the mean value of the column
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram
    plt.suptitle(f'Distribution of {feature}', fontsize=16, fontweight='bold')
    return fig

def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(data[feature])  # length of the column
    count = data[feature].nunique()
    if n is None:
        fig = plt.figure(figsize=(count + 1, 8))
        n = count
    else:
        fig = plt.figure(figsize=(n + 1, 8))

    plt.xticks(rotation=75, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n].sort_values(),
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage
    top_col = '' if count < n else f'(Top {n} categories)'
    plt.title(f'Distribution of {feature} {top_col}', fontsize=16, fontweight='bold')
    #plt.show()  # show the plot
    return fig


# In[7]:


# Target distribution
# Map label to binary income feature
df['income'] = df['label'].apply(lambda x: 1 if x.strip() == '>50K' else 0)
print(df['income'].value_counts(normalize=True))
explode = (0.05, 0.1)

plt.figure(figsize=(9,6))
df['income'].value_counts().plot.pie(autopct='%1.1f%%', labels=['<=50K', '>50K'], shadow=True, startangle=90,
              textprops={'fontsize': 11, 'fontweight': 'bold'}, explode=explode)
plt.title('Income Class Distribution')
plt.ylabel('')
plt.savefig(f'{output_dirs[0]}/target_distribution.png')
plt.show()
plt.close()
print("Target Class distribution saved.")

backup_df = df.copy( )


# In[8]:


# Missing data analysis
# Replace '?' , 'Not in universe or children', 'Not in universe under 1 year old', and 'Not in universe' with NaN for easier handling
na_values = ['?', 'Not in universe',
              'Not in universe or children', 'Not in universe under 1 year old']
df.replace(na_values, np.nan, inplace=True)

missing_frac = df.isnull().mean() * 100
missing_frac = missing_frac[missing_frac > 0].sort_values(ascending=False)
top_missing = 20
print(f"Top {top_missing} missing columns (% missing):")
print(missing_frac.head(top_missing))

# Plot missing percentages
plt.figure(figsize=(10,6))
missing_frac.head(top_missing).plot(kind='barh')
plt.xlabel('Percent Missing')
plt.title(f'Top {top_missing} Columns with Highest Missing Percentage')
plt.tight_layout()
plt.savefig(f'{output_dirs[0]}/missing_percentages.png')
plt.show()
plt.close()


# In[13]:


# Drop columns with >40% missing
fraction = 70
missing_frac = (df.isnull().mean()*100).sort_values(ascending=False)
print(f"Columns with >{fraction}% missing:\n", missing_frac[missing_frac > fraction])
high_missing_cols = missing_frac[missing_frac > fraction].index.tolist()
# country_col = ['country of birth father', 'country of birth mother', 'country of birth self']
df.drop(columns=high_missing_cols, inplace=True)
print(f"Dropped {len(high_missing_cols)} columns with >{fraction}% missing.")
print(f"Dropped columns: {high_missing_cols}")
print("Remaining columns:", df.columns.tolist())
print("\nMissing Values After Dropping:\n", df.isnull().sum())

# Numeric features distribution
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Remove target and weight if present
numeric_cols = [c for c in numeric_cols if c not in ['income', 'weight']]

num_rows = len(numeric_cols)//3 + 1 if len(numeric_cols) % 3 != 0 else len(numeric_cols)//3

for i, col in enumerate(numeric_cols):
    #histogram_boxplot(df, col, kde=True)
    sns.histplot(data=df, x=col, hue='label', stat="count",kde=True) #"count", "density", "percent", "probability", "proportion", "frequency"
    col_name = col.replace(' ', '_')
    plt.tight_layout()
    plt.savefig(f'{output_dirs[0]}/numeric_distributions_{col_name}.png')
    plt.show()
    plt.close()


# In[14]:


# Categorical features - top categories
from matplotlib.pyplot import axes


categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if 'label' in categorical_cols:
    categorical_cols.remove('label')

for i, col in enumerate(categorical_cols):
    labeled_barplot(df, col, perc=True, n=10)
    #plt.tick_params(axis='x', rotation=25)
    col_name = col.replace(' ', '_')
    plt.tight_layout()
    plt.savefig(f'{output_dirs[0]}/categorical_top10_{col_name}.png')
    plt.show()
    plt.close()


# ### Bivariate Analysis: Numeric vs Target (Label)
# 

# In[15]:


# Bivariate analysis: numeric vs target

# function to plot distributions wrt target
def distribution_plot_wrt_target(data, predictor, target):

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    target_uniq = data[target].unique()

    axs[0, 0].set_title("Distribution of target for target=" + str(target_uniq[0]), fontsize=14, fontweight='bold')
    sns.histplot(
        data=data[data[target] == target_uniq[0]],
        x=predictor,
        kde=True,
        ax=axs[0, 0],
        color="teal",
        stat="density",
    )

    axs[0, 1].set_title("Distribution of target for target=" + str(target_uniq[1]), fontsize=14, fontweight='bold')
    sns.histplot(
        data=data[data[target] == target_uniq[1]],
        x=predictor,
        kde=True,
        ax=axs[0, 1],
        color="orange",
        stat="density",
    )

    axs[1, 0].set_title("Boxplot w.r.t target", fontsize=14, fontweight='bold')
    sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 0], palette="gist_rainbow")
    axs[1, 0].set_xlabel('Income', fontsize=14, fontweight='bold')


    axs[1, 1].set_title("Boxplot (without outliers) w.r.t target", fontsize=14, fontweight='bold')
    sns.boxplot(
        data=data,
        x=target,
        y=predictor,
        ax=axs[1, 1],
        showfliers=False,
        palette="gist_rainbow",
    )
    plt.suptitle(f'{predictor} distribution by {target}', fontsize=16, fontweight='bold')
    col_name = predictor.replace(' ', '_')
    axs[1, 0].set_xlabel('Income', fontsize=14, fontweight='bold')
    plt.savefig(f'{output_dirs[0]}/numeric_vs_target_{col_name}.png')

    plt.tight_layout()
    # plt.show()
    return fig

def stacked_barplot(data, predictor, target, figsize=(10, 6), num = 10):
    """
    Print the category counts and plot a stacked bar chart

    data: dataframe
    predictor: independent variable
    target: target variable
    figsize: size of figure (default (10,6))
    palette: color palette for the bars (default 'Set2')
    num: number of top categories to display (default 10)
    """
    fig = plt.figure(figsize=figsize)
    count = data[predictor].nunique()
    col_name = predictor.replace(' ', '_')
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    print(tab1)
    print("-" * 120)
    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )
    fig=tab.plot(kind="bar", stacked=True, ax=fig.gca())
    plt.xlabel(predictor, fontsize=14, fontweight='bold')
    plt.ylabel('Income Proportion', fontsize=14, fontweight='bold')
    #plt.xticks(rotation=25)
    plt.title(f'Income proportion by {col_name}', fontsize=16, fontweight='bold')
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(f'{output_dirs[0]}/cat_vs_target_{col_name}.png')
    return fig


target = 'label'
for i, col in enumerate(numeric_cols):
    #sns.boxplot(x='income', y=col, data=df, ax=axes[i])
    fig = distribution_plot_wrt_target(df, col, target)
    plt.show()
    plt.close()




# In[20]:


age_fig = sns.catplot(x='label', y='age', hue='sex', col='race', col_wrap=3, data=df, kind='bar', palette='Set2', errorbar=None)
age_fig.fig.suptitle('Age distribution by Income and Race', y=1.02, fontsize=14, fontweight='bold')
age_fig.savefig(f'{output_dirs[0]}/age_by_income_race.png')



# In[21]:


age_sex_fig = sns.catplot(x='label', y='age', hue='sex', data=df, kind='bar', palette='Set2', errorbar=None)
age_sex_fig.fig.suptitle('Age distribution by Income and Sex', y=1.02, fontsize=14, fontweight='bold')
age_sex_fig.savefig(f'{output_dirs[0]}/age_by_income_sex.png')



# In[22]:


wage_fig = sns.catplot(x='label', y='wage per hour', col='citizenship', hue='sex', col_wrap=3, data=df, kind='bar', palette='Set2', errorbar=None)
wage_fig.fig.suptitle('Wage distribution by Income and Sex', y=1.02, fontsize=14, fontweight='bold')
wage_fig.savefig(f'{output_dirs[0]}/wage_by_income_sex.png')


# In[23]:


wage_race_fig = sns.catplot(x='label', y='wage per hour', hue='sex',col='race', data=df, col_wrap=3,kind='bar', palette='Set2', errorbar=None)
wage_race_fig.fig.suptitle('Wage distribution by Income and Race', y=1.02, fontsize=14, fontweight='bold')
wage_race_fig.savefig(f'{output_dirs[0]}/wage_by_income_race.png')


# In[24]:


cap_gains_fig = sns.catplot(x='label', y='capital gains', hue='sex', data=df, kind='bar', col='race', col_wrap=3, palette='Set2', errorbar=None)
cap_gains_fig.fig.suptitle('Capital Gains distribution by Income and Race', y=1.02, fontsize=14, fontweight='bold')
cap_gains_fig.savefig(f'{output_dirs[0]}/capital_gains_by_income_race.png')



# In[25]:


# Bivariate: categorical vs target (stacked bar)
for col in categorical_cols[:10]:
    #fig = plt.figure(figsize=(10, 6))
    fig=stacked_barplot(df, col, 'label', figsize=(9,6), num=10)
    plt.show()
    plt.close()

# Correlation matrix
ncorr = len(numeric_cols) + 1  # +1 for the target variable 'income'
corr = df[numeric_cols + ['income']].corr()
plt.figure(figsize=(ncorr,ncorr))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix', fontsize=18, fontweight='bold')
plt.xlabel('Features', fontsize=14, fontweight='bold')
plt.ylabel('Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dirs[0]}/correlation_matrix.png')
plt.show()
plt.close()


print(f"EDA completed. Plots saved in '{output_dirs[0]}/'.")


# In[ ]:





# ### Preprocessing

# In[26]:


# =============================================================================
# Preprocessing
# =============================================================================
X = df.drop(['label', 'income', 'weight'], axis=1, errors='ignore')
y = df['income']

# Update column lists after potential drops
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Preprocessing pipelines
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# =============================================================================
# Split Dataset into Train/Test Split
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")


# ### Model Training and Hyperparameter Tuning

# In[27]:


# =============================================================================
# Model Training & Hyperparameter Tuning
# =============================================================================
def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"\n{name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    return {'name': name, 'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc}

# defining a function to compute different metrics to check performance of a classification model built using sklearn
def model_performance_classification_sklearn(model, predictors, target):
    """
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    """

    # predicting using the independent variables
    pred = model.predict(predictors)

    acc = accuracy_score(target, pred)  # to compute Accuracy
    recall = recall_score(target, pred)  # to compute Recall
    precision = precision_score(target, pred)  # to compute Precision
    f1 = f1_score(target, pred)  # to compute F1-score

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {"Accuracy": acc, "Recall": recall, "Precision": precision, "F1": f1,},
        index=[0],
    )

    return df_perf

def confusion_matrix_sklearn(model, predictors, target):
    """
    To plot the confusion_matrix with percentages

    model: classifier
    predictors: independent variables
    target: dependent variable
    """
    y_pred = model.predict(predictors)
    cm = confusion_matrix(target, y_pred)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix with Percentages")
    plt.tight_layout()
    # plt.show()
    return cm

# Models to compare
models = []
results = []

# Logistic Regression
lr_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
])
param_lr = {'classifier__C': [0.01, 0.1, 1, 10]}
grid_lr = GridSearchCV(lr_pipeline, param_lr, cv=5, scoring='roc_auc', n_jobs=-1)
grid_lr.fit(X_train, y_train)
models.append(('Logistic Regression', grid_lr.best_estimator_))
res = evaluate_model(grid_lr.best_estimator_, X_test, y_test, 'Logistic Regression')
results.append(res)

# Random Forest
rf_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])
param_rf = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5]
}
grid_rf = GridSearchCV(rf_pipeline, param_rf, cv=5, scoring='roc_auc', n_jobs=-1)
grid_rf.fit(X_train, y_train)
models.append(('Random Forest', grid_rf.best_estimator_))
res = evaluate_model(grid_rf.best_estimator_, X_test, y_test, 'Random Forest')
results.append(res)

# XGBoost
xgb_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False))
])
param_xgb = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 6, 9],
    'classifier__learning_rate': [0.01, 0.1],
    'classifier__subsample': [0.8, 1.0]
}
grid_xgb = GridSearchCV(xgb_pipeline, param_xgb, cv=5, scoring='roc_auc', n_jobs=-1)
grid_xgb.fit(X_train, y_train)
models.append(('XGBoost', grid_xgb.best_estimator_))
res = evaluate_model(grid_xgb.best_estimator_, X_test, y_test, 'XGBoost')
results.append(res)

# Save results table
results_df = pd.DataFrame(results)
results_df.to_csv(f'{output_dirs[0]}/model_comparison.csv', index=False)
print(f"\nModel comparison saved to '{output_dirs[0]}/model_comparison.csv'")


# #### Confusion Matrix of all Models

# In[28]:


# =============================================================================
# Confusion matrices for all models
# ==============================================================================
for name, model in models:
    cm = confusion_matrix_sklearn(model, X_test, y_test)
    plt.title(f'{name} Confusion Matrix')
    plt.savefig(f'{output_dirs[0]}/{name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.show()
    plt.close()



# ### ROC-AUC curves of the models

# In[29]:


# =============================================================================
# ROC Curves of the Models
# =============================================================================
best_model_dict = {}
plt.figure(figsize=(8,6))
for name, model in models:
    y_proba = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    best_auc = best_model_dict.get('AUC', auc )
    if auc >= best_auc:
        best_model_dict['AUC'] = auc
        best_model_dict['model'] = model
        best_model_dict['name'] = name
    sns.lineplot(x=fpr, y=tpr, label=f'{name} (AUC={auc:.3f})')
plt.plot([0,1],[0,1], 'k--')
plt.xlabel('False Positive Rate (FPR)', fontsize=14, fontweight='bold')
plt.ylabel('True Positive Rate (TPR)', fontsize=14, fontweight='bold')
plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_dirs[0]}/roc_curves.png')
plt.show()
plt.close()


# ### Feature of Importance of the Best Model
# 

# In[30]:


# =============================================================================
# Feature Importance (XGBoost) – seaborn barplot
# =============================================================================

best_model = best_model_dict['model']
preprocessor_fitted = best_model.named_steps['preprocessor']
cat_feature_names = preprocessor_fitted.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
all_feature_names = np.concatenate([numeric_cols, cat_feature_names])
importances = best_model.named_steps['classifier'].feature_importances_
feat_imp = pd.Series(importances, index=all_feature_names).sort_values(ascending=False).head(10)

plt.figure(figsize=(9,7))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
plt.xlabel('Importance', fontsize=14, fontweight='bold')
plt.ylabel('Feature', fontsize=14, fontweight='bold')
plt.title('Top 10 Feature Importances (XGBoost)', fontsize=16, fontweight='bold')
plt.xticks(rotation=75)
plt.tight_layout()
plt.savefig(f'{output_dirs[0]}/feature_importance_top10.png')
plt.show()
plt.close()

# =============================================================================
# Feature Importance Table (as image) – using matplotlib table
# =============================================================================
def save_feature_importance_table(feat_imp_series, filename):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    ax.axis('tight')
    data = feat_imp_series.reset_index()
    data.columns = ['Feature', 'Importance']
    data['Importance'] = data['Importance']
    cell_text = [[f"{x:.2f}" if isinstance(x, float) else x for x in row] for row in data.values]
    col_widths = [0.7, 0.3]
    table = ax.table(cellText=cell_text, colLabels=data.columns, colWidths=col_widths, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.5, 1.5)
    plt.title('Feature Importance (XGBoost)', fontsize=16, fontweight='bold')
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close()

save_feature_importance_table(feat_imp.head(10), f'{output_dirs[0]}/feature_importance_table.png')


# ### Model Architecture

# In[31]:


# =============================================================================
# Model Architecture Diagram (custom matplotlib)
# =============================================================================
def draw_architecture_diagram():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    box_style = dict( facecolor='lightblue', edgecolor='black')

    # Logistic Regression
    ax = axes[0]
    ax.text(0.5, 0.9, 'Logistic Regression', ha='center', fontsize=12, fontweight='bold')
    ax.add_patch(plt.Rectangle((0.2, 0.5), 0.6, 0.2, **box_style))
    ax.text(0.5, 0.6, 'Input Features', ha='center', va='center')
    ax.add_patch(plt.Rectangle((0.2, 0.2), 0.6, 0.2, **box_style))
    ax.text(0.5, 0.3, 'Linear Combination\n+ Sigmoid', ha='center', va='center')
    ax.arrow(0.5, 0.55, 0, -0.1, head_width=0.05, head_length=0.05, fc='black', ec='black')
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')

    # Random Forest
    ax = axes[1]
    ax.text(0.5, 0.9, 'Random Forest', ha='center', fontsize=12, fontweight='bold')
    ax.add_patch(plt.Rectangle((0.2, 0.6), 0.6, 0.2, **box_style))
    ax.text(0.5, 0.7, 'Bootstrap Sample 1', ha='center', va='center')
    ax.add_patch(plt.Rectangle((0.2, 0.3), 0.6, 0.2, **box_style))
    ax.text(0.5, 0.4, 'Bootstrap Sample 2', ha='center', va='center')
    ax.add_patch(plt.Rectangle((0.2, 0.0), 0.6, 0.2, **box_style))
    ax.text(0.5, 0.1, 'Bootstrap Sample n', ha='center', va='center')
    ax.arrow(0.5, 0.65, 0, -0.1, head_width=0.05, head_length=0.05, fc='black', ec='black')
    ax.arrow(0.5, 0.35, 0, -0.1, head_width=0.05, head_length=0.05, fc='black', ec='black')
    ax.text(0.5, -0.15, 'Majority Voting', ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightgreen'))
    ax.set_xlim(0,1); ax.set_ylim(-0.2,1); ax.axis('off')

    # XGBoost
    ax = axes[2]
    ax.text(0.5, 0.9, 'XGBoost', ha='center', fontsize=12, fontweight='bold')
    ax.add_patch(plt.Rectangle((0.2, 0.6), 0.6, 0.2, **box_style))
    ax.text(0.5, 0.7, 'Tree 1 (Residuals)', ha='center', va='center')
    ax.add_patch(plt.Rectangle((0.2, 0.3), 0.6, 0.2, **box_style))
    ax.text(0.5, 0.4, 'Tree 2 (Residuals)', ha='center', va='center')
    ax.add_patch(plt.Rectangle((0.2, 0.0), 0.6, 0.2, **box_style))
    ax.text(0.5, 0.1, 'Tree n (Residuals)', ha='center', va='center')
    ax.arrow(0.5, 0.65, 0, -0.1, head_width=0.05, head_length=0.05, fc='black', ec='black')
    ax.arrow(0.5, 0.35, 0, -0.1, head_width=0.05, head_length=0.05, fc='black', ec='black')
    ax.text(0.5, -0.15, 'Weighted Sum', ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightgreen'))
    ax.set_xlim(0,1); ax.set_ylim(-0.2,1); ax.axis('off')

    plt.suptitle('Model Architecture Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots_classification/model_architectures.png')
    plt.show()
    plt.close()

draw_architecture_diagram()

# =============================================================================
# Save Best Model
# =============================================================================
joblib.dump(best_model, f'{output_dirs[1]}/best_classifier.pkl')
print(f"\nClassification completed. All plots saved in '{output_dirs[0]}/'.")
print(f"Best model saved to '{output_dirs[1]}/best_classifier.pkl'.")


# In[ ]:





# In[ ]:




