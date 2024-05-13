# %%
import copy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
# %matplotlib inline
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# %% [markdown]
# 0.85 needed

# %% [markdown]
# # Reading Datasets

# %%
df_train = pd.read_csv("/kaggle/input/fraud-detection/train.csv")
test_df = pd.read_csv("/kaggle/input/fraud-detection/test.csv")

# %%
Y = df_train["isFraud"]
df_train.drop(labels=["isFraud"], axis = 1, inplace=True) #We drop transaction ID because it is unique for each and every row and doesn't provides any essential information.

# %%
train_df = pd.concat([df_train.assign(ind="train"), test_df.assign(ind="test")])

# %%
train_df.drop(labels=["TransactionID"], axis = 1, inplace=True) #We drop transaction ID because it is unique for each and every row and doesn't provides any essential information.
train_df.head()

# %%
import datetime
START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
train_df['DT_M'] = train_df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
train_df['DT_M'] = (train_df['DT_M'].dt.year-2017)*12 + train_df['DT_M'].dt.month 

# %%
train_df['DT_M']

# %% [markdown]
# # Dropping Columns

# %%
Empty_col_train = []
SameValue_col_train = []
MaxSameValue_col_train = []

for f in train_df.columns :
    if train_df[f].isna().sum()/len(train_df) >= 0.9 :
        Empty_col_train.append(f)
    if train_df[f].nunique() <= 1 :
        SameValue_col_train.append(f)
    if train_df[f].value_counts(dropna=False, normalize=True).values[0] >= 0.9 :
        MaxSameValue_col_train.append(f)



# %%
cols_tobedropped = list(set(Empty_col_train+SameValue_col_train+MaxSameValue_col_train))
# cols_tobedropped = Empty_col_train

train_df.drop(cols_tobedropped, axis=1, inplace=True)
test_df.drop(cols_tobedropped, axis=1, inplace=True)

# %% [markdown]
# ### Categorical and numerical features sorting
# 

# %%
def cat_num_features(df): 
    cat_cols = []
    numer_cols = []
    
    # Given Categorical Features 
    cat_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2','M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9','DeviceType', 'DeviceInfo']
    cat_cols+=["id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29", "id_30", "id_31", "id_33", "id_34", "id_35", "id_36", "id_37", "id_38"]


    # Updating the Categorical Feature Names List based on the columns present in the dataframe
    cat_cols = [feature for feature in cat_cols if feature in df.columns.values]
    numer_cols = [feature for feature in df.columns if feature not in cat_cols]
    
    return (cat_cols, numer_cols) 

# %%
cat_cols, numer_cols = cat_num_features(train_df)
categorical_feature_indices = [train_df.columns.get_loc(f) for f in cat_cols]

# %%
numer_cols.remove("ind")

# %% [markdown]
# ### Removing Inter-Dependent Features
# 
# We have group of attributes for id(12 -38), C(1-15), D(1-16) and V(1-339). Some of the attributes in their groups are inter-dependent and thus we can remove some of the attributes from that particular group. To do this, we will make use of VIF values (Variance Inflation Factor).
# A variance inflation factor (VIF) is a measure of the amount of multicollinearity in a set of multiple regression variables. 

# %% [markdown]
# # C Features

# %%
c_features = ["C"+str(i) for i in range(1,15) if "C"+str(i) in train_df.columns]

# %%
# Create correlation matrix
corr_matrix = train_df[c_features].corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] >= 0.9)]

# Drop features 
train_df.drop(to_drop, axis=1, inplace=True)
for i in to_drop:
    numer_cols.remove(i)


# %% [markdown]
# # V-features

# %%
v_features = ["V"+str(i) for i in range(1,340) if "V"+str(i) in train_df.columns]


# %%
# we will take all columns and group them based on missing percentage
nan_dict = {}
for col in v_features:
    count = df_train[col].isnull().sum()
    try:
        nan_dict[count].append(col)
    except:
        nan_dict[count] = [col]
        
for k,v in nan_dict.items():
    print(f'#####' * 4)
    print(f'NAN count = {k} percent: {(int(k)/df_train.shape[0])*100} %')
    print(v)

# %% [markdown]
# # V1-V11

# %%
def reduce_groups(grps):
    '''
    determining column that have more unique values among a group of atttributes
    '''
    use = []
    for col in grps:
        max_unique = 0
        max_index = 0
        for i,c in enumerate(col):
            n = df_train[c].nunique()
            if n > max_unique:
                max_unique = n
                max_index = i
        use.append(col[max_index])
    return use

# %%
def coorelation_analysis(cols,title='Coorelation Analysis',size=(12,12)):
    cols = sorted(cols)
    fig,axes = plt.subplots(1,1,figsize=size)
    df_corr = train_df[cols].corr()
    sns.heatmap(df_corr,annot=True,cmap='RdBu_r')
    axes.title.set_text(title)
    plt.show()

# %%
g1_cols = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11']
coorelation_analysis(g1_cols)

# %%
pairs = [['V1'], ['V2', 'V3'], ['V4', 'V5'], ['V6', 'V7'], ['V8', 'V9'], ['V10', 'V11']]
g1 = reduce_groups(pairs)
g1

# %% [markdown]
# # V12-V34

# %%
g2_cols = ['V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34']
coorelation_analysis(g2_cols, title='Coorelation Analysis: V12-V34',size=(len(g2_cols), len(g2_cols)))

# %%
pairs = [['V12','V13'],['V14'],['V15','V16','V17','V18','V21','V22','V31','V32','V33','V34'],['V19','V20'],['V23','V24'],['V25','V26'],['V27','V28'],['V29','V30']]

g2 = reduce_groups(pairs)
g2

# %% [markdown]
# # V35-V52

# %%
cols = ['V35', 'V40', 'V41', 'V39', 'V38', 'V51', 'V37', 'V52', 'V36', 'V50', 'V48', 'V42', 'V43', 'V44', 'V46', 'V47', 'V45', 'V49']
coorelation_analysis(cols,title='Coorelation Analysis: V35-V52',size=(12,12))

# %%
pairs = [['V35','V36'],['V37','V38'],['V39','V40','V42','V43','V50','V51','V52'],['V41'],
         ['V44','V45'],['V46','V47'],['V48','V49']]

g3 = reduce_groups(pairs)
g3

# %% [markdown]
# # V53-V74

# %%
g4_cols = ['V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V73', 'V74']
coorelation_analysis(g4_cols,  title='Coorelation Analysis: V53-V74',size=(len(g4_cols), len(g4_cols)))

# %%
pairs = [['V53','V54'],['V55'],['V56'],['V57', 'V58', 'V59', 'V60', 'V63', 'V64', 'V71', 'V72', 'V73', 'V74'],['V61','V62'], ['V65'],['V66','V67'],['V68'],['V69','V70']]

g4 = reduce_groups(pairs)
g4

# %% [markdown]
# # V75 - V94

# %%
g5_cols = ['V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92', 'V93', 'V94']
coorelation_analysis(g5_cols, size=(len(g5_cols), len(g5_cols)))

# %%
pairs = [['V75','V76'],['V77','V78'],['V79','V80','V81','V84','V85','V92','V93','V94'],['V82','V83'],['V86','V87'], ['V88'],['V89'],['V90','V91']]

g5 = reduce_groups(pairs)
g5

# %% [markdown]
# # V95-V137

# %%
g6_cols = ['V95', 'V96', 'V97', 'V99', 'V100', 'V126', 'V127', 'V128', 'V130', 'V131']
coorelation_analysis(g6_cols)

# %%
pairs = [['V100', 'V99'], ['V126', 'V127', 'V128', 'V95', 'V96', 'V97'], ['V130', 'V131']]
g6 = reduce_groups(pairs)
g6

# %% [markdown]
# # V138 - V163

# %%
cols = ['V138', 'V139', 'V140', 'V141', 'V142', 'V146', 'V147', 'V148', 'V149', 'V153', 'V154', 'V155', 'V156', 'V157', 'V158', 'V161', 'V162', 'V163']
coorelation_analysis(cols, size=(len(cols), len(cols)))

# %%
pairs = [['V138'],['V139','V140'],['V141','V142'],['V146','V147'],['V148','V149','V153','V154','V155', 'V156','V157','V158'],['V161','V162','V163']]

g13 = reduce_groups(pairs)
g13

# %% [markdown]
# # V143- V166

# %%
cols = ['V143', 'V144', 'V145', 'V150', 'V151', 'V152', 'V159', 'V160', 'V164', 'V165', 'V166']
coorelation_analysis(cols)

# %%
pairs = [['V143','V164','V165'],['V144','V145','V150','V151','V152','V159','V160'],['V166']]

g14 = reduce_groups(pairs)
g14

# %% [markdown]
# # V167 - V216

# %%
cols = ['V167', 'V168', 'V172', 'V173', 'V176', 'V177', 'V178', 'V179', 'V181', 'V182', 'V183'] #['V186', 'V187', 'V190', 'V191', 'V192', 'V193', 'V196', 'V199', 'V202', 'V203', 'V204', 'V205', 'V206', 'V207', 'V211', 'V212', 'V213', 'V214', 'V215', 'V216']
coorelation_analysis(cols,title='Coorelation Analysis: V167~V216',size=(20,20))

# %%
pairs = [['V167','V168','V177','V178','V179'],['V172','V176'],['V173'],['V181','V182','V183']]
temp = reduce_groups(pairs)
temp

# %%
cols = ['V186','V187','V190','V191','V192','V193','V196','V199','V202','V203','V204','V211','V212','V213','V205','V206','V207','V214','V215','V216']

coorelation_analysis(cols,title='Coorelation Analysis',size=(20,20))

# %%
pairs = [['V186','V187','V190','V191','V192','V193','V196','V199'],['V202','V203','V204','V211','V212','V213'],['V205','V206'],['V207'],['V214','V215','V216']]
temp1 = reduce_groups(pairs)
temp1

# %%
g7 = temp+temp1
g7

# %% [markdown]
# # V169 - V210

# %%
g8_cols = ['V169', 'V170', 'V171', 'V174', 'V175', 'V180', 'V184', 'V185', 'V188', 'V189', 'V194', 'V195', 'V197', 'V198', 'V200', 'V201', 'V208', 'V209', 'V210']
coorelation_analysis(g8_cols, size = (len(g8_cols), len(g8_cols)))

# %%
pairs = [['V169'],['V170','V171','V200','V201'],['V174','V175'],['V180'],['V184','V185'],['V188','V189'],['V194','V195','V197','V198'],['V208','V210'], ['V209']]
g8 = reduce_groups(pairs)
g8

# %% [markdown]
# # V217 - V278

# %%
# cols = ['V217', 'V218', 'V219', 'V223', 'V224', 'V225', 'V226', 'V228', 'V229', 'V230', 'V231', 'V232', 'V233', 'V235', 'V236', 'V237', 'V240', 'V241', 'V242', 'V243', 'V244', 'V246', 'V247', 'V248', 'V249', 'V252', 'V253', 'V254', 'V257', 'V258', 'V260', 'V261', 'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V269', 'V273', 'V274', 'V275', 'V276', 'V277', 'V278']
cols = ['V217','V218','V219','V231','V232','V233','V236','V237','V223','V224','V225','V226','V228','V229','V230','V235']
coorelation_analysis(cols, size = (len(cols), len(cols)))

# %%
pairs = [['V217','V218','V219','V231','V232','V233','V236','V237'],['V223'],['V224','V225'],['V226'],['V228'],['V229','V230'],['V235']]
temp = reduce_groups(pairs)
temp

# %%
cols = ['V240','V241','V242','V243','V244','V258','V246','V257','V247','V248','V249','V253','V254','V252','V260','V261','V262']
coorelation_analysis(cols,title='Coorelation Analysis',size=(20,20))

# %%
pairs = [['V240','V241'],['V242','V243','V244','V258'],['V246','V257'],['V247','V248','V249','V253','V254'],['V252'],['V260'],['V261','V262']]
temp1 = reduce_groups(pairs)
temp1

# %%
cols =  ['V263','V265','V264','V266','V269','V267','V268','V273','V274','V275','V276','V277','V278']
coorelation_analysis(cols,title='Coorelation Analysis',size=(20,20))

# %%
pairs =[['V263','V265','V264'],['V266','V269'],['V267','V268'],['V273','V274','V275'],['V276','V277','V278']]
temp2 = reduce_groups(pairs)
temp2

# %%
g9 = temp+temp1+temp2

# %% [markdown]
# # V220 - V272

# %%
g10_cols = ['V220', 'V221', 'V222', 'V227', 'V234', 'V238', 'V239', 'V245', 'V250', 'V251', 'V255', 'V256', 'V259', 'V270', 'V271', 'V272']
coorelation_analysis(g10_cols, size = (len(g10_cols), len(g10_cols)))

# %%
pairs = [['V220'],['V221','V222','V227','V245','V255','V256','V259'],['V234'],['V238','V239'],['V250','V251'],['V270','V271','V272']]
g10 = reduce_groups(pairs)
g10

# %% [markdown]
# # G11

# %%
g11_cols = ['V279', 'V280', 'V285', 'V287', 'V291', 'V292', 'V294', 'V302', 'V303', 'V304', 'V306', 'V307', 'V308', 'V310', 'V312', 'V317']
coorelation_analysis(g10_cols, size = (len(g10_cols), len(g10_cols)))

# %%
pairs = [['V279', 'V280', 'V294', 'V306', 'V307', 'V308', 'V317'], ['V285', 'V287'], ['V291', 'V292'], ['V302', 'V303', 'V304'], ['V310', 'V312']]
g11 = reduce_groups(pairs)
g11

# %% [markdown]
# # G12

# %%
g12_cols = ['V282', 'V283', 'V288', 'V289', 'V313', 'V314', 'V315']
coorelation_analysis(g11_cols)

# %%
pairs = [['V282', 'V283'], ['V288', 'V289'], ['V313', 'V314', 'V315']]
g12 = reduce_groups(pairs)

# %%
reduced_V_cols = g1+g2+g3+g4+g5+g6+g7+g8+g9+g10+g11+g12+g13+g14
# reduced_V_cols

# %%
V_features = [i for i in train_df.columns if i[0]=='V']
# V_features

# %%
not_V_red = [i for i in V_features if i not in reduced_V_cols]

# %%
train_df.drop(labels = not_V_red, axis = 1, inplace = True)

# %% [markdown]
# ### Dropping C, D Features based on Backward Feature Selection
# 

# %%
train_df.head()

# %% [markdown]
# # D features

# %%
d_features = ["D"+str(i) for i in range(1,15) if "D"+str(i) in train_df.columns]

# %%
# Create correlation matrix
corr_matrix = train_df[d_features].corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] >= 0.9)]

# Drop features 
train_df.drop(to_drop, axis=1, inplace=True)
for i in to_drop:
    numer_cols.remove(i)


# %%
to_drop

# %%
train_df.head()

# %%
for i in not_V_red: numer_cols.remove(i)

# %%
train_df[numer_cols] = train_df[numer_cols].fillna(train_df[numer_cols].median())   # fills the missing values with median
train_df.head()

# %% [markdown]
# ### Extracting Month from the TransactionDT Attribute
# 

# %%
import datetime
strtdate = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
train_df['DTMon'] = train_df['TransactionDT'].apply(lambda x: (strtdate + datetime.timedelta(seconds = x)))
train_df['DTMon'] = (train_df['DTMon'].dt.year-2017)*12 + train_df['DTMon'].dt.month 

# %%
train_df1 = train_df

# %%
for i in numer_cols:
    train_df1[i] = (train_df1[i] - train_df1[i].min())/(train_df1[i].max() - train_df1[i].min())
train_df1.head()

# %%
for i in numer_cols:
    train_df1[i] = (train_df1[i] - train_df1[i].min())/(train_df1[i].max() - train_df1[i].min())
train_df1.head()

# %%
one_hot = []
label_encode = []
for i in cat_cols:
    if(train_df1[i].nunique() <= 5):
        one_hot.append(i)
    else:
        label_encode.append(i)

# %%
train_df1[cat_cols] = train_df1[cat_cols].fillna('Notthere')
train_df1 = pd.get_dummies(train_df1, columns = one_hot)

# %%
for f in label_encode:
    train_df1[f] = train_df1[f].astype(str)
    le = LabelEncoder()
    le.fit(train_df1[f])
    train_df1[f] = le.transform(train_df1[f])
train_df1.head()

# %%
test_df1, train_df1 = train_df1[train_df1["ind"].eq("test")], train_df1[train_df1["ind"].eq("train")]


# %% [markdown]
# # Label Encoding, use one-hot but

# %%
train_df1.drop(labels=["ind"], axis=1, inplace=True)
test_df1.drop(labels=["ind"], axis=1, inplace=True)

# %% [markdown]
# ### Making a Pipeline

# %%
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# %% [markdown]
# use 0.1, 0.5 for over and under

# %%
over = RandomOverSampler(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)
pipeline = Pipeline(steps=[('o', over), ('u', under)])
train_df1, Y = pipeline.fit_resample(train_df1, Y)

# %%
cols = list(train_df1)

# %% [markdown]
# ### Applying GropuKFold based on the new attribute DTMon

# %% [markdown]
# GroupKFold() - K-fold iterator variant with non-overlapping groups 

# %% [markdown]
# split() - Generate indices to split data into training and test set.

# %%
from sklearn.model_selection import GroupKFold
testpred = np.zeros(len(test_df1))

## Only six month data is present in our dataset, thus atmax only 6 splits can be made
kfold = GroupKFold(n_splits=6)

for i, (tr, val) in enumerate( kfold.split(train_df1, Y, groups=train_df1['DT_M']) ):
    month = train_df1.iloc[val]['DT_M'].iloc[0] 
    
    ### Our previous best parameters
    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        n_estimators=2000,
        tree_method='gpu_hist',
        random_state=3,
        subsample=0.8,
        max_depth=12,
        colsample_bytree=0.4,
        learning_rate=0.2
    )      

    model.fit(train_df1[cols].iloc[tr], Y.iloc[tr],eval_set=[(train_df1[cols].iloc[val],Y.iloc[val])],verbose=100, early_stopping_rounds=200)

    testpred += model.predict_proba(test_df1[cols])[:, 1]/kfold.n_splits

# %%
print(testpred)

# %%
testpred2 = np.where(testpred > 0.2, 1, 0)
CSV4 = pd.DataFrame(testpred2)
file = CSV4.to_csv("PredGroupKFold0.2.csv")


