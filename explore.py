import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math as math


# show rows and columns of a data frame

def show_shape(df):
    print('{}{}'.format("rows :", df.shape[0]))
    print('{}{}'.format("columns :", df.shape[1]))


    
    
# get list of numeric column names

def numeric_columns(df):
    return df.select_dtypes(np.number).columns.tolist()

  
    
    
# get list of categorical column names

def categorical_columns(df):
    return df.select_dtypes(['object']).columns.tolist()



# custom function to include unique values of categorical variables in the describe() funtion

def describe_categorical_variables(dataframe):
    
    # generate stat for categorical variable and clone the row index
    cat_columns = categorical_columns(dataframe)
    df_cat_stat = dataframe[cat_columns].describe().transpose()
    df_cat_stat['variable'] = df_cat_stat.axes[0].tolist() # common key to join join
    
    # generate list of unique values for each categorical variable
    row_list = []
    for i in cat_columns:
        lst_unique  = dataframe[i].unique().tolist()
        row = str().strip('[]')
        # convert all elements in the list to string of list as sometime elements can be NaN
        s = [str(j) for j in lst_unique]       
        row  = [i, ','.join(s)]
        row_list.append(row)
    cat_unique_values_df = pd.DataFrame(row_list,columns =['variable','unique values'])
    # join stat output df and unique values df
    df_final = pd.merge(cat_unique_values_df,df_cat_stat, on='variable')
    return df_final


# get all null columns in a data frame
def null_columns(dataframe):
   
    null_s = dataframe.isnull().sum().sort_values(ascending=False)
    dtypes_s = dataframe.dtypes
    df_null= pd.concat([null_s,dtypes_s],axis=1)
    df_null = df_null.rename(columns={0:'count',1:'data_type'})
    return df_null[df_null['count']>0]



# get pairs of variables with correlation > +/- threshold value
# remove diagonal elements

def corr(dataframe,threshold=0.0,target=None):
    
    """
    usage :

    get all correlation pairs
        explore.corr(df) 

    get correlation pairs with corr >=threshold
        explore.corr(df,threshold=0.7) 

    get correlation with a selected variable and corr >=threshold
        explore.corr(df,target='SalePrice',threshold=0.5) 
    
    """
    df_corr = dataframe.corr()
    if target is None:
        df_corr_ = pd.DataFrame([(i,j,df_corr.loc[i,j]) for i in df_corr.index for j in df_corr.columns if abs(df_corr.loc[i,j])>=threshold and i!=j], columns=['var1','var2', 'correlation']).drop_duplicates().sort_values(by=['correlation'],ascending=False)
        # remove duplicates by selecting even records only
        df_corr_ = df_corr_.iloc[::2] # even
    elif target is not None and dataframe[target].dtype != 'object': 
        df_corr_ = pd.DataFrame([(i,df_corr.loc[i,target]) for i in df_corr.index if abs(df_corr.loc[i,target])>=threshold and i!=target], columns=['var', 'correlation']).drop_duplicates().sort_values(by=['correlation'],ascending=False)
    else: 
        raise Exception('target should be a numeric variable') 
     # remove duplicates by selecting even records only
    return df_corr_  # even






def show_scatterplot(df,corr_pairs=None, x=None, y=None):

    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    for variable, subplot in zip(corr_pairs.index.tolist(), ax.flatten()):
        sns.scatterplot(x=df[corr_pairs.loc[variable,][0]], y=df[corr_pairs.loc[variable,][1]], data=df,hue='Utilities', ax=subplot)
        sns.regplot(x=corr_pairs.loc[variable,][0], y=corr_pairs.loc[variable,][1], data=df,scatter=False,color='m' ,ax=subplot)
    

def show_corrplot(dataframe):
    
    corr = dataframe.corr()
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
    

    
def show_histogram(dataframe,columns=None,bins=15,figsize=(20,15)):
    
    if columns is None:
        numeric_cols = numeric_columns(dataframe)
        dataframe[numeric_cols].hist(bins=bins, figsize=figsize, layout=(round(math.sqrt(len(numeric_cols))),math.ceil(math.sqrt(len(numeric_cols)))));
    else:
        numeric_cols = columns
        dataframe[numeric_cols].hist(bins=bins, figsize=figsize, layout=(round(math.sqrt(len(numeric_cols))),math.ceil(math.sqrt(len(numeric_cols)))));

