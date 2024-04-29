import pandas as pd
from scipy import stats
from  scripts.constant import col_high_values
from  scripts.constant import model_columns
from math import log
from math import exp

def fill_null_values(data: pd.DataFrame ):
    data['Solids'] = data['Solids'].fillna(data['Solids'].median())
    data = data.fillna(data.mean())
    return data

def create_benchmark(data : pd.DataFrame, target_var : str = 'Potability' ):
    dict_var = {}
    for column in data.columns:
        if column != target_var:
            correlation, p_value = stats.pointbiserialr(x=data[column], y=data[target_var] )
            dict_var[column] = (correlation)
    df_benchmark = pd.DataFrame.from_dict(dict_var, orient='index', columns=['Correlation_original'])
    return df_benchmark



def compute_tranformations_df (data: pd.DataFrame , col_high_value:str = col_high_values , max_polinomial_degree :int = 10 ):

    df_trans = data.copy()
    col_selected = [x for x in data.columns if x !='Potability']
    # quadratic and cubic
    trans_type = "_power_"
    # deciding the degree of power in this loop
    for power in range(2,max_polinomial_degree):
        for var in col_selected:
            df_trans[var+trans_type+str(power)] = df_trans[var].apply(lambda x: x ** power)
    # squared root
    trans_type = "_sq_root"       
    for var in col_selected:
        df_trans[var+trans_type] = df_trans[var].apply(lambda x: x ** 0,5)
        
    # logatirmic
    trans_type = "_log"        
    for var in col_selected:
        df_trans[var+trans_type] = df_trans[var].apply(lambda x: log(x) if x > 0 else None)
        df_trans[var+trans_type].fillna(df_trans[var+trans_type].mean())

    #subtracting the mean for each variable
    trans_type = "centered"
    for var in col_selected:
        df_trans[var+trans_type] = df_trans[var].apply(lambda x: x - data[var].mean() )


    #exponential 
    trans_type = "_exp"
    for var in col_selected:
        # transforming for a constant to avoid values too big to compute
        if var in col_high_value:
            df_trans[var] = df_trans[var].apply(lambda x: x / 100)

        df_trans[var+trans_type] = df_trans[var].apply(lambda x: exp(x) if abs(x) < 500 else None)
        df_trans[var+trans_type].fillna(df_trans[var+trans_type].mean())

    df_trans= df_trans.fillna(df_trans.mean())
    return df_trans



def compute_tranformations_correlations (df_trans: pd.DataFrame , original_columns :list ,target_var :pd.Series ):
    # Calculate point-biserial correlation for each column, ph_log has been excluded for computational problems
    dict_trans = {}
    for column in df_trans.columns:
        if column not in original_columns:
            # calculating the correlation with the y:
            correlation, p_value = stats.pointbiserialr(x=df_trans[column], y=target_var )
            dict_trans[column] = correlation
    df_compare = pd.DataFrame.from_dict(dict_trans, orient='index', columns=['Correlation_transformed'])
    df_compare['original_column'] = df_compare.apply(lambda row: [x for x in original_columns if row.name.startswith(x)], axis=1)
    df_compare['original_column'] = df_compare.apply(lambda row: row['original_column'][0], axis=1)
    df_compare = df_compare.fillna(0)
    return df_compare


def selecting_transfomed_features_columns (df_benchmark : pd.DataFrame, df_compare : pd.DataFrame, forecast_data : bool = False , train_columns : list = model_columns ):
    # creating a df to compare the correlations
    df_comparison = pd.merge(df_benchmark, df_compare, left_index = True, right_on ='original_column', how='left')
    df_comparison = df_comparison.drop(columns=['original_column'])
    # calculating differences in between the correlations
    df_comparison['Performance_abs'] = abs( abs(df_comparison['Correlation_transformed'] - df_comparison['Correlation_original']) )
    # calculating the odds against the correlation of the original variable
    df_comparison['Performance_odd'] = abs(df_comparison['Correlation_transformed'])/ abs(df_comparison['Correlation_original']) 

    if forecast_data:
        col_to_add= train_columns
    else:
        df_comparison_sel = df_comparison[df_comparison['Performance_odd']>1.00001]
        df_comparison_sel = df_comparison_sel.sort_values(by='Performance_odd', ascending=False)
        col_to_add_= df_comparison_sel.index
        col_to_add = list(col_to_add_) +['Hardness_power_2','Hardness_power_3']
    return col_to_add

def merge_selected_features(data: pd.DataFrame,df_trans: pd.DataFrame, col_to_add: list):
    col_to_add= [x for x in col_to_add if x not in data.columns]
    df_trans = df_trans[col_to_add].copy()
    df_model = pd.merge(data, df_trans, left_index = True, right_index =True, how='left')

    null_val = df_model.isnull().any().sum() 
    if null_val > 0:
        print( f"filling {null_val} null values in the final dataset")
        df_model = df_model.fillna(df_model.mean())
    return df_model


def preprocess_data(data: pd.DataFrame, target_var : str = 'Potability' ,forecast_data : bool = False ):

    data= fill_null_values(data)
    df_benchmark = create_benchmark(data)
    df_trans = compute_tranformations_df (data )
    target_var_series = data[target_var] 
    # creating a df to compare the correlations
    df_compare= compute_tranformations_correlations (df_trans, data.columns,  target_var_series  )
    col_to_add= selecting_transfomed_features_columns (df_benchmark , df_compare , forecast_data )
    model_df = merge_selected_features(data,df_trans, col_to_add)
    return model_df
    