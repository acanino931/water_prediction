from scripts.Import_data import  import_train_data
from scripts.preprocessing_functions import  preprocess_data
import pandas as pd
import warnings
import scripts.modeling as mod
warnings.filterwarnings("ignore", category=FutureWarning)





def load_model(data:pd.DataFrame , target_var : str = 'Potability' ):

    data = import_train_data()
    model_df= preprocess_data(data, target_var = 'Potability',forecast_data = False )
    model= mod.train_the_model(model_df, target_var)

    return model,model_df








