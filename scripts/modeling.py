from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd

# splitting the data
def train_the_model(df_model: pd.DataFrame, target_var):
    X = df_model.drop(columns=[target_var])  # Features
    y = df_model[target_var]  # Target variable


    # Initialize the Extra Trees Classifier
    extra_trees_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)

    extra_trees_clf.fit(X, y)


    return extra_trees_clf