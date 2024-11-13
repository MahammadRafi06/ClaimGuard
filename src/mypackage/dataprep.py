import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
class DataPrep:
    def __init__(self, df):
        """
        Initializes the LoadData class with a DataFrame and identifies columns for processing.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing columns to be processed.
        """
        self.df = df
        # Identify date columns

    def prep(self):
       sclar = StandardScaler()
       self.df = sclar.fit_transform(self.df)
       imp = SimpleImputer(strategy='mean')
       self.df = imp.fit_transform(self.df)
       pc = PCA(n_components=.95)
       self.df = pc.fit_transform(self.df)
       return self.df