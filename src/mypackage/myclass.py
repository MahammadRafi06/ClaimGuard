import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

class LoadData:
    def __init__(self):
        """
        Initializes the LoadData class with a DataFrame and identifies columns for processing.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing columns to be processed.
        """
        self.df = pd.read_csv('data/inpatient.csv',low_memory=False)
        # Identify date columns
        self.dates = [col for col in self.df.columns if 'DT' in col]
        # Identify categorical code columns for encoding
        self.codes = [col for col in self.df.select_dtypes(include=['object']).columns if 'DGNS' in col or 'PRCDR' in col or 'HCPCS' in col]
        # Define columns to be dropped
        self.impcodes = [
            "BENE_ID", "CLM_ID", "NCH_NEAR_LINE_REC_IDENT_CD", "PRVDR_NUM", 
            "FI_NUM", "PRVDR_STATE_CD", "ORG_NPI_NUM", "AT_PHYSN_UPIN", 
            "AT_PHYSN_NPI", "OP_PHYSN_UPIN", "OP_PHYSN_NPI", "OT_PHYSN_UPIN", 
            "OT_PHYSN_NPI", "CLM_LINE_NUM"
        ]
        # Identify Present on Admission (POA) columns
        self.poas = [col for col in self.df.select_dtypes(include=['object']).columns if 'POA' in col]
        # Define additional columns for manual encoding
        self.implicit_code = [
            'NCH_CLM_TYPE_CD', 'CLAIM_QUERY_CODE', 'CLM_FAC_TYPE_CD', 'CLM_SRVC_CLSFCTN_TYPE_CD',
            'CLM_FREQ_CD', 'CLM_MDCR_NON_PMT_RSN_CD', 'NCH_PRMRY_PYR_CD', 'FI_CLM_ACTN_CD',
            'CLM_MCO_PD_SW', 'PTNT_DSCHRG_STUS_CD', 'CLM_IP_ADMSN_TYPE_CD', 'CLM_SRC_IP_ADMSN_CD',
            'NCH_PTNT_STATUS_IND_CD', 'CLM_DRG_CD', 'CLM_DRG_OUTLIER_STAY_CD', 'REV_CNTR', 
            'REV_CNTR_DDCTBL_COINSRNC_CD', 'CLM_PPS_IND_CD'
        ]

    def clean_data(self):
        self.df["NCH_ACTV_OR_CVRD_LVL_CARE_THRU"].rename("NCH_ACTV_OR_CVRD_LVL_CARE_THRU_DT", inplace=True)
        
        # Drop columns in impcodes
        self.df.drop(columns=self.impcodes, axis=1, inplace=True, errors='ignore')

        """
        Converts the identified date columns to datetime, formats them, 
        and converts them to float values for further processing.
        
        Returns:
        pd.DataFrame: The DataFrame with converted date columns.
        """
        for date in self.dates:
            self.df[date] = pd.to_datetime(self.df[date], errors='coerce')
            self.df[date] = self.df[date].dt.strftime('%Y%m%d%H%M%S')
            self.df[date] = self.df[date].astype(float)

        
        self.df["length_of_stay"] = self.df['NCH_BENE_DSCHRG_DT'] - self.df['CLM_ADMSN_DT']
        self.df["statement_days"] = self.df['CLM_THRU_DT'] - self.df['CLM_FROM_DT']
        self.df.drop(['NCH_BENE_DSCHRG_DT', 'CLM_ADMSN_DT'], axis=1, inplace=True)
        self.df.drop(['CLM_THRU_DT', 'CLM_FROM_DT'], axis=1, inplace=True)

        for code in self.codes:
            #self.df[code] = self.df[code].str[0]
            self.df[code] = self.df[code].astype(str).str[0]
            self.df[code] = self.df[code].apply(lambda x: ord(x) if isinstance(x, str) and len(x) == 1 else np.nan)


        """
        Removes specified columns from the DataFrame.
        
        Returns:
        pd.DataFrame: The DataFrame with specified columns dropped.
        """
        for code in self.poas:
            self.df[code] = self.df[code].str.strip().fillna('U')
            le = LabelEncoder()
            self.df[code] = le.fit_transform(self.df[code])



        """
        Encodes the identified POA columns with LabelEncoder.
        
        Returns:
        pd.DataFrame: The DataFrame with encoded POA columns.
        """
        
        """
        Fills NaN values and encodes specific columns with LabelEncoder.
        
        Returns:
        pd.DataFrame: The DataFrame with manually encoded columns.
        """
        for code in self.implicit_code:
            self.df[code] = self.df[code].fillna(-1)
            le = LabelEncoder()
            self.df[code] = le.fit_transform(self.df[code])


        dx_pr_codes = [cd for cd in self.df.columns if 'DGNS' in cd or 'PRCDR' in cd or 'HCPCS' in cd]
        poa = [pa for pa in self.df.columns if 'POA' in pa]
        implicit_code = ['NCH_CLM_TYPE_CD','CLAIM_QUERY_CODE','CLM_FAC_TYPE_CD','CLM_SRVC_CLSFCTN_TYPE_CD','CLM_FREQ_CD','CLM_MDCR_NON_PMT_RSN_CD',
        'NCH_PRMRY_PYR_CD','FI_CLM_ACTN_CD','CLM_MCO_PD_SW','PTNT_DSCHRG_STUS_CD','CLM_IP_ADMSN_TYPE_CD','CLM_SRC_IP_ADMSN_CD',
        'NCH_PTNT_STATUS_IND_CD','CLM_DRG_CD','CLM_DRG_OUTLIER_STAY_CD','REV_CNTR','REV_CNTR_DDCTBL_COINSRNC_CD','CLM_PPS_IND_CD',
        ]
        columns_to_sum = dx_pr_codes + poa + implicit_code
        self.df['Sum'] = self.df[columns_to_sum].sum(axis=1)
        self.df.drop(columns_to_sum, axis=1, inplace=True)
        return self.df

