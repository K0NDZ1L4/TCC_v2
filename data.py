from ucimlrepo import fetch_ucirepo
import pandas as pd


class uciDataset():

    def __init__(self):
        self.importdataset()

    def importdataset(self):
        heart_disease = fetch_ucirepo(id=45)
        features = heart_disease.data.features
        targets = heart_disease.data.targets
        result = pd.concat([features, targets], axis=1)
        result = result.dropna()
        self.features = result[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                                'exang', 'oldpeak', 'slope', 'ca', 'thal']].copy()
        self.targets = result[['num']].copy()



    def getfeatures(self):
        return self.features

    def gettarget(self):
        return self.targets


