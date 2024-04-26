import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class Pca():
    def __new__(cls, data, num=5):
        # Normalizar dados
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)
        # Aplicar PCA
        pca = PCA(n_components=num)
        principal_components = pca.fit_transform(scaled)

        # Criar df dos principais componentes
        principal_df = pd.DataFrame(data=principal_components)
        return principal_df
