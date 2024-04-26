import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
class Svd():
    def __new__(cls, data, num=5):
        # Normalizar dados
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)
        # Aplicar svd
        svd = TruncatedSVD(n_components=num)
        principal_components = svd.fit_transform(scaled)

        # Criar df dos principais componentes
        principal_df = pd.DataFrame(data=principal_components)
        return principal_df
