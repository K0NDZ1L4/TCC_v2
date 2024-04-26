from data import uciDataset
from PCA import Pca
from SVD import Svd
from CNN import Cnn


data = uciDataset()

features = data.getfeatures()
target = data.gettarget()
pca_features = Pca(features)
svd_features = Svd(features)


cnn = Cnn(pca_features, target)
