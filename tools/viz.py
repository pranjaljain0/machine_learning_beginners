# visualize the data
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
# Load dataset
url = "/Users/Pranjal/Documents/python/ml/ml_03/dataset.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
names = ['Fixed acidity','Volatile acidity','Citric acid','Residual sugar','Chlorides','Free sulfur dioxide','Total sulfur dioxide','Density','pH','Sulphates','Alcohol','Quality']
# box and whisker plots
dataset = read_csv(url, names=names,sep=';')

dataset.plot(kind='box', subplots=True, layout=(3,4), sharex=False, sharey=False)
pyplot.show()
# histograms
dataset.hist()
pyplot.show()
# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()