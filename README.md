# HyperDash
This is HyperDash, you go-to hyperparameter tuning dashboard! :fire: :fire:

Creates Dashboard visualizing hyperparameter seach. Implements permutaion tests for models hyperparameer sensitivity.

### Usage 
Import library
``from HyprDash import Dashboard ``
Load data and define hyperparameter grid
```
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

params = {"criterion" : ['gini', 'entropy', 'log_loss'],
          "max_depth" : list(range(3,9)),
          "min_samples_split": list(range(2,7)),
          "min_samples_leaf": list(range(1,5))}

iris = load_iris()
```
Conduct hyperparameter search, save results into "example" directory
```
clf = HyprDash.Dashboard(DecisionTreeClassifier(), params, "exaample")
clf.fit(iris.data, iris.target)
```
Run local server
```
from HyperServer import HyperDashServer
server = HyperDashServer('example')
server.serve_forever()
```


##### To do:
* ~~add table with each hyperparameter score performance in viz site~~
* ~~add permutation tests using numpy (try np.reshape instead of dataframe.groupby)~~
* ~~change bar plots into kernel density esimation plots~~
* ~~add server architecture to host sites~~
* ~~add documentation~~
* restructure into python library
* add demo usage
