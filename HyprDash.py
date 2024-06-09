from _socket import _RetAddress
from socketserver import BaseServer
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from http.server import BaseHTTPRequestHandler, HTTPServer
import pandas as pd
import os

class Dashboard(GridSearchCV):

    def __init__(self, model, params, dirname: str, path: str = os.getcwd()) -> None:

        # initialize the parent class GridSearchCVS
        super().__init__(model, params)

        # self.dirname -> directory name of main wrapper directory
        self.dirname = dirname

        # self.model -> machine learning model on which we do hyperparameter analysis
        self.model = model

        # self.params -> dictionary consisting of hyperparameters we do gridsearch on, looks like: {"hyperparameter" : [1,2,3]}
        self.params = params

        # self.path -> path to main project directory excluding the directory at the end
        self.path = path

        # self.PARAMS_KEY -> adresses of hyperparameters serieses in data dataframe
        self.PARAMS_KEY = ["param_" + n for n in params]

        #creating project directory if the directory does not exist already, else returning and error
        if  self.dirname  not in os.listdir(self.path):
            print("Creating project directory at: " + self.path + "/" + self.dirname)
            # main project directory
            os.makedirs(os.path.join(self.path, self.dirname))
            #os.makedirs(self.path + "/" + self.dirname)
            # main figures and plots directory
            os.makedirs(os.path.join(self.path, self.dirname, "viz"))
            #os.makedirs(self.path + "/" + self.dirname + '/' + "viz")
        else:
            raise FileExistsError(f"directory named {self.dirname} already exists in {self.path}")
        


    def fit_and_viz(self, x,y, data_file_name:str = "data") -> None:
        
        # do Gridsearch
        super().fit(x,y)
        
        print("Training ended succesfully.")
        # self.DATA_FILE_NAME -> file name of main data file
        self.DATA_FILE_NAME = data_file_name + ".csv"

        # self.data -> main pandas.Dataframe for data
        self.data = pd.DataFrame(self.cv_results_)

        # deleting unnecessary column
        del self.data["params"]

        # saving gridsearch results into csv file
        self.data.to_csv(self.path + '/' + self.dirname + '/' + self.DATA_FILE_NAME)


        # Generate bar plots of mean score by parameters
        for param in self.PARAMS_KEY:
            score = 'mean_test_score'
            # title: mean test score by param
            title= score.replace("_", " ") +" by "+  param[6:].replace("_", " ")
            # rotating xticks for 360 degrees because it rotates xticks by 90 degrees, god knows why
            self.data.groupby([param])[score].mean().plot(kind="bar", title=title, rot=360, edgecolor = "black").figure.savefig(os.path.join(self.path, self.dirname, "viz", param +"_plot.png"))
            #figures  path: "self.path + '/' + self.dirname + '/' + "viz" + '/' + param +"_plot.png"





if __name__ == "__main__":
    params = {"criterion" : ['gini', 'entropy'], "max_depth" : list(range(4,7)), "min_samples_split": list(range(2,4)), "min_samples_leaf": list(range(1,5))}
    #clf = GridSearchCV(DecisionTreeClassifier(), params)
    iris = load_iris()
    clf = Dashboard(DecisionTreeClassifier(), params, "lollllol")


    clf.fit_and_viz(iris.data, iris.target)