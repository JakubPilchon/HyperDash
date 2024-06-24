from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time

class Dashboard(GridSearchCV):
    r"""
    Class creating Hyperdash directory containing HTML files and plots. 

    This class inherits form `sklearn.model_selection_GridSearchCV`.

    Parameters:
        model : 
             the scikit-learn estimator interface. The machine learning model on which we conduct hyperparameter search
        params : dict
             dictionary consisting of hyperparameters we do gridsearch on, looks like: {"hyperparameter" : [1,2,3]}
        dirname : str
             directory name of main wrapper directory
        path : str, optional
             path where we want to hyperdash directory to appear, if not provided path is assumed to be current working directory

    Attributes:
        PARAMS_KEY : list of strings
             list of hyperparameters, with 'param_' added on the beggining. Adresses of hyperparameters serieses in data DataFrame.
        is_fitted : Bool = False
             stores information whether `fit()` mehod was previously called.
                    
    Atributes after calling fit() method:
        data : pandas.DataFrame
             Dataframe containing gridsearch results
        time : float
             Time duration of gridsearch
        DATA_FILE_NAME : str
             file name on which are stored gridsearch results

    Public Methods:
        fit() :
             Conducts hyperparametes Gridsearch. Saves its results into `data` and "data.csv".
             Generates and save visualizations plots into '\viz' directory. Creates HTML dasboards.
        permutation_tests() :
             Conducts permutation tests over ALL of hyperparameters.

    Private Methods:
        __create_mainwebsite():
             Creates main HTML dashboard. Saves it as "index.html".
             Site consists of basic information panel, and table displaying Gridsearch results.
        __create_viz();
             Creates main HTML dashboard. Saves it as "viz_site.html".
             Site consists of panels containing plots and other informations.
        """
  

    def __init__(self, model, params, dirname: str, path: str = os.getcwd()) -> None:
        self.is_fitted = False

        # initialize the parent class GridSearchCVS
        super().__init__(model, params)

        self.dirname = dirname

        self.model = model

        self.params = params

        self.path = path

        self.PARAMS_KEY = ["param_" + n for n in params]

        #creating project directory if the directory does not exist already, else returning an error
        if  self.dirname  not in os.listdir(self.path):
            print("Creating project directory at: " + self.path + "/" + self.dirname)
            # main project directory
            os.makedirs(os.path.join(self.path, self.dirname))
            # main figures and plots directory
            os.makedirs(os.path.join(self.path, self.dirname, "viz"))
        else:
            raise FileExistsError(f"directory named {self.dirname} already exists in {self.path}")
        


    def fit(self, x,y, data_file_name:str = "data", test_iter:int = 2000, test_alpha:float = .05) -> None:
        r"""
            Conducts hyperparametes Gridsearch. Saves its results into `data` and "data.csv".
            Generates and save visualizations plots into '\viz' directory. Creates HTML dasboards.

            Parameters:
                x : array-like
                     data on which model is tested.
                y : array-like
                     data target on which model perforamnce is evaluated.
                data_file_name : str, optional
                     file name on which gridsearch results are stored. Default is "data".
                test_iter : int, optional
                     iterations of permutation tests. Default is `2000`.
                test_alpha : floay, optional
                     threshold for statistical significance. Default is `0.05`. Needs to be a value between `0.` and `1.0`.
                     If p value is smaller than `test_alpha` then it is safe to assume that alternative hypothesis is true.
        """

        if not (test_alpha > 0. and test_alpha < 1.):
            raise ValueError("test_alpha value must be between 0 and 1. ")
        
        self.is_fitted = True
        
        # conduct Gridsearch, also measure time elapsed during searching
        start = time.time()
        super().fit(x,y)
        end = time.time()

        self.time = end - start
        
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
            title= "Scores distribution by "+  param[6:].replace("_", " ")

            with plt.style.context('dark_background'):
                fig, ax = plt.subplots()
                self.data[[param, "mean_test_score"]].groupby(param)["mean_test_score"].plot(kind="kde", ax=ax, legend=True, title=title, xlabel="Score")
            ax.set_facecolor("#383C43")
            fig.set_facecolor("#2B2E33")

            fig.savefig(os.path.join(self.path, self.dirname, "viz", param +"_plot.png"))
                        
            #figures  path: "self.path + '/' + self.dirname + '/' + "viz" + '/' + param +"_plot.png"

        # Conduct permutaton tests
        p_values = self.permutation_tests(test_iter)
        #create websites
        self.__create_mainwebsite()
        self.__create_viz_website(p_values, test_alpha)

    def permutation_tests(self, N:int) -> dict:
        """
            Conducts permutation tests over ALL of hyperparameters.
            Permutation Test hypothesis:
                h_0 -> The observed standard deviation in grouped hyperparameter mean scores is due to random chance and NOT due to diffrences in given hyperparameter
                h_n -> The observed standard deviation in grouped hyperparameter mean scores is due to diffrences in given hyperparameter and not random chance

            Warning: `fit()` method needs to be called beforehand.

            Parameters:
                N : int
                    Number of iterations of permutation tests.
            
            Returns:
                p_values : dict
                    Dictioanary consisting of `{"hyperparameter" : p value}` pairs. p value is percents.
        """
        if not self.is_fitted:
            raise Exception("You need to call self.fit_and_viz beforehand")
        
        # Permutation Test
        # h_0 -> The observed standard deviation in grouped hyperparameter mean scores is due to random chance and NOT due to diffrences in given hyperparameter
        # h_n -> The observed standard deviation in grouped hyperparameter mean scores is due to diffrences in given hyperparameter and not random chance

        #real_data - a hashmap consisting of pairs {parameter_name: standard deviation of mean scores by hyperparameter}
        real_data = {param:self.data[["mean_test_score", param]].groupby(param).mean().std().values[0] for param in self.PARAMS_KEY}

        # mock_data is numpy array for efficency reasons 
        mock_data = np.array(self.data["mean_test_score"])

        # dictionary consisting of pairs: {parameter_name: 0}
        # It will be used to store p_values
        p_values = {param:0 for param in self.PARAMS_KEY}

        # N - number of iterations
        for _ in range(N):
            #rearange mock_data without replacement 
            mock_data = np.random.permutation(mock_data)

            # iterate over every hyperparameter tested
            for param in self.PARAMS_KEY:
                # variations - number of possibilities in hyperparameter
                variations = len(self.params[param[6:]])
                # reshape dataset in order to mimick grouping by hyperparameter
                np_data = mock_data.reshape((variations, int(len(mock_data)/variations)))   
                
                # if simulated outcome is bigger or equal to the observed in real dataset, then add 1 to p_values dict
                if real_data[param] <= np.std(np.mean(np_data, axis=-1)):
                    p_values[param] += 1

        # write p_values into percentiles
        p_values = {param:p_values[param]/N * 100 for param in p_values}

        return p_values



    def __create_mainwebsite(self) -> None:
        """
        Creates main HTML dashboard. Saves it as "index.html".
        Site consists of basic information panel, and table displaying Gridsearch results.

        Warning: `fit()` method needs to be called beforehand.
        """

        if not self.is_fitted:
            raise Exception("You need to call self.fit_and_viz beforehand")

        # site with tables layout:
        html_text = '''
        <!DOCTYPE html>
        <html lang = "en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>HyprDash</title>
            <style>
                body {
                    font-family: Arial, Helvetica, sans-serif;
                    background: #35383d;
                    color: white;
                }
                div.header {
                    display: inline-flex;
                    width: 99%;
                    background: rgb(101,69,151);
                    background: linear-gradient(84deg, rgba(101,69,151,1) 0%, rgba(87,74,226,1) 100%);
                    text-align: center;
                    padding: 10px;
                }
                ul {
                    
                    list-style-type: none;
                    padding: 10px;
                }
                li:hover {
                    background: #AB81CD;
                }
                li {
                    float: right;
                    display: inline;
                    padding-inline: 20px;
                    padding-top: 10px;
                    padding-bottom: 10px;
                }
                a {
                    color: white;
                    text-decoration: none;
                }
                table.dataframe {
                    padding: 10px;
                    margin-left: auto;
                    margin-right: auto; 
                    text-align: center;
                    border-style:none;
                }
                                div.info {
                    padding-top: 5px;
                    padding-bottom: 5px;
                    margin-left: auto;
                    margin-right: auto;
                    margin-top: 10px;
                    margin-bottom: 10px;
                    width: 1000px;
                    background-color: rgb(43,46,51);
                    border: 3px solid rgb(56, 60, 67);
                    border-radius: 5px;
                    gap: 10px;
                    display: flex;
                }
                div.separator {
                    width: 50%;
                    font-size: large;
                    padding-left: 10px;
                }
                #overview {
                    border: 3px solid rgb(66, 70, 77);
                    border-radius: 5px;
                    background: #AB81CD;
                    padding: 5px;
                    margin-left: 0px;
                    max-width: 50%;
                    border-radius: 10px;
                    margin-top: 10px;
                    font-size: larger;
                }
                th {
                    background-color:rgb(101,69,151);
                    padding: 5px;
                }
                tr:nth-child(even) {
                background-color: rgb(43,46,51);
                }
                tr:nth-child(odd) {
                background-color: rgb(56, 60, 67);
                }
                tr:hover {
                    background-color: rgb(33,36,41);
                }       
            </style>
        </head>

        <body>
        <div class="header">
            <h1>HyprDash</h1>
            <ul>
                <li><a href="index.html">Results</a></li>
                <li><a href="viz_site.html">Hyperparameter Analytics</a></li>
            </ul>
        </div>
        
        <div class="table">
            [info]
        <br>
            [table]
        </div></body></html>'''

        # overview table, to be pasted in [info]
        overview = """
                <div class = "info"><div class="separator">
                    <div id="overview"><b>Overview</b></div><br> Number of iterations: {} <br><br> Best model score: {} <br> Mean  model score: {} <br> Model score standard deviation: {}
                </div>
                <div class="separator">
                        <br><br><br>Tuning time: {}s <br> <br> Best time: {}s <br> Mean model time: {}s <br> Model time standard deviation: {}s </div></div>"""
        
        overview = overview.format(len(self.data), str(self.best_score_), self.data["mean_test_score"].mean(), self.data["mean_test_score"].std(),
                                    self.time, self.data["mean_fit_time"].min(), self.data["mean_fit_time"].mean(), self.data["mean_fit_time"].std())

        html_text = html_text.replace("[info]", overview)
        # insert table with gridsearch results into layout
        html_text = html_text.replace("[table]", self.data.sort_values("rank_test_score").reset_index().to_html(columns=['mean_fit_time', 'split0_test_score', 'split1_test_score',
       'split2_test_score', 'split3_test_score', 'split4_test_score', 'mean_test_score', 'std_test_score', 'rank_test_score'] + self.PARAMS_KEY, border=0))
        
        #create website file
        file_name = "index.html"
        
        with open(os.path.join(self.path, self.dirname, file_name), "w") as f:
            f.write(html_text)

    def __create_viz_website(self, p_values:dict, alpha:float) -> None:
        """
        Creates main HTML dashboard. Saves it as "viz_site.html".
        Site consists of panels containing plots and other informations. 

        Warning: `fit()` method needs to be called beforehand.

        Parameters:
            p_values : dict
                 Dictionary containing pairs: `{"hyperparameter name" : p value}`
            alpha : float
                 threshold for statistical significance.
        """

        if not self.is_fitted:
            raise Exception("You need to call self.fit_and_viz beforehand")
        
        vis_site = '''<!DOCTYPE html>
        <html lang = "en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>HyprDash</title>
            <style>
                body {
                    overflow: scroll;
                    font-family: Arial, Helvetica, sans-serif;
                    background: #35383d;
                    color: white;
                }
                div.header {
                    display: inline-flex;
                    width: 99%;
                    background: rgb(101,69,151);
                    background: linear-gradient(84deg, rgba(101,69,151,1) 0%, rgba(87,74,226,1) 100%);
                    text-align: center;
                    padding: 10px;
                }
                ul {
                    
                    list-style-type: none;
                    padding: 10px;
                }
                li:hover {
                    background: #AB81CD;
                }
                li {
                    float: right;
                    display: inline;
                    padding-inline: 20px;
                    padding-top: 10px;
                    padding-bottom: 10px;
                }
                a {
                    color: white;
                    text-decoration: none;
                }
                div.main_container {
                    padding: 20px;
                    display: grid;
                    gap: 10px;
                }
                div.feature_container {
                    padding-bottom: 5px;
                    margin-left: auto;
                    margin-right: auto;
                    min-width: 1100px;
                    background-color: rgb(43,46,51);
                    border: 3px solid rgb(56, 60, 67);
                    border-radius: 5px;

                    gap: 10px;
                }
                table.dataframe {
                    margin: 10px;
                }
                #title_and_viz {
                    min-width: 680px;
                    float: left;
                }
                #stats {
                    float: right;
                    min-width: 420px;
                }
                #title {
                    border: 3px solid rgb(66, 70, 77);
                    border-radius: 5px;
                    background: #AB81CD;
                    padding: 2px;
                    max-width: 50%;
                    height: 10%;
                    border-radius: 10px;
                    margin: 10px;
                }
                #pvalue {
                  max-width: 350px;
                  border-radius: 5px;
                  border: 3px solid rgb(56, 60, 67);
                  margin: 5px;
                  overflow: auto;
                  padding: 5px;
                }
                img {
                    display: block;
                    text-align: center;
                    margin-left: 1%;
                    margin-right: auto;
                }
                th {
                    background-color:rgb(101,69,151);
                    padding: 5px;
                }
                tr:nth-child(even) {
                background-color: rgb(43,46,51);
                }
                tr:nth-child(odd) {
                background-color: rgb(56, 60, 67);
                }
                tr:hover {
                    background-color: rgb(33,36,41);
                }       
            </style>
        </head>

        <body>
        <div class="header">
        <h1>HyprDash</h1>
        <ul>
            <li><a href="index.html">Results</a></li>
            <li><a href="viz_site.html">Hyperparameter Analytics</a></li>
            </ul>
        </div>
        <div class="main_container">

                <!-- Main data analysis container-->
            [features]
            
            

        </div></body></html>'''

        container = '''<div class="feature_container">
            <div id="title_and_viz" class = "separator">
                <div id="title"><h2>{}</h2></div>
                <div id="image_containter"><img src='{}' alt="text"></div>
            </div>

            <div id="stats" class = "separator"> 
                {} <br> <div id="pvalue"> {} <br> P value: {}</div></div></div>'''

        table = ''

        #iterate over hyperparameters
        for param in self.PARAMS_KEY:
            # add title
            title = param[6:]
            # add path to plots
            source = os.path.join("viz", param+'_plot.png')

            # generate score performance table using pandas
            scores_table = self.data[["mean_test_score", param]].groupby(param).mean().reset_index().rename(columns={'mean_test_score': 'Mean test score', param: param[6:].capitalize().replace("_", " ")}).to_html(border=0)

            # test if p value is bigger or equal to alpha
            if p_values[param] >= alpha * 100:
                # if p value is bigger than alpha value, then we suppose that alternative hypothesis is true
                true_h = "The observed standard deviation in grouped hyperparameter mean scores is due to random chance and NOT due to diffrences in given hyperparameter"
            else:
                # otherwise it is safe to assume that alternative hypothesis is true
                true_h = "The observed standard deviation in grouped hyperparameter mean scores is due to diffrences in given hyperparameter and not random chance"

            # add feature container html code into main threshold
            table += container.format(title, source, scores_table, true_h ,p_values[param])
        # add feature containers to main sites
        vis_site = vis_site.replace('[features]', table)

        # save sites as viz_site.html inside project directory
        with open(os.path.join(self.path, self.dirname, 'viz_site.html'), "w") as f:
            f.write(vis_site)

if __name__ == "__main__":
    params = {"criterion" : ['gini', 'entropy'], "max_depth" : list(range(4,7)), "min_samples_split": list(range(2,4)), "min_samples_leaf": list(range(1,5))}
    #clf = GridSearchCV(DecisionTreeClassifier(), params)
    iris = load_iris()
    clf = Dashboard(DecisionTreeClassifier(), params, "lollllol")

    clf.fit_and_viz(iris.data, iris.target)