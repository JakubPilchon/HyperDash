from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from http.server import BaseHTTPRequestHandler, HTTPServer
import pandas as pd
import matplotlib.pyplot as plt
import os

class Dashboard(GridSearchCV):

    def __init__(self, model, params, dirname: str, path: str = os.getcwd()) -> None:
        self.is_fitted = False

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
        
        self.is_fitted = True
        
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

            with plt.style.context('dark_background'):
                fig, ax = plt.subplots()
                self.data.groupby([param])['mean_test_score'].mean().plot(kind= 'bar', ax=ax,
                                                                         edgecolor="w",rot=360,title=title, color=['#AB81CD'])
            ax.set_facecolor("#383C43")
            fig.set_facecolor("#2B2E33")

            fig.savefig(os.path.join(self.path, self.dirname, "viz", param +"_plot.png"))

            #with plt.style.context('dark_background'):
                #plt.bar(self.data.groupby([param])[score].mean(), title=title, rot=360, edgecolor = "white", color=['#AB81CD'])
                #self.data.groupby([param])[score].mean()
                #.plot(kind="bar", title=title, rot=360, edgecolor = "white", color=['#AB81CD']).figure.savefig(os.path.join(self.path, self.dirname, "viz", param +"_plot.png"))
            #figures  path: "self.path + '/' + self.dirname + '/' + "viz" + '/' + param +"_plot.png"

        #create websites
        self.create_mainwebsite()
        self.create_viz_website()

    def create_mainwebsite(self):
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
        <br>
            [table]
        </div></body></html>'''

        # get insert table with gridsearch results into layout
        html_text = html_text.replace("[table]", self.data.sort_values("rank_test_score").reset_index().to_html(columns=['mean_fit_time', 'split0_test_score', 'split1_test_score',
       'split2_test_score', 'split3_test_score', 'split4_test_score', 'mean_test_score', 'std_test_score', 'rank_test_score'] + self.PARAMS_KEY, border=0))
        
        #create website file
        file_name = "index.html"
        
        with open(os.path.join(self.path, self.dirname, file_name), "w") as f:
            f.write(html_text)

    def create_viz_website(self):

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
                    width: 1000px;
                    background-color: rgb(43,46,51);
                    border: 3px solid rgb(56, 60, 67);
                    border-radius: 5px;

                    gap: 10px;
                }
                #title_and_viz {
                    min-width: 680px;
                    float: left;
                }
                #stats {
                    float: right;
                    width: 320px;
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
                img {
                    display: block;
                    text-align: center;
                    margin-left: 1%;
                    margin-right: auto;
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
                <h3>Score std: {}</h3>
                <h3>time std: {}</h3><br></div></div>'''

        table = ''
        #iterate over hyperparameters
        for param in self.PARAMS_KEY:
            # add title
            title = param[6:]
            # add path to plots
            source = os.path.join("viz", param+'_plot.png')
            # add standard deviation of mean model scores by hyperparameter
            score_std = str(self.data.groupby([param])['mean_test_score'].mean().std())[:7]
            # add standard deviation of mean model training time by hyperparameter
            time_std = str(self.data.groupby([param])['mean_fit_time'].mean().std())[:7]

            # add feature container html code into main threshold
            table += container.format(title, source, score_std, time_std)
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