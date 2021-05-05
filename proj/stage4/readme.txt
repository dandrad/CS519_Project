1) Code Organization:

To keep our code modular and portable, our code is found in several different files
each with a distinct purpose. Primarily, our code is contained in Jupyter Notebooks
as we found these to be the easiest to work with. In each Jupyter Notebook, we
facilitate one task, such as implementing a specific model or creating a type of graph.
The only code file which is not in a Jupyter Notebook is the python script which
handles loading and processing the raw data. This is left in a single python script,
and largely in a single function, so that the data can be reduced and accessed
in any program you'd like. It contains parameters which allow you to control
what sort of processing takes place, such as the encoding, feature selection etc.

Files:
load_and_reduce.py - handles loading and processing raw data
data_barplot_exploration.ipynb - handles creating bar plots to explore data
data_boxplot_exploration.ipynb - handles creating box plots to explore data
rf_model_testing.ipynb - handles implementing and evaluating random forest regressors, grid search to find parameters
linear regression + RANSAC.ipynb - handles running and testing linear regression and RANSAC models
correlation matrix.ipynb - handles correlation plots, basic linear model
Austin_Animal_Center_Intakes.csv - Raw Intake Data
Austin_Animal_Center_Outcome.csv - Raw Outcome Data


2) Commands to Run Code:
Because our code is contained in Jupyter Notebooks, no real commands are needed.
To run our code, simply launch a Jupyter Notebook browser, and open the desired program.
Assuming support for Jupyter Notebooks are installed, to start a Jupyter Notebook
in terminal, you can run the command "jupyter notebook" in any directory to open
up a navigation page, or to open a specific Jupyter Notebook, run "jupyter notebook filename.ipynb"


3) Dataset Information
The most up to date Intake and Outcome data can be found here:
https://data.austintexas.gov/browse?q=austin%20animal%20center&sortBy=relevance&utf8=%E2%9C%93

The dataset is contained in two separate files, Austin_Animal_Center_Intakes.csv and
Austin_Animal_Center_Outcomes.csv.

Austin_Animal_Center_Intakes.csv - Contains information of animals when they enter shelter.
Contains ID, name, Date/Time of Intake, Condition, Intake Type, Animal Type, Sex upon Intake, Age, Breed, Color

Austin_Animal_Center_Outcomes.csv - Contains information of animals when they leave shelter.
Contains ID, name, Date/Time of Outcome, Condition, Outcome Type, Animal Type, Sex upon Intake, Age, Breed, Color

These two datasets are used in tandem to form the dataset used in our models.
The difference in time of Outcome/Intake is calculated to determine the time in shelter,
serving as the target, and remaining columns are used as attributes in the models.
The datasets are merged on the unique Animal ID, and for animals with more than one
intake/outcome, the most recent entries are used. More information about the dataset
and how it is processed can be found in the report.
