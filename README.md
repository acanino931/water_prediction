# Project Name

## Description
The project provides predictions regarding the potability of water and the corresponding probability that it is potable.

## Process
It loads the original data, preprocesses it, and trains the model.
Afterwards, it reads the prediction data of the independent variables from a CSV file
Finally it prints the predictions in the console and export an excel sheet qith the result in the output folder 

located in the directory /data/data_forecast/ that the user must update, and returns the predictions.

## Installation
Create a python virtual environment from the terminal typing the comand "python -m venv myenv".
Activate the environment with the comand "myenv\Scripts\activate".
Select this new environment as your python interpreter.
Install all the requirements with the comand from the terminal "pip install -r requirements.txt"

## Usage
Input a csv file with the required variables is the path "./data/data_forecast/". 
In this moment some example data are loaded in that folder.
Run the  main file executing the python script.
Check the results in the output folder
