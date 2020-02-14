# Description

Here contains the code and notebook for the experiments part of the paper: **Finding Plausible Explanations First, Not as an Afterthought**. To run the code in this repository, you have to have a **windows** environment and make sure you have installed python version > 3.6.

# Reproduce the experiments

1. Clone the repository and install the dependencies. All of the dependencies should be able to be installed via `pip`. 
2. All of the experiments are in `experiments.ipynb`. Just open it using Jupyter Notebook and run all of the cells. 
3. The datasets used in the experiments are provided in `data` folder. If you want to change dataset in the `experiments.ipynb`, simply change the string parameter passed into the function `dataset_config`. The choices of the dataset include 'zoo', 'adult', 'lending', 'HELOC'. 
4. Please note that the expresso program is required for the experiments and is provided as part of the repository. 