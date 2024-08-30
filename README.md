# collaborations-UFRO-2024-2s

UFRO IngMat students, 2024 second semester. 
- Teacher: Andres Avila Barrera
- Paranal expert: Juan Pablo Gil

## Motivation
Below there are two proposed challenges based on the [Paranal Observations](https://huggingface.co/datasets/Paranal/parlogs-observations) log dataset and real needs in the Paranal Observatory:

1. [Command visualization app](./1-command-visualization/task_description.ipynb)
2. [Compare Naive Bayes and Sentiment Analysis to predict errors](./2-NB-SA-comparison/task_description.ipynb)

The specific instructions and some examples are available in each directory.

## Instructions

### General
* Please use Python 10 or newer in your system
* Any library must be declared in requirements.txt
* Select the directory of your challenge, fill the preamble including the authors
* All the notebooks must be able to be executed at any time. Don't break compatibility 
* Commit often!

### Notebook organization:
* task_description.ipynb : Motivation and some first examples.
* results.ipynb : Final report, with at least executable examples, screenshots, user manual (if applicable) and discussion of results. 
* 2024-MM-DD: interim work by week or month, useful to track progress.

### References

1. https://huggingface.co/datasets/Paranal/parlogs-observations : Dataset of Paranal Observations 
1. https://github.com/paranal-sw/parlogs-observations : Tutorials in Parlogs Observations dataset

## Install

```bash
# Download the repo
git clone https://github.com/paranal-sw/collaborations-UFRO-2024-2s.git

# Create the virtual environment
cd collaborations-UFRO-2024-2s
python -m venv venv

# Activate the environment
source venv/bin/activate
python -m pip install --upgrade pip wheel

# Install the Python libraries
pip install -r requirements.txt

# To enable Jupyter
python -m ipykernel install --user --name=venv --display-name "Python (venv)"

# Start Jupyter Notebooks (unless you have your own editor like VS Code)
jupyter-notebook
```

