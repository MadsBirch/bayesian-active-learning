bayesian-active-learning
==============================

This is the repository for the project on Bayesian Active Learning.


## Motivation
- Modern Deep Neural Networks need large amount of data to perform well but annotating data is expensive. Active Learning presents a solution by selecting the most informative data samples to annotate.

- Batch aware methods are needed as Deep Neural Networks are data hungry and training the model is resource heavy. Hence, training the model again after adding only one new sample to the training set is not feasible. For that reason we need batch aware methods, selecting a diverse set of samples. We cannot simply choose the top ranking samples as datasets often contain many near duplicate samples. BatchBALD is an example of a batch aware methods, where the correlation between the query batch samples are taken into account.

- Labelling all samples is expensive and leads to redundant labels. Labelling randomly also leads to redundant samples. By labelling actively the aim is to reduce the number of redundant samples by querying samples that maximise information gain w.r.t. model parameters.


## Results
The goal of this project was to replicate the results of the paper *Deep Bayesian Active Learning with Image Data* (https://arxiv.org/abs/1703.02910). Similar to Gal et al 2017, I found that implementing an active learning framework, selecting the most informative data points, outperformed a standard random sampling strategy (Fig. 1). The BALD acquisition function assigns the highest scores to data points which are most informative w.r.t. the model parameters. 

<br />
<figure>
    <img src="https://github.com/MadsBirch/bayesian-active-learning/blob/master/reports/figures/figures/AL_MNIST.png"  width="80%">
    <figcaption> <em>Fig. 1 - MNIST test accuracy as a function of number of samples quired from the unlaballed pool. Three acquisition functions were implemented; <em>random, BALD and BatchBALD.</em></figcaption>
</figure>

## Project Organization

------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
