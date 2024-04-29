Bayesian Active Learning
==============================
Modern Deep Neural Networks need large amount of data to perform well but annotating data is expensive. Active Learning presents a solution by selecting the most informative data samples to annotate.

## Motivation
- Annotating data is expensive. 
- Modern deep neural networks are data hungry.
- Sample datapoints that maximize information gain w.r.t. model parameters.
- Reduce cost of training ML models.

Bayesian Active Learning
- Consider the uncertainty w.r.t. the model parameters.


## Experiments and Results
### TwoMoons Decision Boundaries
Investigating the sampling behavior of each acquisition function it is clear that:
- Margin sampling selects instances where the decision margin (difference between the first most probable and second most probable predicted labels) is smallest. In the context of the TwoMoons dataset, margin sampling is likely to focus on instances near the decision boundary between the two moons. This is because instances near the boundary are those where the model is most uncertain between the two classes, resulting in smaller margins.
- Entropy samples where the entropy is highest, which is highest when the predictive distribution is uniform. This is most likely to happen along the decision boundary.
- BALD is also more likely to sample along the decision boundary (frist term), but due to the second term, samples that the model agree are confusing are given a large negative weight.


![image](https://github.com/MadsBirch/bal/assets/23211921/fde592d2-9388-4067-b726-78a218fe127e)


Project Organization
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
