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


### Understanding BALD
BALD:
- The first term selects samples with high predictive uncertainty. 
- The second term down-weigh samples that are inherently ambiguous. 

![image](https://github.com/MadsBirch/bal/assets/23211921/52149802-b0af-4ce9-bfec-de47177a6f98)



### MNIST Learning Curves
The goal of this project was to replicate the results of the paper Deep Bayesian Active Learning with Image Data (https://arxiv.org/abs/1703.02910). Similar to Gal et al 2017, I found that implementing an active learning framework, selecting the most informative data points, outperformed a standard random sampling strategy (Fig. 1). The BALD acquisition function assigns the highest scores to data points which are most informative w.r.t. the model parameters.

Setup:
- Monte Carlo Dropout (T=10)
- Query batch size of 100.

Findings:
- AL learn faster and plateau at a higher accuracy.

![Fig. 1 - MNIST test accuracy as a function of number of samples quired from the unlaballed pool. Three acquisition functions were implemented; random, BALD and BatchBALD.](https://github.com/MadsBirch/bal/assets/23211921/766c2d82-180c-4ebd-b705-cc6fe9ff16c1)

### BALD vs BatchBALD
Batch aware methods are necessary as Deep Neural Networks are expensive to train, meaning that adding only one data point to the training set does not justify re-training of the model considering the small amount of additional information gained from a single data point. Non batch aware acquisition functions can be and are used to query multiple data points, however this is not optimal, since the highest ranking data points are often similar. Batch aware methods such as BatchBALD takes this correlation between the data points in the query set, maximising the diversity of the queried samples.

Motivation for BatchBALD:
- Deep Neural Networks are computationally expensive to train.
- Ensuring batch diversity to maximize information gain.
- BALD for multiple samples sums up the mutual information for each sample, which means that the overlaps are doulbe counted.
- BatchBALD conisders the joint mutual iniormation.
  

![Screenshot 2024-04-29 at 11 18 41](https://github.com/MadsBirch/bal/assets/23211921/af2e1612-af35-4b59-89f1-e64edf77aba4)


Results:
- No difference between BALD and BatchBALD was found, but the batch size was small, 4.
- A larger batch size would likely change the results, but increasing the batch size increases the memory footprint.

![image](https://github.com/MadsBirch/bal/assets/23211921/40ab8499-3382-42f5-9235-ab974d709036)


## Conclusion
What was found?
1. Active Learning leads to faster learning and higher accuracy.
2. The desired sampling behavior was confirmed in 2D.
3. BALD did not lead to faster learning than margin or entropy sampling, but came at a computional cost.
4. BatchBALD was not preffered over BALD for a batch size of 4.


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
