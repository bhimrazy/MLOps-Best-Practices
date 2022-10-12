# MLOps-Best-Practices

This repository contains about the way to start your MLOps Journey and it's best practices.

## Experiment Tracking

Experiment tracking is the processing of managing all the different experiments and their components, such as parameters, metrics, models and other artifacts.

    - Organize all the necessary components of a specific experiment.
      It's important to have everything in one place and know where
      it is so you can use them later.
    - Reproduce past results (easily) using saved experiments.
    - Log iterative improvements across time, data, ideas, teams, etc.

## Tools

These are fantastic tools that provide features like dashboards, seamless integration, hyperparameter search, reports and even debugging!

    - MLFlow : 100% Free and open-source (used by Microsoft, Facebook, Databricks and others.)
    - Comet ML (used by Google AI, HuggingFace, etc.)
    - Neptune (used by Roche, NewYorker, etc.)
    - Weights and Biases (used by Open AI, Toyota Research, etc.)

## convert notebook to markdown

```shell
    pip install nbconvert
    sudo apt-get install pandoc
    jupyter nbconvert --to markdown notebook.ipynb

```
