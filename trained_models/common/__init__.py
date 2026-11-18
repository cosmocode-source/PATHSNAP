"""Shared code for the five PathSnap models: data loading, metrics, plotting and the common experiment harness.

Every model script in trained_models/modelN/ imports from here, so all five models are trained on the same rows and
folds and are evaluated by exactly the same code.
"""
