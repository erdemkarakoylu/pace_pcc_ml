This repository contains code for a machine learning effort to predict Phytoplankton Community Composition (PCC) from Hyperpsectral sea surface reflectance data as well as ancillary data.
The data is simulated global ocean color scene encompassing 31 days that represent the month of December 2021.
The predictive model is an XGBoost Regressor with a Multi- (7-) output regressor head. The model can predict 6 types of phytoplankton and total chlorophyll a. The phytoplankton types are:
* Diatoms (dia)
* Chlorophytes (chl)
* Cyanobacteria (cya)
* Coccolithophores (coc)
* Dinoflagellates (din)
* Phaeocystis (pha)

Model's Bayesian hyperparameter optimization was conducted using Optuna.

Shapley values were used to explain predictions. 

The manuscript repository can be found [here](https://github.com/erdemkarakoylu/PCC_PACE_manuscript/tree/main).

