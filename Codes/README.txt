Code Structure and Usage

This folder contains all scripts required for model training and test data evaluation.

ADNI Data Structure
- adni_data.R: Formats the FC and corresponding datasets for further downstream analysis.

MODEL DEFINITIONS
- FUNGVA_model_definition.R: Defines the proposed FUNGVA model.
- MLP_FiLM1_model_definition.R, MLP_FiLM2_model_definition.R: Define the competing VAE models with FiLM-based decoders.

MODEL SELECTION
- Fungva_model_choice.R: Used to select the optimal latent dimension and hyperparameters for the FUNGVA model. This script can be adapted for the competing models.

SUPPORTING UTILITIES
- libraries.R: Loads required R packages.
- loss_metric.R: Implements loss functions, including VAE loss (reconstruction + KL divergence) and the penalty term.
- datasets.R: Handles data loading, preprocessing, and application of model-specific masks.

MODEL TRAINING AND SAVED WEIGHTS
- FUNGVA_ld12_lam0.1.R: Trains the FUNGVA model (latent_dim = 12, λ = 0.1).
- MLP_FiLM1_ld10_lam0.1.R: Trains the MLP_FiLM1 model (latent_dim = 10, λ = 0.1).
- MLP_FiLM2_ld10_lam1.R: Trains the MLP_FiLM2 model (latent_dim = 10, λ = 1).

Each script saves trained model weights across three random seeds.

TEST EVALUATION
- test_evaluation.R: Computes performance metrics on the test dataset.
- test_evaluation_heatmaps.R: Generates heatmaps comparing observed and predicted differences across models.
