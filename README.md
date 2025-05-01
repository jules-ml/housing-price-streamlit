# Housing Price Prediction App

A Streamlit-based web app that predicts residential home prices using Gradient Boosting and LASSO-selected features. Upload your own housing data, view predictions, analyze price distributions, and test individual property estimates — all in one interactive tool.

## Live Demo (Optional)
You can deploy this project on [Streamlit Cloud](https://streamlit.io/cloud) or host it locally using the steps below.

---

## Features

- Upload CSVs of housing data to get batch predictions
- View summary statistics (mean, min, max) of predicted prices
- Plot the distribution of sale prices
- Predict sale price for a single house with a form-based input
- Download results as a CSV

---

## Technologies Used

- `Streamlit` for the user interface
- `scikit-learn` for modeling and feature selection
- `GradientBoostingRegressor` as the final model
- `LassoCV` for dimensionality reduction
- `matplotlib` for distribution plots
- `joblib` for model serialization

---

## Local Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/jules-ml/housing-price-streamlit.git
cd housing-price-streamlit
pip install -r requirements.txt
