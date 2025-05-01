# app.py
import streamlit as st
import pandas as pd
from pipeline import predict_price
import matplotlib.pyplot as plt

st.set_page_config(page_title="Housing Price Predictor", layout="centered")
st.title("🏠 Housing Price Prediction App")

st.write("Upload a CSV file containing housing data to predict `SalePrice`. You can also enter a single house manually.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    try:
        # Read uploaded file
        input_df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded and parsed successfully!")

        # Show preview
        st.subheader("Input Preview")
        st.dataframe(input_df.head())

        # Run prediction
        st.subheader("Predicted Prices")
        predictions = predict_price(input_df)
        output_df = input_df.copy()
        output_df["Predicted_SalePrice"] = predictions

        # Summary statistics
        st.write("### 📊 Summary Statistics")
        st.write(f"**Mean Predicted Price:** ${predictions.mean():,.0f}")
        st.write(f"**Min Predicted Price:** ${predictions.min():,.0f}")
        st.write(f"**Max Predicted Price:** ${predictions.max():,.0f}")

        # Plot distribution
        st.write("### 📉 Predicted Price Distribution")
        fig, ax = plt.subplots()
        ax.hist(predictions, bins=30, color='skyblue', edgecolor='black')
        ax.set_title("Distribution of Predicted Sale Prices")
        ax.set_xlabel("Sale Price")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        st.dataframe(output_df[["Predicted_SalePrice"]].head())

        # Download option
        csv = output_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name='predicted_prices.csv',
            mime='text/csv',
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("👈 Upload a CSV file to begin.")

st.markdown("---")
st.header("🏗️ Try a Single House Entry")

with st.form("single_entry_form"):
    st.write("Manually enter details for one property:")

    OverallQual = st.slider("Overall Quality (1-10)", 1, 10, 7)
    GrLivArea = st.number_input("Above Ground Living Area (sq ft)", min_value=300, max_value=5000, value=1710)
    GarageCars = st.slider("Garage Cars", 0, 4, 2)
    GarageArea = st.number_input("Garage Area (sq ft)", min_value=0, max_value=1500, value=548)
    TotalBsmtSF = st.number_input("Total Basement SF", min_value=0, max_value=3000, value=856)
    FullBath = st.slider("Full Bathrooms", 0, 4, 2)
    YearBuilt = st.number_input("Year Built", min_value=1870, max_value=2024, value=2003)
    TotRmsAbvGrd = st.slider("Total Rooms Above Grade", 2, 14, 8)
    Fireplaces = st.slider("Number of Fireplaces", 0, 3, 0)

    submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            input_dict = {
                "OverallQual": OverallQual,
                "GrLivArea": GrLivArea,
                "GarageCars": GarageCars,
                "GarageArea": GarageArea,
                "TotalBsmtSF": TotalBsmtSF,
                "FullBath": FullBath,
                "YearBuilt": YearBuilt,
                "TotRmsAbvGrd": TotRmsAbvGrd,
                "Fireplaces": Fireplaces
            }
            input_df = pd.DataFrame([input_dict])
            pred = predict_price(input_df)
            st.success(f"🏠 Estimated Sale Price: ${pred[0]:,.0f}")
        except Exception as e:
            st.error(f"❌ Could not run prediction: {e}")
