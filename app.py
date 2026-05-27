import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ✔️ Both models are in ONE file
from inference.predict_freight import predict_freight_cost, predict_invoice_flag


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown("""
# Vendor Invoice Intelligence Portal

### AI-Driven Freight Cost Prediction & Invoice Risk Flagging

This system uses machine learning to:

- Forecast freight costs
- Detect risky invoices
- Reduce manual verification effort
""")

st.divider()


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("Select Module")

selected_model = st.sidebar.radio(
    "Choose Prediction Module",
    [
        "Freight Cost Prediction",
        "Invoice Manual Approval Flag"
    ]
)

st.sidebar.markdown("""
### Business Impact
- Better cost prediction  
- Fraud detection  
- Faster finance workflow  
""")


# ---------------------------------------------------
# FREIGHT COST PREDICTION
# ---------------------------------------------------
if selected_model == "Freight Cost Prediction":

    st.subheader("Freight Cost Prediction")

    with st.form("freight_form"):

        col1, col2 = st.columns(2)

        with col1:
            quantity = st.number_input(
                "Quantity",
                min_value=1,
                value=1200
            )

        with col2:
            dollars = st.number_input(
                "Invoice Dollars",
                min_value=1.0,
                value=18500.0
            )

        submit = st.form_submit_button("Predict Freight Cost")

    if submit:

        input_data = {
            "Quantity": [quantity],
            "Dollars": [dollars]
        }

        try:
            result = predict_freight_cost(input_data)
            prediction = result["Predicted_Freight"]

            st.success("Prediction successful")

            st.metric(
                "Estimated Freight Cost",
                f"${prediction[0]:,.2f}"
            )

        except Exception as e:
            st.error(f"Error in freight prediction: {e}")


# ---------------------------------------------------
# INVOICE FLAG PREDICTION
# ---------------------------------------------------
else:

    st.subheader("Invoice Risk Detection")

    with st.form("invoice_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            invoice_quantity = st.number_input("Invoice Quantity", min_value=1, value=50)
            freight = st.number_input("Freight Cost", min_value=0.0, value=1.73)

        with col2:
            invoice_dollars = st.number_input("Invoice Dollars", min_value=1.0, value=352.95)
            total_item_quantity = st.number_input("Total Item Quantity", min_value=1, value=162)

        with col3:
            total_item_dollars = st.number_input("Total Item Dollars", min_value=1.0, value=2476.0)

        submit_flag = st.form_submit_button("Evaluate Invoice")

    if submit_flag:

        input_data = {
            "invoice_quantity": [invoice_quantity],
            "invoice_dollars": [invoice_dollars],
            "Freight": [freight],
            "total_item_quantity": [total_item_quantity],
            "total_item_dollars": [total_item_dollars]
        }

        try:
            result = predict_invoice_flag(input_data)
            flag = result["Predicted_Invoice_Flag"][0]

            if flag == 1:
                st.error("🚨 Invoice requires MANUAL APPROVAL")
            else:
                st.success("✅ Invoice is SAFE for Auto-Approval")

        except Exception as e:
            st.error(f"Error in invoice prediction: {e}")