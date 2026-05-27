import joblib
import pandas as pd
import os


# ---------------------------------------------------
# BASE PATH
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model path (adjust if needed)
MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "invoice_flagging",
    "models",
    "predict_flag_invoice.pkl"
)


# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
def load_model(model_path: str = MODEL_PATH):
    """
    Load trained ML model
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)

    return model


# ---------------------------------------------------
# FREIGHT COST PREDICTION
# ---------------------------------------------------
def predict_freight_cost(input_data):
    """
    Predict freight cost using trained model
    """

    model = load_model()

    df = pd.DataFrame(input_data)

    # Prediction
    df["Predicted_Freight"] = model.predict(df)

    return df


# ---------------------------------------------------
# INVOICE FLAG PREDICTION
# ---------------------------------------------------
def predict_invoice_flag(input_data):
    """
    Predict invoice risk flag (0 = safe, 1 = risky)
    """

    model = load_model()

    df = pd.DataFrame(input_data)

    df["Predicted_Invoice_Flag"] = model.predict(df)

    return df


# ---------------------------------------------------
# LOCAL TESTING
# ---------------------------------------------------
if __name__ == "__main__":

    sample_data = {
        "invoice_quantity": [100, 50, 25, 10],
        "invoice_dollars": [18500, 9000, 3000, 200],
        "Freight": [500, 250, 100, 20],
        "total_item_quantity": [100, 50, 25, 10],
        "total_item_dollars": [18500, 9000, 3000, 200]
    }

    print(predict_invoice_flag(sample_data))
    print(predict_freight_cost(sample_data))