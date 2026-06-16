import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #e0f2fe, #f0fdfa);
}
.card {
    background-color: #ffffff;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
h1, h2, h3 {
    color: #0f172a;
}
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    padding: 10px 18px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align:center;'>📉 Customer Churn Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Predict the probability of customer churn using Machine Learning</p>", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    return df

df = load_data()

# ---------------- DATA PREPROCESSING ----------------
X = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
y = df['Churn']

X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
y = y.str.strip().map({'Yes': 1, 'No': 0})

mask = X.notna().all(axis=1) & y.notna()
X = X[mask]
y = y[mask]

# ---------------- TRAIN MODEL ----------------
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LogisticRegression()
model.fit(x_train, y_train)

#------------------------PLOTS-----------------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📉 Logistic Regression Curves for Input Features")

# Create ranges
tenure_range = np.linspace(X['tenure'].min(), X['tenure'].max(), 200)
monthly_range = np.linspace(X['MonthlyCharges'].min(), X['MonthlyCharges'].max(), 200)
total_range = np.linspace(X['TotalCharges'].min(), X['TotalCharges'].max(), 200)

# Mean values
tenure_mean = X['tenure'].mean()
monthly_mean = X['MonthlyCharges'].mean()
total_mean = X['TotalCharges'].mean()

# Prepare data for each curve
curve_tenure = pd.DataFrame({
    'tenure': tenure_range,
    'MonthlyCharges': monthly_mean,
    'TotalCharges': total_mean
})

curve_monthly = pd.DataFrame({
    'tenure': tenure_mean,
    'MonthlyCharges': monthly_range,
    'TotalCharges': total_mean
})

curve_total = pd.DataFrame({
    'tenure': tenure_mean,
    'MonthlyCharges': monthly_mean,
    'TotalCharges': total_range
})

# Scale
curve_tenure_scaled = scaler.transform(curve_tenure)
curve_monthly_scaled = scaler.transform(curve_monthly)
curve_total_scaled = scaler.transform(curve_total)

# Predict probabilities
prob_tenure = model.predict_proba(curve_tenure_scaled)[:, 1]
prob_monthly = model.predict_proba(curve_monthly_scaled)[:, 1]
prob_total = model.predict_proba(curve_total_scaled)[:, 1]

# --------- SUBPLOTS ---------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Tenure plot
axes[0].plot(tenure_range, prob_tenure, color='crimson', linewidth=3)
axes[0].set_title("Tenure vs Churn Probability")
axes[0].set_xlabel("Tenure (Months)")
axes[0].set_ylabel("Probability of Churn")

# Monthly Charges plot
axes[1].plot(monthly_range, prob_monthly, color='seagreen', linewidth=3)
axes[1].set_title("Monthly Charges vs Churn Probability")
axes[1].set_xlabel("Monthly Charges")

# Total Charges plot
axes[2].plot(total_range, prob_total, color='royalblue', linewidth=3)
axes[2].set_title("Total Charges vs Churn Probability")
axes[2].set_xlabel("Total Charges")

plt.tight_layout()
st.pyplot(fig)

st.markdown("</div>", unsafe_allow_html=True)


# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("🧾 Customer Details")

tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 20.0, 120.0, 70.0)
total_charges = st.sidebar.slider("Total Charges ($)", 0.0, 9000.0, 1500.0)

input_data = np.array([[tenure, monthly_charges, total_charges]])
input_scaled = scaler.transform(input_data)

# ---------------- PREDICTION ----------------
if st.sidebar.button("🔍 Predict Churn Probability"):
    prob = model.predict_proba(input_scaled)[0][1]
    prediction = "Churn" if prob >= 0.5 else "Stay"

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")

    st.metric("Churn Probability", f"{prob*100:.2f}%")

    if prediction == "Churn":
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- MODEL PERFORMANCE ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📈 Model Performance")

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)

st.metric("Accuracy", f"{acc*100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

import seaborn as sns
from sklearn.metrics import confusion_matrix

st.subheader("🔢 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(4, 3))   # 👈 small & clean
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    linewidths=0.5,
    ax=ax
)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")
ax.set_xticklabels(["Stay", "Churn"])
ax.set_yticklabels(["Stay", "Churn"], rotation=0)
ax.set_title("Customer Churn Confusion Matrix")

st.pyplot(fig)

st.write("**Confusion Matrix Interpretation:**")
st.write(f"- True Negatives (TN): {tn} → Stayed & correctly predicted")
st.write(f"- False Positives (FP): {fp} → Stayed but predicted as churn")
st.write(f"- False Negatives (FN): {fn} → Churned but predicted as stay")
st.write(f"- True Positives (TP): {tp} → Churned & correctly predicted")

st.markdown("</div>", unsafe_allow_html=True)








from sklearn.metrics import classification_report

st.subheader("📋 Classification Report")

report = classification_report(
    y_test,
    y_pred,
    target_names=["Stay", "Churn"],
    output_dict=True
)

report_df = pd.DataFrame(report).transpose().round(2)

# Convert to centered HTML table
html_table = report_df.to_html(
    classes="centered-table",
    justify="center"
)

st.markdown("""
<style>
.centered-table {
    margin-left: auto;
    margin-right: auto;
    border-collapse: collapse;
    width: 100%;
}
.centered-table th, .centered-table td {
    text-align: center !important;
    padding: 8px;
    border: 1px solid #ddd;
}
.centered-table th {
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown(html_table, unsafe_allow_html=True)





# ---------------- FOOTER ----------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'> Logistic Regression</p>",
    unsafe_allow_html=True
)
