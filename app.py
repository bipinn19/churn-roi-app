import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn & ROI Simulator", layout="wide")
st.title("📊 Customer Churn & ROI Simulator")

# --- MODEL LOADING ---
@st.cache_resource
def load_model_and_explainer():
    model = joblib.load('churn_pipeline.pkl')
    estimator = model.named_steps['classifier']
    explnr = shap.TreeExplainer(estimator)
    return model, explnr

model, explainer = load_model_and_explainer()

# --- SIDEBAR ---
st.sidebar.header("⚙️ ROI Simulator Settings")
churn_threshold = st.sidebar.slider("Risk Probability Threshold", 0.0, 1.0, 0.50, help="Customers above this risk probability are flagged as churners.")
clv = st.sidebar.number_input("Customer Lifetime Value ($)", value=1000, help="Expected profit from a retained customer.")
retention_cost = st.sidebar.number_input("Retention Campaign Cost ($)", value=50, help="Cost to run the intervention (e.g., discount).")
acceptance_rate = st.sidebar.slider("Offer Acceptance Rate (%)", 0, 100, 30, help="Percentage of targeted customers who accept the offer.")

tab1, tab2 = st.tabs(["👤 Single Customer Prediction", "📁 Enterprise Batch Prediction"])

# --- TAB 1: SINGLE CUSTOMER ---
with tab1:
    st.subheader("👤 Predict Customer Churn & Calculate ROI")
    
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)
        with col2:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
        with col3:
            # FIX 1: Removed max_value cap to prevent crashes when tenure * monthly_charges > 10000
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=float(tenure * monthly_charges))
            
        col4, col5 = st.columns(2)
        with col4:
            contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        with col5:
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            # FIX 2: Correctly handle "No internet service" dependencies across all related features
            if internet_service == "No":
                tech_support = online_security = online_backup = device_protection = streaming_tv = streaming_movies = "No internet service"
            else:
                tech_support = st.selectbox("Tech Support", ["Yes", "No"])
                online_security = online_backup = device_protection = streaming_tv = streaming_movies = "No" # Simplified defaults for UI

    if st.button("🔮 Predict Churn & Calculate ROI"):
        input_dict = {
            'tenure': tenure, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges,
            'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
            'PhoneService': 'Yes', 'MultipleLines': 'No', 'InternetService': internet_service,
            'OnlineSecurity': online_security, 'OnlineBackup': online_backup, 
            'DeviceProtection': device_protection, 'TechSupport': tech_support,
            'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
            'Contract': contract_type, 'PaperlessBilling': 'Yes', 'PaymentMethod': payment_method
        }
        
        input_data = pd.DataFrame([input_dict])
        probability = float(model.predict_proba(input_data)[0][1])
        prediction = 1 if probability >= churn_threshold else 0
        expected_roi = (probability * (acceptance_rate / 100.0) * clv) - retention_cost

        st.write("---")
        if prediction == 1:
            st.error(f"⚠️ **High Risk of Churn!** (Probability: {probability:.0%})")
        else:
            st.success(f"✅ **Below Risk Threshold.** (Churn Probability: {probability:.0%})")
            
        st.subheader("💡 Business ROI Strategy:")
        col_roi1, col_roi2 = st.columns(2)
        with col_roi1:
            # FIX 3: Always show the ROI metric, even for safe customers, removing the contradiction
            st.metric(label="Expected Net Value per Customer", value=f"${expected_roi:.2f}")
        with col_roi2:
            if expected_roi > 0:
                st.success("✅ Profitable to intervene based on current threshold.")
            else:
                st.warning("❌ Not worth intervening. Cost exceeds expected value.")

        st.write("---")
        st.subheader("🧠 Why is this happening? (Signed SHAP Waterfall)")
        with st.spinner("Generating local explanation..."):
            processed_data = model.named_steps['preprocessor'].transform(input_data)
            shap_values = explainer(processed_data)
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            st.pyplot(fig)
            plt.close(fig)

# --- TAB 2: BATCH PREDICTION ---
with tab2:
    st.subheader("📁 Enterprise Batch Prediction & Campaign ROI")
    
    with st.container(border=True):
        st.write("Upload a CSV file containing customer data to score your entire user base.")
        uploaded_file = st.file_uploader("Upload Customer CSV", type="csv")
        
    if uploaded_file is not None:
        batch_data = pd.read_csv(uploaded_file)
        if st.button("🔮 Run Batch Predictions"):
            with st.spinner("Analyzing all customers..."):
                batch_clean = batch_data.copy()
                if 'customerID' in batch_clean.columns: batch_clean = batch_clean.drop('customerID', axis=1)
                if 'Churn' in batch_clean.columns: batch_clean = batch_clean.drop('Churn', axis=1)

                batch_probs = model.predict_proba(batch_clean)[:, 1]
                batch_predictions = (batch_probs >= churn_threshold).astype(int)
                
                results_df = batch_data.copy()
                results_df['Churn_Prediction'] = ['Yes' if p == 1 else 'No' for p in batch_predictions]
                results_df['Churn_Risk_Probability'] = batch_probs
                results_df['Expected_Value'] = (batch_probs * (acceptance_rate / 100.0) * clv) - retention_cost
                
                st.success("✅ Batch Prediction Complete!")
                
                # FIX 4: Only sum expected values that are strictly profitable for flagged churners
                worth_targeting = (results_df['Churn_Prediction'] == 'Yes') & (results_df['Expected_Value'] > 0)
                total_expected_net_roi = float(results_df.loc[worth_targeting, 'Expected_Value'].sum())
                total_churners = int((results_df['Churn_Prediction'] == 'Yes').sum())
                revenue_at_risk = float(batch_data.loc[results_df['Churn_Prediction'] == 'Yes', 'MonthlyCharges'].sum())

                colA, colB, colC = st.columns(3)
                colA.metric("Customers at Risk", total_churners)
                colB.metric("Monthly Revenue at Risk", f"${revenue_at_risk:,.2f}")
                colC.metric("Est. Total Campaign Net ROI", f"${total_expected_net_roi:,.2f}")
                
                display_cols = ['customerID', 'Churn_Prediction', 'Churn_Risk_Probability', 'Expected_Value'] if 'customerID' in results_df.columns else results_df.columns
                st.dataframe(results_df[display_cols])