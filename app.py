import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import plotly.express as px

warnings.filterwarnings("ignore")

# 1. Configuration de la page en mode Wide
st.set_page_config(
    page_title="Insurance Cost Predictor & Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

MODEL_PATH = "insurance_model.pkl"
FEATURES_PATH = "feature_columns.pkl"
REF_AVG_NON_SMOKER = 8400
REF_AVG_SMOKER = 32000

@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return model, feature_columns

def get_bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal weight'
    elif 25 <= bmi < 30:
        return 'Overweight'
    else:
        return 'Obesity'

def build_input_row(age, sex, bmi, children, smoker, region, feature_columns):
    bmi_cat = get_bmi_category(bmi)
    row = {
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        "sex_male": 1 if sex == 'male' else 0,
        "smoker_yes": 1 if smoker == 'yes' else 0,
        "region_northwest": 1 if region == 'northwest' else 0,
        "region_southeast": 1 if region == 'southeast' else 0,
        "region_southwest": 1 if region == 'southwest' else 0,
        "bmi_category_Normal weight": 1 if bmi_cat == 'Normal weight' else 0,
        "bmi_category_Overweight": 1 if bmi_cat == 'Overweight' else 0,
        "bmi_category_Obesity": 1 if bmi_cat == 'Obesity' else 0,
    }
    return pd.DataFrame([row], columns=feature_columns).fillna(0)

# Chargement du modèle et des features
model, feature_columns = load_artifacts()

# En-tête de l'application
st.markdown("<div class='main-header'>🏥 Medical Insurance Cost Estimator</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Predict health insurance charges and analyze risk factors using Machine Learning</div>", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model or feature columns file not found. Please ensure 'insurance_model.pkl' and 'feature_columns.pkl' are present.")
    st.stop()

# --- SIDEBAR : Paramètres du patient ---
st.sidebar.header("📋 Patient Parameters")

age = st.sidebar.slider("Age", 18, 70, 30)
sex = st.sidebar.radio("Sex", ["male", "female"], horizontal=True)
bmi = st.sidebar.slider("BMI (IMC)", 10.0, 50.0, 25.0, step=0.5)
children = st.sidebar.slider("Number of Children", 0, 5, 0)
smoker = st.sidebar.radio("Smoker Status", ["yes", "no"], horizontal=True)
region = st.sidebar.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# --- CALCUL DE LA PRÉDICTION EN TEMPS RÉEL ---
input_row = build_input_row(age, sex, bmi, children, smoker, region, feature_columns)
log_prediction = model.predict(input_row)[0]
predicted_cost = float(np.expm1(log_prediction))

ref_avg = REF_AVG_SMOKER if smoker == "yes" else REF_AVG_NON_SMOKER
diff_from_avg = predicted_cost - ref_avg
diff_pct = (diff_from_avg / ref_avg) * 100

# --- DISPOSITION EN ONGLETS ---
tab1, tab2 = st.tabs(["📊 Prediction & Benchmarks", "📈 Age Trajectory & Sensitivity"])

with tab1:
    # Top KPI Metrics Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Predicted Annual Cost",
            value=f"${predicted_cost:,.2f}",
            delta=f"{diff_pct:+.1f}% vs {smoker.capitalize()} Avg",
            delta_color="inverse"
        )
        
    with col2:
        st.metric(
            label=f"National Avg ({'Smoker' if smoker == 'yes' else 'Non-Smoker'})",
            value=f"${ref_avg:,.2f}"
        )
        
    with col3:
        bmi_cat = get_bmi_category(bmi)
        st.metric(
            label="BMI Category",
            value=bmi_cat,
            delta="High Risk" if bmi_cat == 'Obesity' else ("Moderate Risk" if bmi_cat == 'Overweight' else "Normal"),
            delta_color="off"
        )

    st.markdown("---")

    col_chart, col_summary = st.columns([1.3, 1], gap="large")

    with col_chart:
        st.subheader("📊 Cost vs National Benchmarks")
        
        benchmarks_df = pd.DataFrame({
            "Category": ["Non-Smoker Avg", "Your Prediction", "Smoker Avg"],
            "Cost ($)": [REF_AVG_NON_SMOKER, predicted_cost, REF_AVG_SMOKER],
            "Type": ["Benchmark", "Patient", "Benchmark"]
        })

        fig_bar = px.bar(
            benchmarks_df,
            x="Category",
            y="Cost ($)",
            text="Cost ($)",
            color="Category",
            color_discrete_map={
                "Non-Smoker Avg": "#3B82F6",
                "Your Prediction": "#10B981" if smoker == "no" else "#EF4444",
                "Smoker Avg": "#F59E0B"
            }
        )
        fig_bar.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig_bar.update_layout(
            showlegend=False, 
            height=380, 
            yaxis_range=[0, max(predicted_cost, REF_AVG_SMOKER) * 1.25],
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_summary:
        st.subheader("📝 Patient Risk Summary")
        st.markdown(f"""
        - **Age:** {age} years
        - **Gender:** {sex.capitalize()}
        - **BMI:** {bmi:.1f} (`{bmi_cat}`)
        - **Smoker:** `{"Yes" if smoker == "yes" else "No"}`
        - **Children:** {children}
        - **Region:** {region.capitalize()}
        """)

        if smoker == "yes":
            st.warning("⚠️ **Smoker Impact:** Tobacco usage is the single largest factor driving up your premium.")
        else:
            st.success("✅ **Non-Smoker Discount:** Being a non-smoker significantly lowers your predicted insurance costs.")

        if bmi >= 30:
            st.info("💡 **BMI Note:** BMI in the Obesity range increases healthcare risk parameters.")

with tab2:
    st.subheader("📈 Projected Cost Evolution by Age (18 to 70)")
    st.write("This chart simulates how the predicted cost evolves across age groups while keeping all other parameters constant.")

    # Simulation sur la plage d'âge
    age_range = list(range(18, 71, 2))
    simulated_costs = []

    for a in age_range:
        row_a = build_input_row(a, sex, bmi, children, smoker, region, feature_columns)
        log_p_a = model.predict(row_a)[0]
        simulated_costs.append(np.expm1(log_p_a))

    sim_df = pd.DataFrame({"Age": age_range, "Predicted Cost ($)": simulated_costs})

    fig_line = px.line(
        sim_df,
        x="Age",
        y="Predicted Cost ($)",
        markers=True,
        color_discrete_sequence=["#1E3A8A"]
    )
    
    # Marqueur pour l'âge actuel du patient
    fig_line.add_scatter(
        x=[age],
        y=[predicted_cost],
        mode='markers+text',
        marker=dict(size=14, color='#EF4444'),
        text=[f" Current Age ({age})"],
        textposition="top left",
        name="Current Patient"
    )

    fig_line.update_layout(height=420, showlegend=False)
    st.plotly_chart(fig_line, use_container_width=True)