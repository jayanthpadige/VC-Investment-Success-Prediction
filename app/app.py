import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="VC Investment Success Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "vc_success_prediction_model.pkl"

FEATURE_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "feature_importance.csv"
)

COMPARISON_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "model_comparison.csv"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #9ca3af;
        margin-bottom: 25px;
    }


    /* Sidebar */

    [data-testid="stSidebar"] {
        background-color: #151821;
    }

    [data-testid="stSidebar"] .stButton button {
        width: 100%;
        min-height: 45px;
        border-radius: 10px;
        font-weight: 600;
        text-align: left;
        margin-bottom: 7px;
    }


    /* Result boxes */

    .success-box {
        background-color: #123b29;
        border: 1px solid #1f8f5f;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    .success-box h2 {
        color: #4ade80;
        margin-bottom: 8px;
    }

    .success-box h1 {
        color: white;
        font-size: 48px;
        margin: 5px;
    }

    .success-box p {
        color: #bbf7d0;
    }


    .failure-box {
        background-color: #421b1b;
        border: 1px solid #b94a48;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    .failure-box h2 {
        color: #f87171;
        margin-bottom: 8px;
    }

    .failure-box h1 {
        color: white;
        font-size: 48px;
        margin: 5px;
    }

    .failure-box p {
        color: #fecaca;
    }


    /* Footer */

    .footer {
        text-align: center;
        color: #6b7280;
        padding-top: 35px;
        padding-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


try:

    model = load_model()

except Exception as e:

    st.error("Unable to load the machine learning model.")

    st.code(str(e))

    st.stop()


# ============================================================
# LOAD FEATURE IMPORTANCE
# ============================================================

@st.cache_data
def load_feature_importance():

    if FEATURE_PATH.exists():

        return pd.read_csv(FEATURE_PATH)

    return pd.DataFrame()


# ============================================================
# LOAD MODEL COMPARISON
# ============================================================

@st.cache_data
def load_model_comparison():

    if COMPARISON_PATH.exists():

        return pd.read_csv(COMPARISON_PATH)

    return pd.DataFrame()


feature_df = load_feature_importance()

comparison_df = load_model_comparison()


# ============================================================
# GET CATEGORY OPTIONS FROM MODEL
# ============================================================

def get_category_options(column_name, default_values):

    try:

        preprocessor = model.named_steps["preprocessor"]

        for transformer_name, transformer, columns in preprocessor.transformers_:

            if transformer_name == "cat":

                encoder = transformer.named_steps["onehot"]

                for index, column in enumerate(columns):

                    if column == column_name:

                        return list(
                            encoder.categories_[index]
                        )

    except Exception:

        pass

    return default_values


state_options = get_category_options(
    "state_code",
    ["CA"]
)

category_options = get_category_options(
    "category_code",
    ["music"]
)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:

    st.session_state.page = "Prediction"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🚀 VC Predictor")

st.sidebar.caption("NAVIGATION")


# ------------------------------------------------------------
# Prediction Button
# ------------------------------------------------------------

if st.sidebar.button(
    "🔮  Prediction",
    use_container_width=True,
    type=(
        "primary"
        if st.session_state.page == "Prediction"
        else "secondary"
    )
):

    st.session_state.page = "Prediction"

    st.rerun()


# ------------------------------------------------------------
# Model Information Button
# ------------------------------------------------------------

if st.sidebar.button(
    "📊  Model Information",
    use_container_width=True,
    type=(
        "primary"
        if st.session_state.page == "Model Information"
        else "secondary"
    )
):

    st.session_state.page = "Model Information"

    st.rerun()


# ------------------------------------------------------------
# Feature Importance Button
# ------------------------------------------------------------

if st.sidebar.button(
    "⭐  Feature Importance",
    use_container_width=True,
    type=(
        "primary"
        if st.session_state.page == "Feature Importance"
        else "secondary"
    )
):

    st.session_state.page = "Feature Importance"

    st.rerun()


# ------------------------------------------------------------
# About Project Button
# ------------------------------------------------------------

if st.sidebar.button(
    "ℹ️  About Project",
    use_container_width=True,
    type=(
        "primary"
        if st.session_state.page == "About Project"
        else "secondary"
    )
):

    st.session_state.page = "About Project"

    st.rerun()


# ------------------------------------------------------------
# Sidebar Project Information
# ------------------------------------------------------------

st.sidebar.divider()

st.sidebar.subheader("📊 Project Overview")

st.sidebar.info(
    """
**VC Investment Success Predictor**

Machine Learning application for predicting startup investment outcomes.

**🤖 Model:** Gradient Boosting

**📁 Dataset:** Startup Investment Dataset

**🎯 Accuracy:** 83.24%

**📈 ROC-AUC:** 86.57%
"""
)


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '📊 VC Investment Success Predictor'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict whether a startup is likely to be '
    'Acquired / Successful based on its funding, '
    'milestones, relationships and characteristics.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# PAGE 1 — PREDICTION
# ============================================================

if st.session_state.page == "Prediction":

    st.header("🔮 Startup Success Prediction")

    st.info(
        "Enter the startup details below and click "
        "**Predict Investment Success**."
    )


    # --------------------------------------------------------
    # STARTUP INFORMATION
    # --------------------------------------------------------

    st.subheader("🏢 Startup Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        state_code = st.selectbox(
            "State",
            state_options
        )

    with col2:

        category_code = st.selectbox(
            "Startup Category",
            category_options
        )

    with col3:

        is_top500 = st.selectbox(
            "Top 500 Startup?",
            ["No", "Yes"]
        )


    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    st.subheader("📍 Location")

    col1, col2 = st.columns(2)

    with col1:

        latitude = st.number_input(
            "Latitude",
            value=37.7749,
            format="%.6f"
        )

    with col2:

        longitude = st.number_input(
            "Longitude",
            value=-122.4194,
            format="%.6f"
        )


    # --------------------------------------------------------
    # FUNDING AND AGE
    # --------------------------------------------------------

    st.subheader("💰 Funding & Startup Age")

    col1, col2, col3 = st.columns(3)

    with col1:

        age_first_funding_year = st.number_input(
            "Age at First Funding (Years)",
            min_value=0.0,
            value=1.0,
            step=0.1
        )

    with col2:

        age_last_funding_year = st.number_input(
            "Age at Last Funding (Years)",
            min_value=0.0,
            value=5.0,
            step=0.1
        )

    with col3:

        funding_rounds = st.number_input(
            "Funding Rounds",
            min_value=0,
            value=3,
            step=1
        )


    col1, col2 = st.columns(2)

    with col1:

        funding_total_usd = st.number_input(
            "Total Funding (USD)",
            min_value=0.0,
            value=1000000.0,
            step=100000.0
        )

    with col2:

        avg_participants = st.number_input(
            "Average Participants",
            min_value=0.0,
            value=3.0,
            step=0.1
        )


    # --------------------------------------------------------
    # MILESTONES
    # --------------------------------------------------------

    st.subheader("🏆 Milestones")

    col1, col2, col3 = st.columns(3)

    with col1:

        age_first_milestone_year = st.number_input(
            "Age at First Milestone",
            min_value=0.0,
            value=2.0,
            step=0.1
        )

    with col2:

        age_last_milestone_year = st.number_input(
            "Age at Last Milestone",
            min_value=0.0,
            value=4.0,
            step=0.1
        )

    with col3:

        milestones = st.number_input(
            "Number of Milestones",
            min_value=0,
            value=3,
            step=1
        )


    # --------------------------------------------------------
    # RELATIONSHIPS
    # --------------------------------------------------------

    st.subheader("🤝 Relationships")

    relationships = st.number_input(
        "Number of Relationships",
        min_value=0,
        value=5,
        step=1
    )


    # --------------------------------------------------------
    # INVESTMENT SOURCES
    # --------------------------------------------------------

    st.subheader("💼 Investment Sources")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        has_VC = st.selectbox(
            "Has VC",
            ["No", "Yes"]
        )

    with col2:

        has_angel = st.selectbox(
            "Has Angel",
            ["No", "Yes"]
        )

    with col3:

        has_roundA = st.selectbox(
            "Round A",
            ["No", "Yes"]
        )

    with col4:

        has_roundB = st.selectbox(
            "Round B",
            ["No", "Yes"]
        )


    col1, col2 = st.columns(2)

    with col1:

        has_roundC = st.selectbox(
            "Round C",
            ["No", "Yes"]
        )

    with col2:

        has_roundD = st.selectbox(
            "Round D",
            ["No", "Yes"]
        )


    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    st.divider()

    predict_button = st.button(
        "🚀  Predict Investment Success",
        use_container_width=True,
        type="primary"
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if predict_button:

        input_data = pd.DataFrame(
            [
                {
                    "state_code": state_code,

                    "latitude": latitude,

                    "longitude": longitude,

                    "age_first_funding_year":
                        age_first_funding_year,

                    "age_last_funding_year":
                        age_last_funding_year,

                    "age_first_milestone_year":
                        age_first_milestone_year,

                    "age_last_milestone_year":
                        age_last_milestone_year,

                    "relationships":
                        relationships,

                    "funding_rounds":
                        funding_rounds,

                    "funding_total_usd":
                        funding_total_usd,

                    "milestones":
                        milestones,

                    "category_code":
                        category_code,

                    "has_VC":
                        1 if has_VC == "Yes" else 0,

                    "has_angel":
                        1 if has_angel == "Yes" else 0,

                    "has_roundA":
                        1 if has_roundA == "Yes" else 0,

                    "has_roundB":
                        1 if has_roundB == "Yes" else 0,

                    "has_roundC":
                        1 if has_roundC == "Yes" else 0,

                    "has_roundD":
                        1 if has_roundD == "Yes" else 0,

                    "avg_participants":
                        avg_participants,

                    "is_top500":
                        1 if is_top500 == "Yes" else 0
                }
            ]
        )


        try:

            prediction = model.predict(
                input_data
            )[0]

            probabilities = model.predict_proba(
                input_data
            )[0]

            closed_probability = probabilities[0] * 100

            success_probability = probabilities[1] * 100


            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.divider()

            st.header("🎯 Prediction Result")


            if prediction == 1:

                st.markdown(
                    f"""
                    <div class="success-box">

                    <h2>✅ ACQUIRED / SUCCESSFUL</h2>

                    <h1>
                    {success_probability:.2f}%
                    </h1>

                    <p>
                    Predicted probability of startup success
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="failure-box">

                    <h2>⚠️ CLOSED / UNSUCCESSFUL</h2>

                    <h1>
                    {closed_probability:.2f}%
                    </h1>

                    <p>
                    Predicted probability of startup closure
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # ------------------------------------------------
            # PROBABILITIES
            # ------------------------------------------------

            st.subheader("📈 Prediction Probabilities")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Acquired / Successful",
                    f"{success_probability:.2f}%"
                )

            with col2:

                st.metric(
                    "Closed / Unsuccessful",
                    f"{closed_probability:.2f}%"
                )


            st.progress(
                min(
                    max(
                        success_probability / 100,
                        0
                    ),
                    1
                )
            )


            # ------------------------------------------------
            # INPUT SUMMARY
            # ------------------------------------------------

            st.subheader("📋 Input Summary")

            display_data = input_data.T.reset_index()

            display_data.columns = [
                "Feature",
                "Value"
            ]

            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True
            )


        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.code(
                str(e)
            )


# ============================================================
# PAGE 2 — MODEL INFORMATION
# ============================================================

elif st.session_state.page == "Model Information":

    st.header("📊 Model Information")

    st.write(
        """
        This project compares multiple machine-learning
        classification algorithms for predicting startup
        investment outcomes.
        """
    )


    # --------------------------------------------------------
    # PROJECT METRICS
    # --------------------------------------------------------

    st.subheader("📌 Project Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            label="📁 Dataset Rows",
            value="923"
        )

    with col2:

        st.metric(
            label="🔢 Input Features",
            value="20"
        )

    with col3:

        st.metric(
            label="🎯 Best Accuracy",
            value="83.24%"
        )

    with col4:

        st.metric(
            label="📈 ROC-AUC",
            value="86.57%"
        )


    st.divider()


    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    st.subheader("🤖 Model Comparison")

    if not comparison_df.empty:

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "Model comparison file was not found."
        )


    st.divider()


    # --------------------------------------------------------
    # SELECTED MODEL
    # --------------------------------------------------------

    st.subheader("🧠 Selected Model")

    st.success(
        "Gradient Boosting Classifier"
    )

    st.write(
        """
        Gradient Boosting was selected as the model used
        in this Streamlit application.
        """
    )


    st.subheader("📈 Test Performance")

    performance_data = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC"
            ],

            "Score": [
                "83.24%",
                "83.46%",
                "92.50%",
                "87.75%",
                "86.57%"
            ]
        }
    )

    st.dataframe(
        performance_data,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 3 — FEATURE IMPORTANCE
# ============================================================

elif st.session_state.page == "Feature Importance":

    st.header("⭐ Feature Importance")

    st.write(
        """
        These features contributed most strongly to the
        Gradient Boosting model's predictions.
        """
    )


    if not feature_df.empty:

        st.subheader("🏆 Top Important Features")

        feature_col = feature_df.columns[0]

        importance_col = feature_df.columns[-1]


        chart_df = feature_df.copy()


        chart_df[importance_col] = pd.to_numeric(
            chart_df[importance_col],
            errors="coerce"
        )


        chart_df = chart_df.dropna(
            subset=[importance_col]
        )


        chart_df = chart_df.sort_values(
            importance_col,
            ascending=False
        ).head(15)


        # Chart

        st.bar_chart(
            chart_df.set_index(
                feature_col
            )[importance_col]
        )


        # Table

        st.subheader("📋 Feature Importance Table")

        st.dataframe(
            chart_df,
            use_container_width=True,
            hide_index=True
        )


    else:

        st.warning(
            "Feature importance file was not found."
        )


# ============================================================
# PAGE 4 — ABOUT PROJECT
# ============================================================

elif st.session_state.page == "About Project":

    st.header("ℹ️ About the Project")


    # --------------------------------------------------------
    # PROBLEM STATEMENT
    # --------------------------------------------------------

    st.subheader("🎯 Problem Statement")

    st.write(
        """
        Venture capital investment involves uncertainty when
        evaluating startup companies.

        This project uses machine learning to predict whether
        a startup is likely to achieve a successful outcome
        based on historical startup characteristics.
        """
    )


    # --------------------------------------------------------
    # OBJECTIVE
    # --------------------------------------------------------

    st.subheader("💡 Objective")

    st.write(
        """
        Build a machine-learning classification system that
        analyzes startup funding, milestones, relationships,
        location and investment characteristics to predict
        startup success.
        """
    )


    # --------------------------------------------------------
    # MACHINE LEARNING WORKFLOW
    # --------------------------------------------------------

    st.subheader("🔄 Machine Learning Workflow")

    workflow_data = pd.DataFrame(
        {
            "Step": [
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7"
            ],

            "Process": [
                "Data Collection",
                "Data Cleaning",
                "Feature Engineering",
                "Preprocessing",
                "Model Training",
                "Model Evaluation",
                "Deployment"
            ],

            "Description": [
                "Historical startup investment data was collected.",
                "Missing values and unnecessary columns were handled.",
                "Numerical and categorical features were prepared.",
                "Numerical scaling and categorical encoding were performed.",
                "Multiple classification algorithms were trained.",
                "Models were evaluated using classification metrics.",
                "The trained model was integrated into Streamlit."
            ]
        }
    )

    st.dataframe(
        workflow_data,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # TECHNOLOGIES
    # --------------------------------------------------------

    st.subheader("🛠️ Technologies Used")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            """
            **🐍 Programming**

            • Python

            • Pandas

            • NumPy
            """
        )

    with col2:

        st.info(
            """
            **🤖 Machine Learning**

            • Scikit-learn

            • Gradient Boosting

            • Model Evaluation
            """
        )

    with col3:

        st.info(
            """
            **🖥️ Application**

            • Streamlit

            • Joblib

            • VS Code
            """
        )


    # --------------------------------------------------------
    # PROJECT SUMMARY
    # --------------------------------------------------------

    st.divider()

    st.subheader("📌 Project Summary")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "📊 Dataset Rows",
            "923"
        )

        st.metric(
            "🔢 Input Features",
            "20"
        )


    with col2:

        st.metric(
            "🎯 Model Accuracy",
            "83.24%"
        )

        st.metric(
            "📈 ROC-AUC",
            "86.57%"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "VC Investment Success Prediction • "
    "Machine Learning Project"
)