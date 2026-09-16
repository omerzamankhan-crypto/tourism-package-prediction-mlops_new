
import os
import streamlit as st
import pandas as pd
import joblib
from textwrap import dedent




def render_html(content):
    st.markdown(
        dedent(content).strip(),
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ---------------------------------------------------------
# Custom styling
# ---------------------------------------------------------
st.markdown("""
<style>

/* Main page */
.stApp {
    background:
        linear-gradient(135deg, #f4f9ff 0%, #eef9f6 55%, #f9fbff 100%);
    font-family: "Segoe UI", "Inter", Arial, sans-serif;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.4rem;
    padding-bottom: 3rem;
}

/* Hide unnecessary Streamlit elements */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* Hero banner */
.hero-container {
    background: linear-gradient(120deg, #12355b 0%, #176b87 58%, #2a9d8f 100%);
    border-radius: 22px;
    padding: 30px 38px;
    margin-bottom: 25px;
    color: white;
    box-shadow: 0 10px 30px rgba(18, 53, 91, 0.18);
    position: relative;
    overflow: hidden;
}

.hero-container::after {
    content: "✈️  · · ·  📍";
    position: absolute;
    right: 40px;
    top: 34px;
    font-size: 42px;
    opacity: 0.92;
}

.hero-title {
    font-size: 42px;
    font-weight: 750;
    letter-spacing: -0.5px;
    margin-bottom: 5px;
}

.hero-subtitle {
    color: #ddf5f1;
    font-size: 18px;
    margin-bottom: 18px;
}

.hero-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
}

.hero-tag {
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.20);
    border-radius: 20px;
    padding: 7px 13px;
    font-size: 13px;
}

/* Bordered Streamlit containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: rgba(255,255,255,0.94);
    border: 1px solid #dce9ee;
    border-radius: 17px;
    box-shadow: 0 5px 18px rgba(18, 53, 91, 0.07);
}

/* Section headings */
.section-heading {
    color: #12355b;
    font-size: 21px;
    font-weight: 700;
    margin-bottom: 2px;
}

.section-description {
    color: #708090;
    font-size: 13px;
    margin-bottom: 15px;
}

/* Widget labels */
.stSelectbox label,
.stSlider label,
.stNumberInput label {
    color: #24384b !important;
    font-weight: 600 !important;
}

/* Select boxes */
div[data-baseweb="select"] > div {
    background-color: #f7fafc;
    border-color: #d6e2e8;
    border-radius: 9px;
}

/* Number input */
.stNumberInput input {
    background-color: #f7fafc;
}

/* Slider colour */
.stSlider div[data-baseweb="slider"] div {
    color: #ff5a5f;
}

/* Prediction button */
.stButton > button {
    width: 100%;
    min-height: 52px;
    background: linear-gradient(90deg, #ff5a5f, #f04455);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 20px;
    font-size: 17px;
    font-weight: 700;
    box-shadow: 0 7px 18px rgba(240, 68, 85, 0.22);
}

.stButton > button:hover {
    background: linear-gradient(90deg, #f04455, #d93647);
    color: white;
    border: none;
    transform: translateY(-1px);
}

/* Result placeholders and cards */
.result-placeholder {
    background: linear-gradient(145deg, #edf8f6, #f7fbff);
    border: 1px dashed #9dc8c0;
    border-radius: 14px;
    padding: 25px 18px;
    text-align: center;
    color: #547079;
    margin-top: 12px;
}

.result-card {
    background: linear-gradient(145deg, #e7f8ef, #f5fff9);
    border-left: 6px solid #19a974;
    border-radius: 15px;
    padding: 22px;
    margin: 15px 0;
    box-shadow: 0 5px 16px rgba(25, 169, 116, 0.10);
}

.result-label {
    color: #44645b;
    font-size: 15px;
    font-weight: 600;
}

.result-value {
    color: #09875e;
    font-size: 55px;
    line-height: 1.1;
    font-weight: 800;
    margin: 7px 0;
}

.result-message {
    color: #086b4b;
    font-size: 16px;
    font-weight: 650;
    margin-top: 12px;
}

.low-result-card {
    background: linear-gradient(145deg, #eef5ff, #f8fbff);
    border-left: 6px solid #3b82c4;
    border-radius: 15px;
    padding: 22px;
    margin: 15px 0;
}

.low-result-value {
    color: #24699c;
    font-size: 55px;
    line-height: 1.1;
    font-weight: 800;
    margin: 7px 0;
}

.insight-card {
    background-color: #edf6ff;
    border-radius: 13px;
    padding: 17px;
    color: #234f73;
    margin-top: 14px;
}

.disclaimer {
    color: #7d8993;
    font-size: 12px;
    margin-top: 16px;
    text-align: center;
}

/* Responsive hero */
@media (max-width: 800px) {
    .hero-title {
        font-size: 31px;
    }

    .hero-container::after {
        display: none;
    }
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Load the trained model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(
        os.path.dirname(__file__),
        "tourism_model.joblib"
    )
    return joblib.load(model_path)


model = load_model()


# ---------------------------------------------------------
# Hero section
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Tourism Package Prediction</div>
    <div class="hero-subtitle">
        AI-powered customer purchase likelihood assessment
    </div>

    <div class="hero-tags">
        <span class="hero-tag">📊 Smarter insights</span>
        <span class="hero-tag">👥 Better customer targeting</span>
        <span class="hero-tag">✈️ More successful journeys</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Main layout
# ---------------------------------------------------------
form_area, result_area = st.columns([2.15, 1], gap="large")


with form_area:

    profile_col, travel_col = st.columns(2, gap="medium")

    # -----------------------------------------------------
    # Customer profile
    # -----------------------------------------------------
    with profile_col:
        with st.container(border=True):

            st.markdown(
                '<div class="section-heading">👤 Customer Profile</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-description">'
                'Basic demographic and professional information'
                '</div>',
                unsafe_allow_html=True
            )

            Age = st.slider(
                "Age",
                min_value=18,
                max_value=61,
                value=30
            )

            Gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            Occupation = st.selectbox(
                "Occupation",
                [
                    "Salaried",
                    "Small Business",
                    "Large Business",
                    "Free Lancer"
                ]
            )

            MaritalStatus = st.selectbox(
                "Marital Status",
                [
                    "Married",
                    "Single",
                    "Divorced",
                    "Unmarried"
                ]
            )

            Designation = st.selectbox(
                "Designation",
                [
                    "Executive",
                    "Manager",
                    "Senior Manager",
                    "AVP",
                    "VP"
                ]
            )

    # -----------------------------------------------------
    # Travel preferences
    # -----------------------------------------------------
    with travel_col:
        with st.container(border=True):

            st.markdown(
                '<div class="section-heading">🌍 Travel Preferences</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-description">'
                'Customer travel and package preferences'
                '</div>',
                unsafe_allow_html=True
            )

            CityTier = st.selectbox(
                "City Tier",
                [1, 2, 3]
            )

            NumberOfTrips = st.slider(
                "Number of Trips",
                min_value=1,
                max_value=22,
                value=3
            )

            NumberOfPersonVisiting = st.slider(
                "Number of Persons Visiting",
                min_value=1,
                max_value=5,
                value=2
            )

            ProductPitched = st.selectbox(
                "Product Pitched",
                [
                    "Basic",
                    "Standard",
                    "Deluxe",
                    "Super Deluxe",
                    "King"
                ]
            )

            PreferredPropertyStar = st.selectbox(
                "Preferred Property Star",
                [3, 4, 5]
            )

    sales_col, financial_col = st.columns(2, gap="medium")

    # -----------------------------------------------------
    # Sales interaction
    # -----------------------------------------------------
    with sales_col:
        with st.container(border=True):

            st.markdown(
                '<div class="section-heading">📞 Sales Interaction</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-description">'
                'Information collected during the sales process'
                '</div>',
                unsafe_allow_html=True
            )

            TypeofContact = st.selectbox(
                "Type of Contact",
                ["Self Enquiry", "Company Invited"]
            )

            DurationOfPitch = st.slider(
                "Duration of Pitch (minutes)",
                min_value=5,
                max_value=127,
                value=15
            )

            NumberOfFollowups = st.slider(
                "Number of Follow-ups",
                min_value=1,
                max_value=6,
                value=3
            )

            PitchSatisfactionScore = st.slider(
                "Pitch Satisfaction Score",
                min_value=1,
                max_value=5,
                value=3
            )

    # -----------------------------------------------------
    # Additional and financial details
    # -----------------------------------------------------
    with financial_col:
        with st.container(border=True):

            st.markdown(
                '<div class="section-heading">💳 Additional Details</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-description">'
                'Financial and supporting customer information'
                '</div>',
                unsafe_allow_html=True
            )

            Passport = st.selectbox(
                "Has Passport?",
                ["Yes", "No"]
            )

            OwnCar = st.selectbox(
                "Owns a Car?",
                ["Yes", "No"]
            )

            NumberOfChildrenVisiting = st.slider(
                "Number of Children Visiting",
                min_value=0,
                max_value=3,
                value=1
            )

            MonthlyIncome = st.number_input(
                "Monthly Income",
                min_value=1000.0,
                max_value=98678.0,
                value=30000.0,
                step=1000.0
            )


# ---------------------------------------------------------
# Construct the model input DataFrame
# Feature names are unchanged from the original application.
# ---------------------------------------------------------
input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome
}])


classification_threshold = 0.45


# ---------------------------------------------------------
# Prediction panel
# ---------------------------------------------------------
with result_area:
    with st.container(border=True):

        st.markdown(
            '<div class="section-heading">📊 Prediction Result</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-description">'
            'Estimate the customer’s likelihood of purchasing the package'
            '</div>',
            unsafe_allow_html=True
        )

        predict_button = st.button(
            "✨ Predict Purchase Likelihood",
            type="primary"
        )

        if predict_button:

            probability = float(
                model.predict_proba(input_data)[0, 1]
            )

            prediction = int(
                probability >= classification_threshold
            )

            if prediction == 1:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">
                            Estimated Purchase Probability
                        </div>

                        <div class="result-value">
                            {probability:.1%}
                        </div>

                        <div class="result-message">
                            ✓ Likely buyer — prioritize this customer
                            for marketing contact.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(probability)

                st.markdown(
                    """
                    <div class="insight-card">
                        <strong>💡 Marketing insight</strong><br><br>
                        Consider contacting this customer with a
                        personalized tourism package and timely follow-up.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:
                st.markdown(
                    f"""
                    <div class="low-result-card">
                        <div class="result-label">
                            Estimated Purchase Probability
                        </div>

                        <div class="low-result-value">
                            {probability:.1%}
                        </div>

                        <div style="color:#245e89; font-weight:650;">
                            Lower purchase likelihood — use a
                            lower-priority or personalized contact strategy.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(probability)

                st.markdown(
                    """
                    <div class="insight-card">
                        <strong>💡 Marketing insight</strong><br><br>
                        Consider a targeted discount or alternative package
                        before adding this customer to a priority campaign.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:
            st.markdown(
                """
                <div class="result-placeholder">
                    <div style="font-size:42px;">📈</div>
                    <strong>Prediction awaiting input</strong><br><br>
                    Review the customer details and click the prediction
                    button to generate the purchase probability.
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            """
            <div class="disclaimer">
                This prediction supports marketing decisions and should
                not be treated as certainty.
            </div>
            """,
            unsafe_allow_html=True
        )
