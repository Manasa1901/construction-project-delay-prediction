import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Construction Delay Intelligence",
    page_icon="🏗️",
    layout="wide"
)


# =========================================================
# LOAD MODEL AND DATA
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("construction_delay_random_forest.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("construction_project_delay_final.csv")


model = load_model()
df = load_data()


# =========================================================
# TITLE
# =========================================================

st.title("🏗️ Construction Project Delay Intelligence")

st.markdown(
    """
    **Predict construction delays, identify early warning indicators,
    and understand the factors contributing to project delays.**
    """
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "📊 Dashboard",
        "🤖 Delay Predictor",
        "⚠️ Risk Analysis",
        "💡 Key Insights"
    ]
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "📊 Dashboard":

    st.header("Project Overview")

    total_projects = len(df)

    delayed_projects = (
        df["Delay_Status"] == "Delayed"
    ).sum()

    on_time_projects = (
        df["Delay_Status"] == "On Time"
    ).sum()

    average_delay = df["Progress_Delay"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Projects",
        total_projects
    )

    col2.metric(
        "Delayed Projects",
        delayed_projects
    )

    col3.metric(
        "On-Time Projects",
        on_time_projects
    )

    col4.metric(
        "Average Progress Delay",
        f"{average_delay:.2f}"
    )

    st.divider()

    # -----------------------------------------------------
    # Visualization 1
    # -----------------------------------------------------

    st.subheader("Planned vs Actual Progress")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        df["Planned_Progress"],
        df["Actual_Progress"],
        alpha=0.6
    )

    ax.plot(
        [0, 100],
        [0, 100],
        linestyle="--"
    )

    ax.set_xlabel("Planned Progress")
    ax.set_ylabel("Actual Progress")
    ax.set_title("Planned vs Actual Project Progress")

    st.pyplot(fig)

    # -----------------------------------------------------
    # Visualization 2
    # -----------------------------------------------------

    st.subheader("Average Delay by Milestone")

    milestone_data = (
        df.groupby("Milestone")["Progress_Delay"]
        .mean()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    milestone_data.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Milestone")
    ax.set_ylabel("Average Delay")
    ax.set_title("Average Project Delay by Milestone")

    plt.xticks(rotation=45)

    st.pyplot(fig)

    # -----------------------------------------------------
    # Visualization 3
    # -----------------------------------------------------

    st.subheader("Material Delivery vs Project Delay")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        df["Material_Delivery"],
        df["Progress_Delay"],
        alpha=0.6
    )

    ax.set_xlabel("Material Delivery Delay")
    ax.set_ylabel("Project Progress Delay")
    ax.set_title(
        "Material Delivery Delay vs Project Delay"
    )

    st.pyplot(fig)

    # -----------------------------------------------------
    # Visualization 4
    # -----------------------------------------------------

    st.subheader("Average Delay by Project Category")

    category_data = (
        df.groupby("Project_Category")["Progress_Delay"]
        .mean()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    category_data.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Project Category")
    ax.set_ylabel("Average Delay")
    ax.set_title("Average Delay by Project Category")

    plt.xticks(rotation=45)

    st.pyplot(fig)


# =========================================================
# DELAY PREDICTOR
# =========================================================

elif page == "🤖 Delay Predictor":

    st.header("🤖 Construction Delay Predictor")

    st.write(
        "Enter project information below to predict whether "
        "the project is likely to be delayed."
    )

    col1, col2 = st.columns(2)

    with col1:

        project_category = st.selectbox(
            "Project Category",
            sorted(df["Project_Category"].unique())
        )

        project_size = st.number_input(
            "Project Size",
            min_value=500,
            max_value=50000,
            value=10000
        )

        budget = st.number_input(
            "Budget",
            min_value=0.0,
            value=float(df["Budget"].median())
        )

        workforce = st.number_input(
            "Workforce",
            min_value=1,
            max_value=500,
            value=100
        )

        material_delivery = st.number_input(
            "Material Delivery Delay",
            min_value=0,
            max_value=50,
            value=8
        )

    with col2:

        weather = st.selectbox(
            "Weather",
            sorted(df["Weather"].unique())
        )

        contractor = st.selectbox(
            "Contractor",
            sorted(df["Contractor"].unique())
        )

        milestone = st.selectbox(
            "Milestone",
            sorted(df["Milestone"].unique())
        )

        planned_progress = st.slider(
            "Planned Progress",
            min_value=0,
            max_value=100,
            value=70
        )

    st.divider()

    if st.button(
        "🔍 Predict Project Delay",
        use_container_width=True
    ):

        input_data = pd.DataFrame({
            "Project_Category": [project_category],
            "Project_Size": [project_size],
            "Budget": [budget],
            "Workforce": [workforce],
            "Material_Delivery": [material_delivery],
            "Weather": [weather],
            "Contractor": [contractor],
            "Milestone": [milestone],
            "Planned_Progress": [planned_progress]
        })

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        classes = model.classes_

        probability_dict = dict(
            zip(classes, probabilities)
        )

        delay_probability = (
            probability_dict.get("Delayed", 0) * 100
        )

        st.divider()

        if prediction == "Delayed":

            st.error(
                "🔴 HIGH ALERT: Project is predicted to be DELAYED"
            )

        else:

            st.success(
                "🟢 Project is predicted to remain ON TIME"
            )

        st.metric(
            "Delay Probability",
            f"{delay_probability:.2f}%"
        )

        # Early warning indicators

        warnings = []

        if material_delivery >= 10:
            warnings.append(
                "📦 High material delivery delay"
            )

        if workforce < 100:
            warnings.append(
                "👷 Low workforce availability"
            )

        if weather == "Poor":
            warnings.append(
                "🌧️ Poor weather conditions"
            )

        st.subheader("⚠️ Early Warning Indicators")

        if warnings:

            for warning in warnings:
                st.warning(warning)

        else:

            st.success(
                "No major early warning indicators detected."
            )


# =========================================================
# RISK ANALYSIS
# =========================================================

elif page == "⚠️ Risk Analysis":

    st.header("⚠️ Project Risk Analysis")

    # Risk level if available
    if "Risk_Level" in df.columns:

        risk_counts = (
            df["Risk_Level"]
            .value_counts()
        )

        st.subheader("Risk Distribution")

        st.bar_chart(risk_counts)

    st.subheader("High-Risk Projects")

    high_risk = df[
        df["Risk_Level"] == "High Risk"
    ].copy()

    high_risk = high_risk.sort_values(
        "Progress_Delay",
        ascending=False
    )

    columns_to_show = [
        "Project_ID",
        "Project_Category",
        "Workforce",
        "Material_Delivery",
        "Weather",
        "Milestone",
        "Progress_Delay",
        "Risk_Level"
    ]

    st.dataframe(
        high_risk[columns_to_show].head(20),
        use_container_width=True
    )


# =========================================================
# KEY INSIGHTS
# =========================================================

elif page == "💡 Key Insights":

    st.header("💡 Key Insights")

    avg_material_delay = (
        df.groupby("Material_Delay_Group")
        ["Progress_Delay"]
        .mean()
        .sort_values(ascending=False)
    )

    highest_material_group = (
        avg_material_delay.index[0]
    )

    highest_milestone = (
        df.groupby("Milestone")["Progress_Delay"]
        .mean()
        .idxmax()
    )

    highest_category = (
        df.groupby("Project_Category")["Progress_Delay"]
        .mean()
        .idxmax()
    )

    highest_weather = (
        df.groupby("Weather")["Progress_Delay"]
        .mean()
        .idxmax()
    )

    st.info(
        f"📦 **Material Delivery:** "
        f"{highest_material_group} material-delay group "
        f"shows the highest average project delay in this dataset."
    )

    st.info(
        f"🏗️ **Milestone:** "
        f"{highest_milestone} has the highest average "
        f"progress delay in this dataset."
    )

    st.info(
        f"🏢 **Project Category:** "
        f"{highest_category} projects show the highest "
        f"average delay in this dataset."
    )

    st.info(
        f"🌦️ **Weather:** "
        f"{highest_weather} weather is associated with "
        f"the highest average delay in this dataset."
    )

    st.info(
        "👷 Workforce availability can be monitored as "
        "an early-warning indicator for project delays."
    )

    st.info(
        "📊 Planned versus actual progress provides a "
        "direct indicator of project performance."
    )

    st.info(
        "🤖 The Random Forest model combines project, "
        "workforce, material, weather and milestone "
        "information to predict delay status."
    )