
import streamlit as st
import pandas as pd
import joblib

from pathlib import Path
from datetime import date
from recommendation import recommend_donation_centers


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Food Waste Prediction System",
    page_icon="🍱",
    layout="wide"
)


# ==========================================
# 2. LOAD MODEL AND DATASET
# ==========================================

MODEL_PATH = Path("model/food_waste_model.pkl")
DATA_PATH = Path("data/cleaned_train.csv")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


try:
    model = load_model()
    df = load_dataset()

except FileNotFoundError as e:
    st.error(f"Required file not found: {e}")
    st.stop()

except Exception as e:
    st.error(f"Loading error: {e}")
    st.stop()


# ==========================================
# 3. PREPARE DATA
# ==========================================

df["food_waste_kg"] = pd.to_numeric(
    df["food_waste_kg"],
    errors="coerce"
)

if "meals_served" in df.columns:
    df["meals_served"] = pd.to_numeric(
        df["meals_served"],
        errors="coerce"
    )


# ==========================================
# 4. APPLICATION HEADER
# ==========================================

st.title("🍱 AI-Based Food Waste Prediction")

st.subheader(
    "Smart Food Waste Management and Donation Recommendation System"
)

st.write(
    "An intelligent machine learning application that predicts "
    "food waste and recommends suitable donation centers."
)

st.divider()


# ==========================================
# 5. FOOD WASTE ANALYTICS DASHBOARD
# ==========================================

st.header("📊 Food Waste Analytics Dashboard")

total_waste = df["food_waste_kg"].sum()

average_waste = df["food_waste_kg"].mean()

total_records = df["food_waste_kg"].count()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Food Waste",
    f"{total_waste:.2f} kg"
)

col2.metric(
    "Average Food Waste",
    f"{average_waste:.2f} kg"
)

col3.metric(
    "Total Records",
    f"{total_records:,}"
)

if "meals_served" in df.columns:

    st.metric(
        "Total Meals Served",
        f"{df['meals_served'].sum():,.0f}"
    )


# ==========================================
# 6. FOOD WASTE BY CATEGORY
# ==========================================

st.subheader("🥗 Food Waste by Category")

if "waste_category" in df.columns:

    category_waste = (
        df.groupby("waste_category")["food_waste_kg"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(category_waste)

else:

    st.info("Food category column is not available.")


# ==========================================
# 7. FOOD WASTE TREND
# ==========================================

st.subheader("📈 Food Waste Trend Over Time")

if "date" in df.columns:

    chart_df = df.copy()

    chart_df["date"] = pd.to_datetime(
        chart_df["date"],
        errors="coerce"
    )

    daily_waste = (
        chart_df.dropna(subset=["date"])
        .groupby("date")["food_waste_kg"]
        .sum()
        .sort_index()
    )

    if not daily_waste.empty:

        st.line_chart(daily_waste)

    else:

        st.info("No valid dates available for the trend chart.")

else:

    st.info("Date column is not available.")


st.divider()


# ==========================================
# 8. GET DATASET OPTIONS
# ==========================================

def get_options(column, defaults):

    if column in df.columns:

        values = (
            df[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if values:
            return sorted(values)

    return defaults


# ==========================================
# 9. FOOD WASTE PREDICTION FORM
# ==========================================

st.header("🔍 Predict Food Waste")

st.write(
    "Enter the following kitchen details to estimate "
    "the expected food waste."
)

with st.form("food_waste_form"):

    col1, col2 = st.columns(2)

    with col1:

        meals_served = st.number_input(
            "Meals Served",
            min_value=0,
            value=200
        )

        kitchen_staff = st.number_input(
            "Kitchen Staff",
            min_value=0,
            value=10
        )

        temperature_C = st.number_input(
            "Temperature (°C)",
            value=25.0
        )

        humidity_percent = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0
        )

        past_waste_kg = st.number_input(
            "Past Food Waste (kg)",
            min_value=0.0,
            value=10.0
        )

    with col2:

        selected_date = st.date_input(
            "Date",
            value=date.today()
        )

        day_of_week = st.selectbox(
            "Day of Week",
            get_options(
                "day_of_week",
                [
                    "Monday", "Tuesday", "Wednesday",
                    "Thursday", "Friday", "Saturday", "Sunday"
                ]
            )
        )

        special_event = st.selectbox(
            "Special Event",
            get_options(
                "special_event",
                ["No", "Yes"]
            )
        )

        staff_experience = st.selectbox(
            "Staff Experience",
            get_options(
                "staff_experience",
                ["Beginner", "Intermediate", "Experienced"]
            )
        )

        waste_category = st.selectbox(
            "Food Category",
            get_options(
                "waste_category",
                ["Vegetables", "Fruits", "Dairy", "Meat"]
            )
        )

    submitted = st.form_submit_button(
        "Predict Food Waste",
        use_container_width=True
    )


# ==========================================
# 10. MACHINE LEARNING PREDICTION
# ==========================================

if submitted:

    input_data = pd.DataFrame([{

        "meals_served": meals_served,

        "kitchen_staff": kitchen_staff,

        "temperature_C": temperature_C,

        "humidity_percent": humidity_percent,

        "day_of_week": day_of_week,

        "special_event": special_event,

        "past_waste_kg": past_waste_kg,

        "staff_experience": staff_experience,

        "waste_category": waste_category,

        "month": selected_date.month,

        "day": selected_date.day

    }])

    try:

        prediction = model.predict(input_data)[0]

        prediction = max(0, prediction)

        st.divider()

        st.header("📊 Prediction Result")

        st.metric(
            "Predicted Food Waste",
            f"{prediction:.2f} kg"
        )

        if prediction > 20:

            st.warning(
                "High predicted food waste. "
                "Consider improving food preparation planning."
            )

        elif prediction > 10:

            st.info(
                "Moderate predicted food waste. "
                "Consider improving portion planning."
            )

        else:

            st.success(
                "Low predicted food waste."
            )

        st.caption(
            "The prediction is an estimate generated by "
            "the trained Random Forest model."
        )


        # ==========================================
        # 11. SMART DONATION RECOMMENDATIONS
        # ==========================================

        st.divider()

        st.header("🤝 Smart Donation Recommendations")

        if prediction > 0:

            try:

                recommendations = recommend_donation_centers(
                    waste_category,
                    prediction
                )

                if not recommendations.empty:

                    st.success(
                        "Matching donation centers found!"
                    )

                    st.write(
                        f"Food Category: **{waste_category}**"
                    )

                    st.write(
                        f"Predicted Surplus: **{prediction:.2f} kg**"
                    )

                    st.dataframe(
                        recommendations,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.caption(
                        "Sample donation center data is used "
                        "for project demonstration. Verify food "
                        "safety and acceptance before donation."
                    )

                else:

                    st.info(
                        "No suitable donation centers found "
                        "for this category and quantity."
                    )

            except Exception as e:

                st.error(
                    f"Donation recommendation error: {e}"
                )

        else:

            st.info(
                "No surplus food predicted."
            )

    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )


# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.caption(
    "AI-Based Food Waste Prediction and Smart Donation "
    "Recommendation System | "
    "Machine Learning: Random Forest Regressor | "
    "Developed using Python and Streamlit"
)

