import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import zscore

# Forecast function (summarized version of our tuned model)
def forecast_grade(input_df, category, year=2025):
    params = {
        "excellent": {"sensitivity_factor": 3.0, "trend_weight": 0.15, "max_effect": 0.15, "direction": "both"},
        "good": {"sensitivity_factor": 4.0, "trend_weight": 0.10, "max_effect": 0.05, "direction": "both"},
        "mid": {"sensitivity_factor": 5.0, "trend_weight": 0.05, "max_effect": 0.03, "direction": "both"},
        "bad": {"sensitivity_factor": 1.0, "trend_weight": 0.00, "max_effect": 0.10, "direction": "positive"},
    }[category]

    df = input_df.copy()
    df['Min_Passed'] = df[['Passed_Romanian_Pct', 'Passed_Math_Pct']].min(axis=1)
    df['Demand'] = (df['Participation_Pct'] / 100) * (df['Min_Passed'] / 100)

    grade_std = np.std(df['Admission_Grade'])
    if any(abs(df['Admission_Grade'].diff().dropna()) > 0.5):
        degree = 1
    elif grade_std < 0.2:
        degree = 1
    elif grade_std < 0.5:
        degree = 2
    else:
        degree = 3

    use_fallback = any(abs(df['Admission_Grade'].diff().dropna()) > 0.7)
    if not use_fallback:
        grade_poly = np.poly1d(np.polyfit(df['Year'], df['Admission_Grade'], degree))
        raw_forecast = grade_poly(year)
        demand_poly = np.poly1d(np.polyfit(df['Year'], df['Demand'], degree))
        demand_forecast = demand_poly(year)
    else:
        raw_forecast = np.polyfit(df['Year'], df['Admission_Grade'], 1)[0] * (year - df['Year'].mean()) + df['Admission_Grade'].mean()
        demand_forecast = np.polyfit(df['Year'], df['Demand'], 1)[0] * (year - df['Year'].mean()) + df['Demand'].mean()

    demand_delta = demand_forecast - df['Demand'].mean()
    x = demand_delta * 40 * params['sensitivity_factor']
    if params['direction'] == "both":
        demand_adjustment = np.tanh(x) * params['max_effect']
    elif params['direction'] == "positive":
        demand_adjustment = max(np.tanh(x), 0) * params['max_effect']
    else:
        demand_adjustment = 0

    adjusted = raw_forecast + demand_adjustment

    if params['trend_weight'] > 0:
        trend_slope, trend_intercept = np.polyfit(df['Year'], df['Admission_Grade'], 1)
        trend_grade = trend_slope * year + trend_intercept
        grade_diff = adjusted - df['Admission_Grade'].iloc[-1]
        trend_diff = trend_grade - df['Admission_Grade'].iloc[-1]
        if np.sign(grade_diff) != np.sign(trend_diff) and abs(grade_diff - trend_diff) > 0.3:
            adjusted = params['trend_weight'] * trend_grade + (1 - params['trend_weight']) * adjusted

    adjusted = max(min(adjusted, 10.0), 5.0)
    return round(raw_forecast, 2), round(adjusted, 2), round(demand_forecast, 4)

# Streamlit App
st.title("🎓 High School Admission Grade Forecaster")

school_type = st.selectbox("Select your school category", ["excellent", "good", "mid", "bad"])
years = st.multiselect("Select years (3 only)", [2022, 2023, 2024], default=[2022, 2023, 2024])

grades = []
for year in years:
    g = st.number_input(f"Grade {year}", min_value=5.0, max_value=10.0, step=0.01, key=f"g{year}")
    grades.append(g)

# Use default participation and passing rates (can be made school-specific later)
default_participation = {2022: 95.5, 2023: 95.3, 2024: 95.3}
default_romanian = {2022: 86.1, 2023: 77.4, 2024: 77.6}
default_math = {2022: 77.5, 2023: 75.4, 2024: 68.7}

if st.button("📈 Forecast Grade"):
    df = pd.DataFrame({
        "Year": years,
        "Admission_Grade": grades,
        "Participation_Pct": [default_participation[y] for y in years],
        "Passed_Romanian_Pct": [default_romanian[y] for y in years],
        "Passed_Math_Pct": [default_math[y] for y in years]
    })
    raw, adjusted, demand = forecast_grade(df, school_type)
    st.success(f"🎯 Raw Forecast for 2025: {raw}")
    st.success(f"📊 Adjusted Forecast for 2025: {adjusted}")
    st.info(f"📉 Forecasted Demand Coefficient: {demand}")

    st.line_chart(pd.DataFrame({"Admission Grade": grades}, index=years))

