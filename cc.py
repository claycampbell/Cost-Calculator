import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Title and Description
st.title("AI Resource Allocation and Profitability Calculator")
st.write("Enter roles, costs, hourly rates, and resource allocations to calculate profitability and margin.")

# Sidebar Inputs
st.sidebar.header("Inputs")

# Timeline Inputs
start_date = st.sidebar.date_input("Start Date", datetime(2025, 1, 1))
months_of_analysis = st.sidebar.number_input("Months of Analysis", min_value=1, max_value=120, value=24)

# Work hours per month
hours_per_month = st.sidebar.number_input("Working Hours per Month", value=160)

# Editable Role Tables for US and Nearshore
st.header("Roles, Costs, and Hourly Bill Rates")

# US Roles
st.subheader("US Roles")
us_role_data = {
    "Role": ["Data Engineer", "Data Scientist", "Software Developer", "UI/UX Designer"],
    "Annual Salary": [150000, 160000, 140000, 130000],
    "Overhead %": [0.3, 0.3, 0.3, 0.3],
    "Hourly Bill Rate": [150, 170, 140, 130],  # Hourly bill rate per resource
}
us_role_df = pd.DataFrame(us_role_data)
us_role_df = st.data_editor(us_role_df, use_container_width=True, num_rows="dynamic", key="us_roles")

# Nearshore Roles
st.subheader("Nearshore Roles")
nearshore_role_data = {
    "Role": ["Data Engineer", "Data Scientist", "Software Developer", "UI/UX Designer"],
    "Annual Salary": [80000, 90000, 70000, 65000],
    "Overhead %": [0.25, 0.25, 0.25, 0.25],
    "Hourly Bill Rate": [100, 110, 90, 85],  # Hourly bill rate per resource
}
nearshore_role_df = pd.DataFrame(nearshore_role_data)
nearshore_role_df = st.data_editor(nearshore_role_df, use_container_width=True, num_rows="dynamic", key="nearshore_roles")

# Timeline Section
st.header("Resource Allocation Timeline")
months = pd.date_range(start=start_date, periods=months_of_analysis, freq='MS')
timeline_data = {
    "Month": months,
}
# Add US and Nearshore roles as columns for resource allocation
for role in us_role_df["Role"]:
    timeline_data[f"US {role}"] = [0] * months_of_analysis
for role in nearshore_role_df["Role"]:
    timeline_data[f"Nearshore {role}"] = [0] * months_of_analysis

timeline_df = pd.DataFrame(timeline_data)

# Allow user to edit the timeline
timeline_df = st.data_editor(timeline_df, use_container_width=True, num_rows="dynamic", key="timeline")

# Cost and Revenue Calculations
total_costs = []
total_revenues = []
profits = []
margins = []

for _, row in timeline_df.iterrows():
    monthly_cost = 0
    monthly_revenue = 0

    # Calculate costs and revenues for US roles
    for _, role_row in us_role_df.iterrows():
        role = role_row["Role"]
        column = f"US {role}"
        if column in timeline_df.columns:
            salary = role_row["Annual Salary"]
            overhead = role_row["Overhead %"]
            monthly_salary = (salary * (1 + overhead)) / 12
            resource_count = row[column]
            monthly_cost += resource_count * monthly_salary

            hourly_bill_rate = role_row["Hourly Bill Rate"]
            monthly_revenue += resource_count * hourly_bill_rate * hours_per_month

    # Calculate costs and revenues for Nearshore roles
    for _, role_row in nearshore_role_df.iterrows():
        role = role_row["Role"]
        column = f"Nearshore {role}"
        if column in timeline_df.columns:
            salary = role_row["Annual Salary"]
            overhead = role_row["Overhead %"]
            monthly_salary = (salary * (1 + overhead)) / 12
            resource_count = row[column]
            monthly_cost += resource_count * monthly_salary

            hourly_bill_rate = role_row["Hourly Bill Rate"]
            monthly_revenue += resource_count * hourly_bill_rate * hours_per_month

    total_costs.append(monthly_cost)
    total_revenues.append(monthly_revenue)
    profit = monthly_revenue - monthly_cost
    profits.append(profit)
    margin = (profit / monthly_revenue) * 100 if monthly_revenue > 0 else 0
    margins.append(margin)

# Add calculated columns to timeline_df
timeline_df["Total Cost"] = total_costs
timeline_df["Total Revenue"] = total_revenues
timeline_df["Profit"] = profits
timeline_df["Margin %"] = margins

# Display Results
st.subheader("Results")
st.write(timeline_df)

# Profitability Chart
st.subheader("Profitability Over Time")
fig, ax = plt.subplots()
ax.plot(timeline_df["Month"], timeline_df["Profit"], label="Profit", color="green")
ax.set_title("Profit Over Time")
ax.set_xlabel("Month")
ax.set_ylabel("Profit")
ax.legend()
st.pyplot(fig)

# Margin Chart
st.subheader("Margin % Over Time")
fig, ax = plt.subplots()
ax.plot(timeline_df["Month"], timeline_df["Margin %"], label="Margin %", color="blue")
ax.set_title("Margin % Over Time")
ax.set_xlabel("Month")
ax.set_ylabel("Margin %")
ax.legend()
st.pyplot(fig)

# Download Option
st.subheader("Download Results")
st.download_button(
    label="Download as CSV",
    data=timeline_df.to_csv(index=False),
    file_name="resource_allocation_profitability.csv",
    mime="text/csv"
)
