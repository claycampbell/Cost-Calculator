import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Title and Description
st.title("AI Cost Model and Profitability Dashboard")
st.write("Analyze resource allocation, operational costs, revenue, and profitability.")

# Sidebar Inputs
st.sidebar.header("Inputs")
start_date = st.sidebar.date_input("Start Date", datetime(2025, 1, 1))
months_of_analysis = st.sidebar.number_input("Months of Analysis", min_value=1, max_value=120, value=24)
hours_per_month = st.sidebar.number_input("Working Hours per Month", value=160)

# Tabs for different pages
tab_names = ["Personnel Costs", "Operational Costs", "Admin Costs", "Revenue", "P&L", "Dashboard"]
tabs = st.tabs(tab_names)

# Shared DataFrames for storing calculated results
personnel_costs_df = pd.DataFrame()
operational_costs_df = pd.DataFrame()
admin_costs_df = pd.DataFrame()
revenue_df = pd.DataFrame()
pnl_df = pd.DataFrame()

# Tab 1: Personnel Costs
with tabs[0]:
    st.header("Personnel Costs")

    # US Roles
    st.subheader("US Roles")
    us_role_data = {
        "Role": ["Data Engineer", "Data Scientist", "Software Developer", "UI/UX Designer"],
        "Annual Salary": [150000, 160000, 140000, 130000],
        "Overhead %": [0.3, 0.3, 0.3, 0.3],
        "Hourly Bill Rate": [150, 170, 140, 130],
    }
    us_role_df = pd.DataFrame(us_role_data)
    us_role_df = st.data_editor(us_role_df, use_container_width=True, num_rows="dynamic", key="us_roles")

    # Nearshore Roles
    st.subheader("Nearshore Roles")
    nearshore_role_data = {
        "Role": ["Data Engineer", "Data Scientist", "Software Developer", "UI/UX Designer"],
        "Annual Salary": [80000, 90000, 70000, 65000],
        "Overhead %": [0.25, 0.25, 0.25, 0.25],
        "Hourly Bill Rate": [100, 110, 90, 85],
    }
    nearshore_role_df = pd.DataFrame(nearshore_role_data)
    nearshore_role_df = st.data_editor(nearshore_role_df, use_container_width=True, num_rows="dynamic", key="nearshore_roles")

    # Timeline Section
    st.header("Resource Allocation Timeline")
    months = pd.date_range(start=start_date, periods=months_of_analysis, freq='MS')

    timeline_data = {"Month": months}
    for role in us_role_df["Role"]:
        timeline_data[f"US {role}"] = [0] * months_of_analysis
    for role in nearshore_role_df["Role"]:
        timeline_data[f"Nearshore {role}"] = [0] * months_of_analysis

    timeline_df = pd.DataFrame(timeline_data)
    timeline_df = st.data_editor(timeline_df, use_container_width=True, num_rows="dynamic", key="timeline")

    # Cost and Revenue Calculations
    personnel_costs = []
    personnel_revenue = []
    for _, row in timeline_df.iterrows():
        monthly_cost = 0
        monthly_revenue = 0
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

        personnel_costs.append(monthly_cost)
        personnel_revenue.append(monthly_revenue)

    timeline_df["Total Cost"] = personnel_costs
    timeline_df["Total Revenue"] = personnel_revenue
    personnel_costs_df = timeline_df.copy()

    # Display Results
    st.subheader("Personnel Cost Results")
    st.write(timeline_df)

# Tab 2: Operational Costs
# Tab 2: Operational Costs
with tabs[1]:
    st.header("Operational Costs")
    operational_data = {
        "Month": months,
        "Databricks Cost": [5000] * months_of_analysis,
        "Cloud Storage Cost": [2000] * months_of_analysis,
        "Monitoring Tools Cost": [1000] * months_of_analysis,
    }
    operational_costs_df = pd.DataFrame(operational_data)
    
    # Calculate Total Operational Costs (excluding 'Month')
    operational_costs_df["Total Operational Costs"] = operational_costs_df.loc[:, operational_costs_df.columns != "Month"].sum(axis=1)
    
    st.write(operational_costs_df)

# Tab 3: Admin Costs
with tabs[2]:
    st.header("Admin Costs")
    admin_data = {
        "Month": months,
        "Account Manager Cost": [10000] * months_of_analysis,
        "PM/Coordination Cost": [7000] * months_of_analysis,
    }
    admin_costs_df = pd.DataFrame(admin_data)
    # Exclude non-numeric columns like 'Month' when summing
    admin_costs_df["Total Admin Costs"] = admin_costs_df.loc[:, admin_costs_df.columns != "Month"].sum(axis=1)

    st.write(admin_costs_df)

# Tab 4: Revenue
with tabs[3]:
    st.header("Revenue")
    revenue_df = personnel_costs_df[["Month", "Total Revenue"]].copy()
    st.write(revenue_df)

# Tab 5: Profit & Loss
with tabs[4]:
    st.header("Profit & Loss")
    pnl_df = pd.DataFrame({
        "Month": months,
        "Total Costs": personnel_costs_df["Total Cost"] + operational_costs_df["Total Operational Costs"] + admin_costs_df["Total Admin Costs"],
        "Total Revenue": personnel_costs_df["Total Revenue"],
    })
    pnl_df["Profit"] = pnl_df["Total Revenue"] - pnl_df["Total Costs"]
    pnl_df["Margin %"] = (pnl_df["Profit"] / pnl_df["Total Revenue"]) * 100
    st.write(pnl_df)

# Tab 6: Dashboard
with tabs[5]:
    st.header("Dashboard")
    st.metric("Total Profit", f"${pnl_df['Profit'].sum():,.2f}")
    st.metric("Total Revenue", f"${pnl_df['Total Revenue'].sum():,.2f}")
    st.metric("Total Costs", f"${pnl_df['Total Costs'].sum():,.2f}")
    fig, ax = plt.subplots()
    ax.plot(pnl_df["Month"], pnl_df["Profit"], label="Profit", color="green")
    ax.set_title("Profit Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Profit")
    ax.legend()
    st.pyplot(fig)
