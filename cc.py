import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def initialize_session_state():
    """Initialize session state variables"""
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'inputs' not in st.session_state:
        st.session_state.inputs = {
            'project_info': {},
            'team_composition': {},
            'duration_volume': {},
            'operational_costs': {},
            'additional_services': {},
            'discounts': {}
        }


def navigation():
    """Handle navigation between steps"""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.session_state.current_step > 0:
            if st.button('← Previous'):
                st.session_state.current_step -= 1
                st.rerun()

    with col3:
        if st.session_state.current_step < 6:  # Adjust based on total steps
            if st.button('Next →'):
                st.session_state.current_step += 1
                st.rerun()

def validate_dataframe(df, required_columns, numeric_columns, name):
    """Validate DataFrame structure and content"""
    errors = []
    
    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        errors.append(f"{name}: Missing required columns: {', '.join(missing_cols)}")
    
    # Validate numeric columns
    for col in numeric_columns:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"{name}: {col} must contain numeric values")
            else:
                # Check for specific ranges
                if 'Overhead %' in col:
                    if not all((df[col] >= 0) & (df[col] <= 1)):
                        errors.append(f"{name}: {col} must be between 0 and 1")
                elif 'Salary' in col:
                    if not all(df[col] > 0):
                        errors.append(f"{name}: {col} must be positive")
    
    return errors

def validate_inputs(inputs):
    """Validate all user inputs"""
    errors = []
    
    # Validate team composition
    required_cols = ['Role', 'Annual Salary', 'Overhead %']
    numeric_cols = ['Annual Salary', 'Overhead %']
    
    if 'team_composition' in inputs:
        team = inputs['team_composition']
        if 'us_roles' in team:
            errors.extend(validate_dataframe(team['us_roles'], required_cols, numeric_cols, "US Roles"))
        if 'nearshore_roles' in team:
            errors.extend(validate_dataframe(team['nearshore_roles'], required_cols, numeric_cols, "Nearshore Roles"))
    
    # Validate duration and volume
    if 'duration_volume' in inputs:
        dv = inputs['duration_volume']
        if dv.get('num_pods', 0) < 1:
            errors.append("Number of pods must be at least 1")
        if dv.get('duration_months', 0) < 1:
            errors.append("Duration must be at least 1 month")
        if not (50 <= dv.get('nearshore_percentage', 0) <= 80):
            errors.append("Nearshore percentage must be between 50% and 80%")
    
    return errors

def calculate_discounts(num_pods, duration_months, nearshore_percentage):
    """Calculate and validate discounts"""
    try:
        # Volume discount
        volume_discount = 0
        if num_pods >= 10:
            volume_discount = 0.15
        elif num_pods >= 6:
            volume_discount = 0.10
        elif num_pods >= 3:
            volume_discount = 0.05
        
        # Duration discount
        duration_discount = 0
        if duration_months >= 25:
            duration_discount = 0.15
        elif duration_months >= 13:
            duration_discount = 0.10
        elif duration_months >= 6:
            duration_discount = 0.05
        
        # Nearshore discount
        nearshore_discount = 0
        if nearshore_percentage >= 80:
            nearshore_discount = 0.15
        elif nearshore_percentage >= 70:
            nearshore_discount = 0.10
        elif nearshore_percentage >= 60:
            nearshore_discount = 0.05
        
        # Calculate total discount with cap
        total_discount = 1 - (
            (1 - volume_discount) *
            (1 - duration_discount) *
            (1 - nearshore_discount)
        )
        
        # Cap total discount at 50%
        total_discount = min(total_discount, 0.5)
        
        return {
            'volume': volume_discount,
            'duration': duration_discount,
            'nearshore': nearshore_discount,
            'total': total_discount
        }
    except Exception as e:
        st.error(f"Error calculating discounts: {str(e)}")
        return None

def validate_operational_costs(op_costs_df):
    """Validate operational costs"""
    if op_costs_df.empty:
        return ["Operational costs table cannot be empty"]
    
    errors = []
    if not pd.api.types.is_numeric_dtype(op_costs_df['Monthly Cost']):
        errors.append("Monthly Cost must contain numeric values")
    elif any(op_costs_df['Monthly Cost'] < 0):
        errors.append("Monthly Cost cannot be negative")
    
    return errors

def calculate_profit_metrics(base_cost, selling_price, duration_months):
    """Calculate and validate profit metrics"""
    try:
        if base_cost <= 0:
            raise ValueError("Base cost must be positive")
        if selling_price <= 0:
            raise ValueError("Selling price must be positive")
        if duration_months < 1:
            raise ValueError("Duration must be at least 1 month")
        
        profit = selling_price - base_cost
        margin_percentage = (profit / selling_price) * 100
        
        return {
            'profit': profit,
            'margin': margin_percentage,
            'is_profitable': profit > 0
        }
    except Exception as e:
        st.error(f"Error calculating profit metrics: {str(e)}")
        return None

def create_profit_projection(monthly_revenue, monthly_cost, duration):
    """Create and validate profit projection"""
    try:
        if duration < 1:
            raise ValueError("Duration must be at least 1 month")
        
        projection = pd.DataFrame({
            'Month': range(1, duration + 1),
            'Revenue': [monthly_revenue] * duration,
            'Cost': [monthly_cost] * duration
        })
        
        projection['Profit'] = projection['Revenue'] - projection['Cost']
        projection['Margin %'] = (projection['Profit'] / projection['Revenue']) * 100
        projection['Cumulative Profit'] = projection['Profit'].cumsum()
        
        return projection
    except Exception as e:
        st.error(f"Error creating profit projection: {str(e)}")
        return None

def export_quote_summary(inputs, profit_projection):
    """Create comprehensive export of quote and financials"""
    try:
        # Basic information
        summary = {
            'Project Info': {
                'Client': inputs['project_info'].get('client_name', ''),
                'Project': inputs['project_info'].get('project_name', ''),
                'Start Date': inputs['project_info'].get('start_date', '')
            },
            'Configuration': {
                'Number of Pods': inputs['duration_volume'].get('num_pods', 0),
                'Duration (Months)': inputs['duration_volume'].get('duration_months', 0),
                'Nearshore %': inputs['duration_volume'].get('nearshore_percentage', 0)
            },
            'Discounts': inputs['duration_volume'].get('discounts', {}),
            'Financial Projection': profit_projection
        }
        
        return pd.json_normalize(summary)
    except Exception as e:
        st.error(f"Error creating export summary: {str(e)}")
        return None
def step_1_project_info():
    """Project Information"""
    st.header("Step 1: Project Information")

    project_info = st.session_state.inputs['project_info']

    project_info['client_name'] = st.text_input(
        "Client Name",
        value=project_info.get('client_name', '')
    )
    project_info['project_name'] = st.text_input(
        "Project Name",
        value=project_info.get('project_name', '')
    )
    project_info['start_date'] = st.date_input(
        "Project Start Date",
        value=project_info.get('start_date', datetime.now())
    )


def step_2_team_composition():
    """Team Composition and Rates"""
    st.header("Step 2: Team Composition and Rates")
    
    team = st.session_state.inputs['team_composition']
    
    # US-based roles
    st.subheader("US-Based Roles")
    us_roles_data = {
        "Role": ["Data Engineer", "Data Scientist", "Software Developer", "UI/UX Designer"],
        "Annual Salary": [150000, 160000, 140000, 130000],
        "Overhead %": [0.3, 0.3, 0.3, 0.3],
        "Hourly Bill Rate": [175, 185, 165, 155],  # Added billable rates
        "Utilization %": [85, 85, 85, 85]  # Added utilization targets
    }
    team['us_roles'] = st.data_editor(
        pd.DataFrame(us_roles_data),
        key="us_roles_editor",
        use_container_width=True
    )

    # Nearshore roles
    st.subheader("Nearshore Roles")
    nearshore_roles_data = {
        "Role": ["Data Engineer", "Data Scientist", "Software Developer", "UI/UX Designer"],
        "Annual Salary": [80000, 90000, 70000, 65000],
        "Overhead %": [0.25, 0.25, 0.25, 0.25],
        "Hourly Bill Rate": [95, 105, 85, 80],  # Added billable rates
        "Utilization %": [85, 85, 85, 85]  # Added utilization targets
    }
    team['nearshore_roles'] = st.data_editor(
        pd.DataFrame(nearshore_roles_data),
        key="nearshore_roles_editor",
        use_container_width=True
    )

    # Display rate analysis
    st.subheader("Rate Analysis")
    
    def calculate_rate_metrics(roles_df):
        metrics = []
        hours_per_month = 160  # Assuming 160 billable hours per month
        
        for _, row in roles_df.iterrows():
            monthly_salary_cost = (row['Annual Salary'] * (1 + row['Overhead %'])) / 12
            monthly_billable_hours = hours_per_month * (row['Utilization %'] / 100)
            monthly_revenue = monthly_billable_hours * row['Hourly Bill Rate']
            monthly_profit = monthly_revenue - monthly_salary_cost
            margin = (monthly_profit / monthly_revenue) * 100 if monthly_revenue > 0 else 0
            
            metrics.append({
                'Role': row['Role'],
                'Monthly Cost': monthly_salary_cost,
                'Monthly Revenue': monthly_revenue,
                'Monthly Profit': monthly_profit,
                'Margin %': margin,
                'Effective Hourly Cost': monthly_salary_cost / monthly_billable_hours
            })
        
        return pd.DataFrame(metrics)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("US Roles Rate Analysis")
        us_metrics = calculate_rate_metrics(team['us_roles'])
        st.dataframe(
            us_metrics.style.format({
                'Monthly Cost': '${:,.2f}',
                'Monthly Revenue': '${:,.2f}',
                'Monthly Profit': '${:,.2f}',
                'Margin %': '{:.1f}%',
                'Effective Hourly Cost': '${:,.2f}'
            })
        )
    
    with col2:
        st.write("Nearshore Roles Rate Analysis")
        nearshore_metrics = calculate_rate_metrics(team['nearshore_roles'])
        st.dataframe(
            nearshore_metrics.style.format({
                'Monthly Cost': '${:,.2f}',
                'Monthly Revenue': '${:,.2f}',
                'Monthly Profit': '${:,.2f}',
                'Margin %': '{:.1f}%',
                'Effective Hourly Cost': '${:,.2f}'
            })
        )

def calculate_pod_revenue(team_composition, num_pods, utilization):
    """Calculate pod revenue based on billable rates"""
    monthly_revenue = 0
    hours_per_month = 160
    
    # Calculate US roles revenue
    for _, role in team_composition['us_roles'].iterrows():
        monthly_billable_hours = hours_per_month * (role['Utilization %'] / 100)
        monthly_revenue += monthly_billable_hours * role['Hourly Bill Rate']
    
    # Calculate nearshore roles revenue
    for _, role in team_composition['nearshore_roles'].iterrows():
        monthly_billable_hours = hours_per_month * (role['Utilization %'] / 100)
        monthly_revenue += monthly_billable_hours * role['Hourly Bill Rate']
    
    return monthly_revenue * num_pods

def step_3_duration_volume():
    """Duration and Volume"""
    st.header("Step 3: Pod Configuration and Pricing")

    duration_volume = st.session_state.inputs['duration_volume']

    # Pod Configuration
    col1, col2 = st.columns(2)

    with col1:
        duration_volume['num_pods'] = st.number_input(
            "Number of Pods",
            min_value=1,
            value=duration_volume.get('num_pods', 1)
        )
        duration_volume['duration_months'] = st.number_input(
            "Engagement Duration (Months)",
            min_value=1,
            value=duration_volume.get('duration_months', 6)
        )

    with col2:
        duration_volume['nearshore_percentage'] = st.slider(
            "Nearshore Percentage",
            min_value=50,
            max_value=80,
            value=duration_volume.get('nearshore_percentage', 50),
            step=10
        )

    # Automatic Discount Calculations
    st.subheader("Automatic Discounts")

    # Volume-based discount
    volume_discount = 0
    if duration_volume['num_pods'] >= 10:
        volume_discount = 0.15
    elif duration_volume['num_pods'] >= 6:
        volume_discount = 0.10
    elif duration_volume['num_pods'] >= 3:
        volume_discount = 0.05

    # Duration-based discount
    duration_discount = 0
    if duration_volume['duration_months'] >= 25:
        duration_discount = 0.15
    elif duration_volume['duration_months'] >= 13:
        duration_discount = 0.10
    elif duration_volume['duration_months'] >= 6:
        duration_discount = 0.05

    # Nearshore discount
    nearshore_discount = 0
    if duration_volume['nearshore_percentage'] >= 80:
        nearshore_discount = 0.15
    elif duration_volume['nearshore_percentage'] >= 70:
        nearshore_discount = 0.10
    elif duration_volume['nearshore_percentage'] >= 60:
        nearshore_discount = 0.05

    # Display discount summary
    discount_data = {
        "Discount Type": ["Volume", "Duration", "Nearshore"],
        "Percentage": [
            f"{volume_discount*100:.1f}%",
            f"{duration_discount*100:.1f}%",
            f"{nearshore_discount*100:.1f}%"
        ],
        "Reason": [
            f"{duration_volume['num_pods']} pods",
            f"{duration_volume['duration_months']} months",
            f"{duration_volume['nearshore_percentage']}% nearshore"
        ]
    }
    st.table(pd.DataFrame(discount_data))

    # Store discounts in session state
    duration_volume['discounts'] = {
        'volume': volume_discount,
        'duration': duration_discount,
        'nearshore': nearshore_discount
    }

def step_4_operational_costs():
    """Operational Costs"""
    st.header("Step 4: Operational Costs")

    op_costs = st.session_state.inputs['operational_costs']

    operational_data = {
        "Cost Category": ["Databricks", "Cloud Storage", "Monitoring Tools", "Training", "Travel"],
        "Monthly Cost": [5000, 2000, 1000, 500, 1000],
    }

    op_costs['costs'] = st.data_editor(
        pd.DataFrame(operational_data),
        key="operational_costs_editor"
    )


def step_5_additional_services():
    """Additional Services"""
    st.header("Step 5: Additional Services")

    additional = st.session_state.inputs['additional_services']

    addon_data = {
        "Role": ["Engagement Manager", "ML Ops Engineer", "Cloud Architect", "QA Engineer"],
        "US Cost": [12000, 18000, 20000, 10000],
        "Nearshore Cost": [8000, 12000, 14000, 6000]
    }

    st.write("Available Add-on Services")
    st.dataframe(pd.DataFrame(addon_data))

    additional['selected_services'] = {}
    for role in addon_data["Role"]:
        col1, col2 = st.columns(2)
        with col1:
            us_count = st.number_input(
                f"US {role} Count", min_value=0, value=0)
        with col2:
            ns_count = st.number_input(
                f"Nearshore {role} Count", min_value=0, value=0)
        additional['selected_services'][role] = {
            "US": us_count, "Nearshore": ns_count}


def step_6_discounts():
    """Discounts and Incentives"""
    st.header("Step 6: Discounts and Incentives")

    discounts = st.session_state.inputs['discounts']

    # Calculate automatic discounts based on volume and duration
    num_pods = st.session_state.inputs['duration_volume']['num_pods']
    duration = st.session_state.inputs['duration_volume']['duration_months']

    # Volume discount
    volume_discount = 0
    if num_pods >= 10:
        volume_discount = 0.15
    elif num_pods >= 6:
        volume_discount = 0.10
    elif num_pods >= 3:
        volume_discount = 0.05

    # Duration discount
    duration_discount = 0
    if duration >= 25:
        duration_discount = 0.15
    elif duration >= 13:
        duration_discount = 0.10
    elif duration >= 6:
        duration_discount = 0.05

    discounts['volume_discount'] = volume_discount
    discounts['duration_discount'] = duration_discount

    st.write(f"Volume Discount: {volume_discount*100}%")
    st.write(f"Duration Discount: {duration_discount*100}%")

    # Custom discount
    discounts['custom_discount'] = st.number_input(
        "Additional Custom Discount (%)",
        min_value=0.0,
        max_value=100.0,
        value=discounts.get('custom_discount', 0.0)
    ) / 100

def step_7_final_quote():
    """Enhanced final quote with revenue based on billable rates"""
    st.header("Final Quote Summary")
    
    try:
        # Validate all inputs
        errors = validate_inputs(st.session_state.inputs)
        if errors:
            for error in errors:
                st.error(error)
            return

        # Initialize constants
        hours_per_month = 160  # Standard billable hours per month
        
        # Get team composition and configuration details
        team = st.session_state.inputs['team_composition']
        duration_volume = st.session_state.inputs['duration_volume']
        num_pods = duration_volume['num_pods']
        duration = duration_volume['duration_months']
        nearshore_percentage = duration_volume['nearshore_percentage']

        # Calculate base costs
        us_costs = (team['us_roles']['Annual Salary'] * (1 + team['us_roles']['Overhead %'])).sum() / 12
        nearshore_costs = (team['nearshore_roles']['Annual Salary'] * (1 + team['nearshore_roles']['Overhead %'])).sum() / 12
        base_pod_cost = us_costs + nearshore_costs

        # Calculate base revenue from billable rates
        us_revenue = sum(
            hours_per_month * (row['Utilization %'] / 100) * row['Hourly Bill Rate']
            for _, row in team['us_roles'].iterrows()
        )
        nearshore_revenue = sum(
            hours_per_month * (row['Utilization %'] / 100) * row['Hourly Bill Rate']
            for _, row in team['nearshore_roles'].iterrows()
        )
        base_pod_revenue = us_revenue + nearshore_revenue

        # Calculate discounts
        discounts = calculate_discounts(num_pods, duration, nearshore_percentage)
        total_discount = discounts['total']

        # Get operational costs
        op_costs = st.session_state.inputs['operational_costs']['costs']['Monthly Cost'].sum()

        # Calculate additional services costs and revenue
        additional_services = st.session_state.inputs['additional_services']['selected_services']
        addon_costs = sum(
            service["US"] * 12000 + service["Nearshore"] * 8000
            for service in additional_services.values()
        )
        addon_revenue = sum(
            (service["US"] * 15000 + service["Nearshore"] * 10000)  # Assuming markup on add-on services
            for service in additional_services.values()
        )

        # Calculate monthly figures
        monthly_cost = (base_pod_cost * num_pods) + op_costs + addon_costs
        monthly_revenue = (base_pod_revenue * num_pods * (1 - total_discount)) + addon_revenue
        monthly_profit = monthly_revenue - monthly_cost
        monthly_margin = (monthly_profit / monthly_revenue) * 100 if monthly_revenue > 0 else 0

        # Display summary metrics
        st.subheader("Monthly Financial Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Revenue", f"${monthly_revenue:,.2f}")
        with col2:
            st.metric("Cost", f"${monthly_cost:,.2f}")
        with col3:
            st.metric("Profit", f"${monthly_profit:,.2f}")
        with col4:
            st.metric("Margin", f"{monthly_margin:.1f}%")

        # Display rate analysis
        st.subheader("Rate Analysis")
        col1, col2, col3 = st.columns(3)
        
        total_monthly_hours = hours_per_month * num_pods
        with col1:
            st.metric(
                "Average Hourly Rate", 
                f"${(monthly_revenue / total_monthly_hours):,.2f}",
                help="Average revenue per billable hour across all resources"
            )
        with col2:
            st.metric(
                "Effective Cost Rate", 
                f"${(monthly_cost / total_monthly_hours):,.2f}",
                help="Average cost per billable hour across all resources"
            )
        with col3:
            st.metric(
                "Rate Margin", 
                f"${((monthly_revenue - monthly_cost) / total_monthly_hours):,.2f}",
                help="Average profit per billable hour"
            )

        # Display discount summary
        st.subheader("Applied Discounts")
        discount_df = pd.DataFrame({
            "Discount Type": ["Volume", "Duration", "Nearshore", "Total"],
            "Percentage": [
                f"{discounts['volume']*100:.1f}%",
                f"{discounts['duration']*100:.1f}%",
                f"{discounts['nearshore']*100:.1f}%",
                f"{total_discount*100:.1f}%"
            ],
            "Monthly Impact": [
                f"${base_pod_revenue * num_pods * discounts['volume']:,.2f}",
                f"${base_pod_revenue * num_pods * discounts['duration']:,.2f}",
                f"${base_pod_revenue * num_pods * discounts['nearshore']:,.2f}",
                f"${base_pod_revenue * num_pods * total_discount:,.2f}"
            ]
        })
        st.table(discount_df)

        # Create and display profit projection
        profit_projection = create_profit_projection(monthly_revenue, monthly_cost, duration)
        
        if profit_projection is not None:
            st.subheader("Profit Projection")
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(profit_projection['Month'], profit_projection['Revenue'], label='Revenue', color='blue')
            ax.plot(profit_projection['Month'], profit_projection['Cost'], label='Cost', color='red')
            ax.plot(profit_projection['Month'], profit_projection['Profit'], label='Profit', color='green')
            ax.fill_between(profit_projection['Month'], profit_projection['Profit'], color='green', alpha=0.1)
            ax.set_xlabel('Month')
            ax.set_ylabel('Amount ($)')
            ax.legend()
            st.pyplot(fig)

            # Detailed metrics table
            st.subheader("Monthly Financial Metrics")
            st.dataframe(
                profit_projection.style.format({
                    'Revenue': '${:,.2f}',
                    'Cost': '${:,.2f}',
                    'Profit': '${:,.2f}',
                    'Margin %': '{:.1f}%'
                })
            )

        # Calculate and display engagement totals
        total_engagement_revenue = monthly_revenue * duration
        total_engagement_cost = monthly_cost * duration
        total_engagement_profit = monthly_profit * duration

        st.subheader("Total Engagement Summary")
        engagement_metrics = pd.DataFrame({
            "Metric": [
                "Total Revenue",
                "Total Cost",
                "Total Profit",
                "Average Margin",
                "Total Hours",
                "Average Hourly Rate"
            ],
            "Value": [
                f"${total_engagement_revenue:,.2f}",
                f"${total_engagement_cost:,.2f}",
                f"${total_engagement_profit:,.2f}",
                f"{monthly_margin:.1f}%",
                f"{total_monthly_hours * duration:,}",
                f"${(total_engagement_revenue / (total_monthly_hours * duration)):,.2f}"
            ]
        })
        st.table(engagement_metrics)

        # Display warnings
        if monthly_profit < 0:
            st.warning("⚠️ Warning: Project showing negative profit margins")
        if total_discount > 0.4:
            st.warning("⚠️ High discount rate: Consider reviewing pricing strategy")
        if monthly_margin < 20:
            st.warning("⚠️ Low profit margin: Consider reviewing rates or costs")

        # Export options
        if st.button("Export Complete Quote Package"):
            summary_export = export_quote_summary(
                st.session_state.inputs,
                profit_projection
            )
            if summary_export is not None:
                st.download_button(
                    label="Download Quote Package",
                    data=summary_export.to_csv(index=False),
                    file_name="quote_package.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"Error generating final quote: {str(e)}")
        st.error("Detailed error information:", exc_info=True)
def main():
    st.title("AI Pod Quote Builder")

    # Initialize session state
    initialize_session_state()

    # Progress bar
    st.progress(st.session_state.current_step / 6)

    # Steps
    steps = [
        step_1_project_info,
        step_2_team_composition,
        step_3_duration_volume,
        step_4_operational_costs,
        step_5_additional_services,
        step_6_discounts,
        step_7_final_quote
    ]

    # Display current step
    steps[st.session_state.current_step]()

    # Navigation
    navigation()


if __name__ == "__main__":
    main()
