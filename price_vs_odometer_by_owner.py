import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import streamlit as st
from utils import utils
from sidebar import sidebar_components
from price_vs_odometer import polynomial_regression




def plot_price_vs_odo_by_owner(data, num_samples, selected_make, selected_model, selected_variant, selected_fuel_type, selected_Mfg_Year):

    # Get unique manufacturing years for the selected make, model, variant, and fuel type
    Year_options = data[(data['Make'] == selected_make) & 
                        (data['Model'] == selected_model) & 
                        (data['Variant'] == selected_variant) & 
                        (data['Fuel_Type'] == selected_fuel_type)]['Mfg_Year'].unique()
    
    # Allow the user to select a manufacturing year
    selected_Mfg_Year = st.sidebar.selectbox('Select Mfg Year:', Year_options)

    # Filter data based on selected year of manufacture
    filtered_data_year = data[data['Mfg_Year'] == selected_Mfg_Year]

    # Group data by number of owners
    grouped_data = filtered_data_year.groupby('No_Of_Ownership')

    # Initialize the plot
    fig = go.Figure()

    # Initialize list to store coefficients
    coeff_list_2nd_degree = []
    coeff_list_2nd_degree_sample = []

    # Create a smooth odometer range for trendline prediction
    X_smooth = np.linspace(data['Odometer_Reading'].min(), 
                           data['Odometer_Reading'].max(), 100).reshape(-1, 1)

    # Loop through each group (number of owners) and create plots
    for no_of_owners, group in grouped_data:
        if len(group) > 2:  # Ensure at least 3 data points
            # Full data points
            X_full = group['Odometer_Reading'].values.reshape(-1, 1)
            y_full = group['Tradein_MarketPrice'].values

            # Perform 2nd-degree polynomial regression on full data
            degree = 2
            alpha = 0.001
            poly_features_2 = PolynomialFeatures(degree=degree)
            regressor_2 = Ridge(alpha=alpha)
            model_2 = make_pipeline(poly_features_2, regressor_2)
            model_2.fit(X_full, y_full)

            # Make predictions for the full data over the entire range of odometer readings
            y_pred_full_smooth = model_2.predict(X_smooth)

            # Calculate R² value for the full data
            y_pred_train_2_full = model_2.predict(X_full)
            r2_full = r2_score(y_full, y_pred_train_2_full)

            # Sort for plotting
            sorted_indices_full = np.argsort(X_smooth.flatten())
            X_smooth_sorted = X_smooth[sorted_indices_full]
            y_pred_full_sorted = y_pred_full_smooth[sorted_indices_full]

            # Add data points and regression line for the full data
            fig.add_trace(go.Scatter(x=X_full.flatten(), y=y_full, mode='markers', name=f'Owners: {no_of_owners} (Full)'))
            fig.add_trace(go.Scatter(x=X_smooth_sorted.flatten(), y=y_pred_full_sorted, mode='lines', name=f'Owners: {no_of_owners} (Full 2nd Degree fit)'))

            # Store the coefficients and R² value
            intercept_2_full = model_2.named_steps['ridge'].intercept_
            coef_2_full = model_2.named_steps['ridge'].coef_
            equation_2_full = f'y = {intercept_2_full:.3f} + {coef_2_full[1]:.3f}x + {coef_2_full[2]:.3f}x^2'

            coeff_list_2nd_degree.append({
                'Make': selected_make,
                'Model': selected_model,
                'Variant': selected_variant,
                'Fuel_Type': selected_fuel_type,
                'Year of Manufacture': selected_Mfg_Year,
                'No_Of_Ownership': no_of_owners,
                'Number of Data Points': len(group),
                'Intercept (2nd Degree)': intercept_2_full,
                'Coefficient 1 (2nd Degree)': coef_2_full[1], 
                'Coefficient 2 (2nd Degree)': coef_2_full[2],
                'R² (2nd Degree)': r2_full,
                'Equation (2nd Degree)': equation_2_full
            })

    # Sample data points (filtered by odometer range)
    sample_data = filtered_data_year[filtered_data_year['Mfg_Year'] == selected_Mfg_Year]
    # sample_data = filtered_data_year.sample(n=num_samples, random_state=42)
    group_sample = sample_data.groupby('No_Of_Ownership')

    for no_of_owners, group in group_sample:
        if len(group) > 2:  # Ensure at least 3 sample points
            group = group.sample(n=min(num_samples, len(group)), random_state=42)
            X_sample = group['Odometer_Reading'].values.reshape(-1, 1)
            y_sample = group['Tradein_MarketPrice'].values

            # Perform 2nd-degree polynomial regression on sample data
            degree = 2
            alpha = 0.001
            poly_features_2 = PolynomialFeatures(degree=degree)
            regressor_2 = Ridge(alpha=alpha)
            model_2_sample = make_pipeline(poly_features_2, regressor_2)
            model_2_sample.fit(X_sample, y_sample)
            y_pred_sample_smooth = model_2_sample.predict(X_smooth)

            # Sort for plotting
            sorted_indices_full = np.argsort(X_smooth.flatten())
            X_smooth_sorted = X_smooth[sorted_indices_full]
            y_pred_sample_sorted = y_pred_sample_smooth[sorted_indices_full]

            # Add the sample data points and sample regression line
            fig.add_trace(go.Scatter(x=X_sample.flatten(), y=y_sample, mode='markers', name=f'Owners: {no_of_owners} (Sample Data)', marker=dict(symbol='x')))
            fig.add_trace(go.Scatter(x=X_smooth_sorted.flatten(), y=y_pred_sample_sorted, mode='lines', line=dict(dash='dash'), name=f'Owners: {no_of_owners} (Sample 2nd Degree fit)'))

            # Store the coefficients and R² value
            intercept_2_sample = model_2_sample.named_steps['ridge'].intercept_
            coef_2_sample = model_2_sample.named_steps['ridge'].coef_
            equation_2_sample = f'y = {intercept_2_sample:.3f} + {coef_2_sample[1]:.3f}x + {coef_2_sample[2]:.3f}x^2'

            coeff_list_2nd_degree_sample.append({
                'Make': selected_make,
                'Model': selected_model,
                'Variant': selected_variant,
                'Fuel_Type': selected_fuel_type,
                'Year of Manufacture': selected_Mfg_Year,
                'No_Of_Ownership': no_of_owners,
                'Number of Data Points': len(group),
                'Intercept (2nd Degree)': intercept_2_sample,
                'Coefficient 1 (2nd Degree)': coef_2_sample[1], 
                'Coefficient 2 (2nd Degree)': coef_2_sample[2],
                'R² (2nd Degree)': r2_score(y_sample, model_2_sample.predict(X_sample)),
                'Equation (2nd Degree)': equation_2_sample
            })

    # Set plot title and labels
    fig.update_layout(
        title=f'2nd Degree Polynomial Fit by Ownership ({selected_make} {selected_model} {selected_variant} {selected_fuel_type}) - Year {selected_Mfg_Year}',
        xaxis_title='Odometer Reading',
        yaxis_title='IBB Trade-In Price'
    )

    return fig, coeff_list_2nd_degree, coeff_list_2nd_degree_sample