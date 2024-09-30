# polynomial_regression.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
import streamlit as st

def polynomial_regression(odometer_filtered_data, num_samples, selected_make, selected_model, selected_variant, selected_fuel_type, selected_no_of_owners):
    # Group data by year of manufacture
    
    grouped_data = odometer_filtered_data.groupby('Mfg_Year')

    # Initialize the plots
    fig2 = go.Figure()
    fig3 = go.Figure()

    # Create an empty list to store dictionaries of coefficients for both degrees
    coeff_list_2nd_degree = []
    coeff_list_2nd_degree_sample_points = []

    # Loop through each group (year of manufacture) and plot the data and polynomial regression fits for 2nd degree
    for name, group in grouped_data:
        if len(group) > 2:  # Filter groups with less than 3 data points
            # Split the data into independent variables (X) and target variable (y)
            X = group['Odometer_Reading'].values.reshape(-1, 1)
            y = group['Tradein_MarketPrice'].values

            # 2nd-degree polynomial regression using all points
            degree = 2
            alpha = 0.001  # Ridge regularization strength
            poly_features_2 = PolynomialFeatures(degree=degree)
            regressor_2 = Ridge(alpha=alpha)
            model_2 = make_pipeline(poly_features_2, regressor_2)
            model_2.fit(X, y)

            # Make predictions over a range of values for a smoother curve
            X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            y_pred_2 = model_2.predict(X_range)

            # Calculate R² value
            y_pred_train_2 = model_2.predict(X)
            r2_2 = r2_score(y, y_pred_train_2)

            # Sort the values for plotting
            sorted_indices = np.argsort(X_range.flatten())
            X_sorted = X_range[sorted_indices]
            y_pred_2_sorted = y_pred_2[sorted_indices]

            # Plot the data and regression fits for 2nd degree
            fig2.add_trace(go.Scatter(x=X.flatten(), y=y, mode='markers', name=f'Year {name}'))
            fig2.add_trace(go.Scatter(x=X_sorted.flatten(), y=y_pred_2_sorted, mode='lines', name=f'{name} (2nd Degree fit)'))

            # Get the coefficients of the polynomial regression fits
            intercept_2 = model_2.named_steps['ridge'].intercept_
            coef_2 = model_2.named_steps['ridge'].coef_
            equation_2 = f'y = {intercept_2:.3f} + {coef_2[1]:.3f}x + {coef_2[2]:.8f}x^2'

            # Store the coefficients and R² in the list
            coeff_list_2nd_degree.append({'Make': selected_make,
                                          'Model': selected_model,
                                          'Variant': selected_variant,
                                          'Fuel_Type': selected_fuel_type,
                                          'Year of Manufacture': int(name),
                                          'No_Of_Ownership': selected_no_of_owners,
                                          'Number of Data Points': len(group),
                                          'Intercept (2nd Degree)': intercept_2,
                                          'Coefficient 1 (2nd Degree)': coef_2[1],
                                          'Coefficient 2 (2nd Degree)': coef_2[2],
                                          'R² (2nd Degree)': r2_2,
                                          'Equation (2nd Degree)': equation_2})

    # Set plot title and labels for 2nd degree
    fig2.update_layout(
        title=f'2nd Degree Polynomial Fit ({selected_make} {selected_model} {selected_variant} {selected_fuel_type})',
        xaxis_title='Odometer Reading',
        yaxis_title='IBB Trade-In Price'
    )
    
    # Group data by year of manufacture for 3rd-degree fit
    grouped_data_3rd = odometer_filtered_data.groupby('Mfg_Year')

    # Create a list to store stratified samples for 2nd-degree fit
    stratified_samples = []

    # Loop through each group (year of manufacture) and create stratified samples for 3rd-degree fit
    for name, group in grouped_data_3rd:
        if len(group) > 2:  # Filter groups with less than 3 data points
            sampled_group = group.sample(n=min(num_samples, len(group)), random_state=1)
            stratified_samples.append((name, sampled_group))

    # Perform 3rd-degree polynomial regression on the selected samples
    for name, sample_data in stratified_samples:
        if len(sample_data) > 2:
            X_sampled = sample_data['Odometer_Reading'].values.reshape(-1, 1)
            y_sampled = sample_data['Tradein_MarketPrice'].values

            degree = 2
            poly_features_3 = PolynomialFeatures(degree=degree)
            regressor_3 = Ridge(alpha=0.001)
            model_3 = make_pipeline(poly_features_3, regressor_3)
            model_3.fit(X_sampled, y_sampled)

            # Make predictions over a range of values for a smoother curve
            X_range = np.linspace(X_sampled.min(), X_sampled.max(), 100).reshape(-1, 1)
            y_pred_3 = model_3.predict(X_range)

            # Calculate R² value
            y_pred_train_3 = model_3.predict(X_sampled)
            r2_3 = r2_score(y_sampled, y_pred_train_3)

            # Sort the values for plotting
            sorted_indices = np.argsort(X_range.flatten())
            X_sorted = X_range [sorted_indices]
            y_pred_3_sorted = y_pred_3[sorted_indices]

            # Plot the data and regression fits for 3rd degree
            fig3.add_trace(go.Scatter(x=X_sampled.flatten(), y=y_sampled, mode='markers', name=f'Sampled Points Year {name}'))
            fig3.add_trace(go.Scatter(x=X_sorted.flatten(), y=y_pred_3_sorted, mode='lines', name=f'{name} (3rd Degree fit)'))

            # Get the coefficients of the polynomial regression fits
            intercept_3 = model_3.named_steps['ridge'].intercept_
            coef_3 = model_3.named_steps['ridge'].coef_
            equation_3 = f'y = {intercept_3:.3f} + {coef_3[1]:.3f}x + {coef_3[2]:.8f}x^2'

            # Store the coefficients and R² in the list
            coeff_list_2nd_degree_sample_points.append({'Make': selected_make,
                                                      'Model': selected_model,
                                                      'Variant': selected_variant,
                                                      'Fuel_Type': selected_fuel_type,
                                                      'Year of Manufacture': int(name),
                                                      'No_Of_Ownership': selected_no_of_owners,
                                                      'Number of Data Points': len(sample_data),
                                                      'Intercept (sample points)': intercept_3,
                                                      'Coefficient 1 (sample points)': coef_3[1],
                                                      'Coefficient 2 (sample points)': coef_3[2],
                                                      'R² (sample points)': r2_3,
                                                      'Equation (sample points)': equation_3})
            

    # Set plot title and labels for sample points
    fig3.update_layout(
        title=f'sample points Polynomial Fit for All Years ({selected_make} {selected_model} {selected_variant} {selected_fuel_type})',
        xaxis_title='Odometer Reading',
        yaxis_title='IBB Trade-In Price'
    )

    # Convert the list of coefficients and R² values into DataFrames and display them
    coeff_df_2nd_degree = pd.DataFrame(coeff_list_2nd_degree)
    coeff_df_3rd_degree = pd.DataFrame(coeff_list_2nd_degree_sample_points)

    # Visual Comparison
    fig_comparison = go.Figure()
    
    grouped_data = odometer_filtered_data.groupby('Mfg_Year')

    # Plot full set fit
    for index, row in coeff_df_2nd_degree.iterrows():
        x_vals = np.linspace(odometer_filtered_data['Odometer_Reading'].min(), odometer_filtered_data['Odometer_Reading'].max(), 100)
        y_vals = row['Intercept (2nd Degree)'] + row['Coefficient 1 (2nd Degree)'] * x_vals + row['Coefficient 2 (2nd Degree)'] * (x_vals ** 2)
        fig_comparison.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name=f'Full Data {row["Year of Manufacture"]}'))
        x_data = odometer_filtered_data[odometer_filtered_data['Mfg_Year'] == row['Year of Manufacture']]
        # fig_comparison.add_trace(go.Scatter(x=x_data['Odometer_Reading'], y=x_data['Tradein_MarketPrice'], mode='markers', name=f'Full Data {row["Year of Manufacture"]}'))

    # group_data_sample = odometer_filtered_data.sample(n=min(num_samples, len(group)), random_state=42)
    group_data_sample = odometer_filtered_data

    # Plot sampled set fit
    
    for index, row in coeff_df_3rd_degree.iterrows():
        x_vals = np.linspace(group_data_sample['Odometer_Reading'].min(), group_data_sample['Odometer_Reading'].max(), 100)
        y_vals = row['Intercept (sample points)'] + row['Coefficient 1 (sample points)'] * x_vals + row['Coefficient 2 (sample points)'] * (x_vals ** 2)
        fig_comparison.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name=f'Sample Points {row["Year of Manufacture"]}', line=dict(dash='dash')))
        group_data_sample_1 = group_data_sample[group_data_sample['Mfg_Year']==row['Year of Manufacture']]
        group_data_sample_1 = group_data_sample_1.sample(n=min(num_samples, len(group_data_sample_1)), random_state=42)
        x_data = group_data_sample_1[group_data_sample_1['Mfg_Year']==row['Year of Manufacture']]
        fig_comparison.add_trace(go.Scatter(x=x_data['Odometer_Reading'], y=x_data['Tradein_MarketPrice'], mode='markers', name=f'Sample Data {row["Year of Manufacture"]}'))

    fig_comparison.update_layout(
        title=f'Comparison of Polynomial Fits for Full and Sample Data Points ({selected_make} {selected_model} {selected_variant} {selected_fuel_type})',
        xaxis_title='Odometer Reading',
        yaxis_title='IBB Trade-In Price'
    )

    return fig_comparison, coeff_df_2nd_degree, coeff_df_3rd_degree
