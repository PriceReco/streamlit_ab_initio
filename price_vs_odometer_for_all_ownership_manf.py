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


def plot_price_vs_odo_by_owner_for_all_manf(odometer_filtered_data, selected_make, selected_model, selected_variant, selected_fuel_type, selected_no_of_owners):
    
    # # Get unique manufacturing years for the selected make, model, variant, and fuel type
    # Year_options = odometer_filtered_data[(odometer_filtered_data['Make'] == selected_make) & 
    #                     (odometer_filtered_data['Model'] == selected_model) & 
    #                     (odometer_filtered_data['Variant'] == selected_variant) & 
    #                     (odometer_filtered_data['Fuel_Type'] == selected_fuel_type)]['Mfg_Year'].unique()
    
    # Year_options = list(map(str, Year_options)) + ['All']    
    
    # # Allow the user to select a manufacturing year on the main screen
    # selected_Mfg_Year = st.selectbox('Select Manufacturing Year:', Year_options, index=len(Year_options)-1)
    # # st.write(f"Selected Manufacturing Year: {selected_Mfg_Year}")

    # if selected_Mfg_Year == 'All':
    #     filtered_data_year = odometer_filtered_data[(odometer_filtered_data['Make'] == selected_make) &
    #                          (odometer_filtered_data['Model'] == selected_model) &
    #                          (odometer_filtered_data['Variant'] == selected_variant) &
    #                          (odometer_filtered_data['Fuel_Type'] == selected_fuel_type)]
    # else:
    #     filtered_data_year = odometer_filtered_data[(odometer_filtered_data['Make'] == selected_make) &
    #                          (odometer_filtered_data['Model'] == selected_model) &
    #                          (odometer_filtered_data['Variant'] == selected_variant) &
    #                          (odometer_filtered_data['Fuel_Type'] == selected_fuel_type) &
    #                          (odometer_filtered_data['Mfg_Year'] == int(selected_Mfg_Year))]
        
                         
                            
    # Group data by year of manufacture
    grouped_data = odometer_filtered_data.groupby('Mfg_Year')

    # Initialize the plots
    fig2 = go.Figure()

    # Create an empty list to store dictionaries of coefficients for both degrees
    coeff_list_2nd_degree = []
    # coeff_list_2nd_degree_sample_points = []
    
    # Group data by number of owners
    # grouped_data = grouped_data.groupby('No_Of_Ownership')
    # print(grouped_data)

    # Loop through each group (year of manufacture) and plot the data and polynomial regression fits for 2nd degree
    for name, group in grouped_data:
        print(name)
        print(group)
        
        if len(group) > 2:  # Filter groups with less than 3 data points
            # Split the data into independent variables (X) and target variable (y)
            
            grouped_data = group.groupby('No_Of_Ownership')
            
            for num_owner, subgroup in grouped_data:
                print(num_owner)
                print(subgroup)
                
            
                X = subgroup['Odometer_Reading'].values.reshape(-1, 1)
                y = subgroup['Tradein_MarketPrice'].values

                # 2nd-degree polynomial regression using all points
                degree = 2
                alpha = 0.01  # Ridge regularization strength
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
                                            'No_Of_Ownership': num_owner,
                                            'Number of Data Points': len(group),
                                            'Intercept (2nd Degree)': intercept_2,
                                            'Coefficient 1 (2nd Degree)': coef_2[1],
                                            'Coefficient 2 (2nd Degree)': coef_2[2],
                                            'R² (2nd Degree)': r2_2,
                                            'Equation (2nd Degree)': equation_2})

        # Set plot title and labels for 2nd degree
        fig2.update_layout(
            title=f'2nd Degree Polynomial Fit ({selected_make} {selected_model} {selected_variant} {selected_fuel_type})',
            # xaxis_title='Odometer Reading',
            yaxis_title='IBB Trade-In Price'
        )
        
    # Convert the list of coefficients and R² values into DataFrames and display them
    coeff_df_2nd_degree = pd.DataFrame(coeff_list_2nd_degree)

    # Visual Comparison
    fig_comparison = go.Figure()
    
    grouped_data = odometer_filtered_data.groupby('Mfg_Year')


    ownership_colors = ['#1f77b4', '#3D9970', '#FF851B']

    for index, row in coeff_df_2nd_degree.iterrows():
        
        x_vals = np.linspace(odometer_filtered_data['Odometer_Reading'].min(), odometer_filtered_data['Odometer_Reading'].max(), 100)
        
        # Calculate y values and convert to lakhs
        y_vals = (row['Intercept (2nd Degree)'] + 
                row['Coefficient 1 (2nd Degree)'] * x_vals + 
                row['Coefficient 2 (2nd Degree)'] * (x_vals ** 2)) / 100000  # Convert to lakhs
        
        # Get the color for the current ownership
        color = ownership_colors[row['No_Of_Ownership'] % len(ownership_colors)]  # Cycle through the colors for each ownership
        
        # Add the line trace with color
        fig_comparison.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name=f'Owners: {row['No_Of_Ownership']} Year {row["Year of Manufacture"]}', line=dict(color=color)))
        
        # Filter data for the current year of manufacture
        x_data = odometer_filtered_data[odometer_filtered_data['Mfg_Year'] == row['Year of Manufacture']]


    fig_comparison.update_layout(
        title_text=f'Comparison of Polynomial Fits for Full Data Points ({selected_make} {selected_model} {selected_variant} {selected_fuel_type})',
        xaxis_title='Odometer Reading',
        yaxis_title='IBB Trade-In Price (in Lakhs)',  # Updated y-axis title
        yaxis=dict(
            ticksuffix='L'  # This ensures 'L' is added even to values without decimal places
        )
    )
    
    return fig_comparison, coeff_df_2nd_degree
