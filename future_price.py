# polynomial_regression.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
import streamlit as st

# def polynomial_regression_future_price(odometer_filtered_data, num_samples, selected_make, selected_model, selected_variant, selected_fuel_type, selected_no_of_owners, equation_arr):
#     # Group data by year of manufacture
    
#     grouped_data = odometer_filtered_data.groupby('Mfg_Year')
#     print(equation_arr)
#     # Initialize the plots
#     fig2 = go.Figure()
#     fig3 = go.Figure()

#     # Create an empty list to store dictionaries of coefficients for both degrees
#     coeff_list_2nd_degree = []
#     # coeff_list_2nd_degree_sample_points = []

#     # Loop through each group (year of manufacture) and plot the data and polynomial regression fits for 2nd degree
#     for name, group in grouped_data:
#         if len(group) > 2:  # Filter groups with less than 3 data points
#             # Split the data into independent variables (X) and target variable (y)
#             X = group['Odometer_Reading'].values.reshape(-1, 1)
#             y = group['Tradein_MarketPrice'].values

#             # 2nd-degree polynomial regression using all points
#             degree = 2
#             alpha = 0.01  # Ridge regularization strength
#             poly_features_2 = PolynomialFeatures(degree=degree)
#             regressor_2 = Ridge(alpha=alpha)
#             model_2 = make_pipeline(poly_features_2, regressor_2)
#             model_2.fit(X, y)

#             # Make predictions over a range of values for a smoother curve
#             X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
#             y_pred_2 = model_2.predict(X_range)

#             # Calculate R² value
#             y_pred_train_2 = model_2.predict(X)
#             r2_2 = r2_score(y, y_pred_train_2)

#             # Sort the values for plotting
#             sorted_indices = np.argsort(X_range.flatten())
#             X_sorted = X_range[sorted_indices]
#             y_pred_2_sorted = y_pred_2[sorted_indices]

#             # Plot the data and regression fits for 2nd degree
#             fig2.add_trace(go.Scatter(x=X.flatten(), y=y, mode='markers', name=f'Year {name}'))
#             fig2.add_trace(go.Scatter(x=X_sorted.flatten(), y=y_pred_2_sorted, mode='lines', name=f'{name} (2nd Degree fit)'))

#             # Get the coefficients of the polynomial regression fits
#             intercept_2 = model_2.named_steps['ridge'].intercept_
#             coef_2 = model_2.named_steps['ridge'].coef_
#             equation_2 = f'y = {intercept_2:.3f} + {coef_2[1]:.3f}x + {coef_2[2]:.8f}x^2'

#             # Store the coefficients and R² in the list
#             coeff_list_2nd_degree.append({'Make': selected_make,
#                                           'Model': selected_model,
#                                           'Variant': selected_variant,
#                                           'Fuel_Type': selected_fuel_type,
#                                           'Year of Manufacture': int(name),
#                                           'No_Of_Ownership': selected_no_of_owners,
#                                           'Number of Data Points': len(group),
#                                           'Intercept (2nd Degree)': intercept_2,
#                                           'Coefficient 1 (2nd Degree)': coef_2[1],
#                                           'Coefficient 2 (2nd Degree)': coef_2[2],
#                                           'R² (2nd Degree)': r2_2,
#                                           'Equation (2nd Degree)': equation_2})

#     # Set plot title and labels for 2nd degree
#     fig2.update_layout(
#         title=f'2nd Degree Polynomial Fit ({selected_make} {selected_model} {selected_variant} {selected_fuel_type})',
#         xaxis_title='Odometer Reading',
#         yaxis_title='IBB Trade-In Price'
#     )
    
#     # Convert the list of coefficients and R² values into DataFrames and display them
#     coeff_df_2nd_degree = pd.DataFrame(coeff_list_2nd_degree)

#     # Visual Comparison
#     fig_comparison = go.Figure()
    
#     grouped_data = odometer_filtered_data.groupby('Mfg_Year')

#     # Plot full set fit
#     for index, row in coeff_df_2nd_degree.iterrows():
#         x_vals = np.linspace(odometer_filtered_data['Odometer_Reading'].min(), odometer_filtered_data['Odometer_Reading'].max(), 100)
#         y_vals = row['Intercept (2nd Degree)'] + row['Coefficient 1 (2nd Degree)'] * x_vals + row['Coefficient 2 (2nd Degree)'] * (x_vals ** 2)
#         fig_comparison.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name=f'Full Data {row["Year of Manufacture"]}'))
#         x_data = odometer_filtered_data[odometer_filtered_data['Mfg_Year'] == row['Year of Manufacture']]
#         # fig_comparison.add_trace(go.Scatter(x=x_data['Odometer_Reading'], y=x_data['Tradein_MarketPrice'], mode='markers', name=f'Full Data {row["Year of Manufacture"]}'))

#     fig_comparison.update_layout(
#         title=f'Comparison of Polynomial Fits for Full and Sample Data Points ({selected_make} {selected_model} {selected_variant} {selected_fuel_type})',
#         xaxis_title='Odometer Reading',
#         yaxis_title='IBB Trade-In Price'
#     )
    
#     return fig_comparison, coeff_df_2nd_degree

def polynomial_regression_future_price(odometer_filtered_data, num_samples, selected_make, selected_model, selected_variant, selected_fuel_type, selected_no_of_owners, equation_arr):
    
    def calculate_future_price(equation_arr, manf):
        intercept = equation_arr[0] * manf + equation_arr[1]
        coef_1 = equation_arr[2] * manf + equation_arr[3]
        coef_2 = equation_arr[4] * manf + equation_arr[5]
        
        print(intercept)
        print(coef_1)
        print(coef_2)
        
        return intercept, coef_1, coef_2
    
    intercept, coef_1, coef_2 = calculate_future_price(equation_arr, manf=2020)
    
    # Group data by year of manufacture

    grouped_data = odometer_filtered_data.groupby('Mfg_Year')
    print(equation_arr)
    # Initialize the plots
    fig2 = go.Figure()
    fig3 = go.Figure()

    # Create an empty list to store dictionaries of coefficients for both degrees
    coeff_list_2nd_degree = []
    # coeff_list_2nd_degree_sample_points = []

    odo = [0, 12235, 18247, 23482, 28562, 34829, 39582, 45832, 51829, 56839, 61274, 68572, 73625, 78395, 84636, 88263, 92857, 97563, 102854, 108274, 115272, 118738, 125632, 129762, 135263, 140483, 146382, 151293, 156325, 160284]


    # 1576783 + 0.134x - 0.0963x^2
    

    # Loop through each group (year of manufacture) and plot the data and polynomial regression fits for 2nd degree
    for name, group in grouped_data:
        if len(group) > 2:  # Filter groups with less than 3 data points
            # Split the data into independent variables (X) and target variable (y)
            X = group['Odometer_Reading'].values.reshape(-1, 1)
            y = group['Tradein_MarketPrice'].values
            
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
    
    # Convert the list of coefficients and R² values into DataFrames and display them
    coeff_df_2nd_degree = pd.DataFrame(coeff_list_2nd_degree)

    # Visual Comparison
    fig_comparison = go.Figure()
    grouped_data = odometer_filtered_data.groupby('Mfg_Year')

    # Plot full set fit
    for index, row in coeff_df_2nd_degree.iterrows():
        x_vals = np.linspace(odometer_filtered_data['Odometer_Reading'].min(), odometer_filtered_data['Odometer_Reading'].max(), 100)
        y_vals = (row['Intercept (2nd Degree)'] + row['Coefficient 1 (2nd Degree)'] * x_vals + row['Coefficient 2 (2nd Degree)'] * (x_vals ** 2)) / 100000
        fig_comparison.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name=f'Full Data {row["Year of Manufacture"]}'))
        # x_data = odometer_filtered_data[odometer_filtered_data['Mfg_Year'] == row['Year of Manufacture']]
        # fig_comparison.add_trace(go.Scatter(x=x_data['Odometer_Reading'], y=x_data['Tradein_MarketPrice'], mode='markers', name=f'Full Data {row["Year of Manufacture"]}'))

    print(x_vals[0])
    
    print(len(x_vals))
    
    print(y_vals)
    
    coeff_df_2nd_degree['Year of Manufacture'].value_counts()
    
    
    manf = coeff_df_2nd_degree['Year of Manufacture'].value_counts().index
    manf = list(manf)
    # print(manf)
    # manf.append(manf[-1] + 1)
    coeff_list_2nd_degree_future_price = []
    
    for i in range(3):            
        
        year = manf[-1] + i + 1

        intercept, coef_1, coef_2 = calculate_future_price(equation_arr, (manf[-1] + i + 1))

        equation_2 = f'y = {intercept:.3f} + {coef_1:.3f}x + {coef_2:.8f}x^2'
        
        # Store the coefficients and R² in the list
        coeff_list_2nd_degree_future_price.append({'Make': selected_make,
                                        'Model': selected_model,
                                        'Variant': selected_variant,
                                        'Fuel_Type': selected_fuel_type,
                                        'Year of Manufacture': int(year),
                                        'No_Of_Ownership': selected_no_of_owners,
                                        'Number of Data Points': len(group),
                                        'Intercept (2nd Degree)': intercept,
                                        'Coefficient 1 (2nd Degree)': coef_1,
                                        'Coefficient 2 (2nd Degree)': coef_2,
                                        # 'R² (2nd Degree)': r2_2,
                                        'Equation (2nd Degree)': equation_2})
        
        y = []

        for i in x_vals:
            # print(i)
            price = (intercept + coef_1 * i + coef_2 * (i **2)) / 100000
            # price = price / 100000
            y.append(price)
        
                    
        print(y)
        # print(len(y))
        df_future_price = pd.DataFrame()
    
        # df_future_price['odometer_reading'] = odo
        # df_future_price['future_price'] = y
        
        # print(df_future_price)
        
        # x_vals = np.linspace(df_future_price['odometer_reading'].min(), df_future_price['odometer_reading'].max(), 100)

        # year = manf[-1] + i + 1
        # print(year)
        
        fig_comparison.add_trace(go.Scatter(x=x_vals, y=y, mode='lines', line=dict(dash='dash'), name=f'Future Price {(year)}'))
        
        fig_comparison.update_layout(
                title=f'2nd Degree Polynomial Fit ({selected_make} {selected_model} {selected_variant} {selected_fuel_type})',
                xaxis_title='Odometer Reading',
                yaxis_title='IBB Trade-In Price (in Lakhs)',
                yaxis=dict(
                    # tickformat=".1f",
                    ticksuffix='L'
                    )
        )
        
        
    coeff_df_2nd_degree_future = pd.DataFrame(coeff_list_2nd_degree_future_price)


    return fig_comparison, coeff_df_2nd_degree, coeff_df_2nd_degree_future


