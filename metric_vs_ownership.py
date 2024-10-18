import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from price_vs_odometer import polynomial_regression
from sidebar import sidebar_components
import numpy as np


def metric_vs_ownership_plot(coeff_list_2nd_degree, coeff_list_2nd_degree_sample_points):
    # Function to plot metrics vs. year of manufacture with trendlines
    def plot_metric_vs_year(df, metric, title, ylabel, outlier_points=None, include_outliers_label='No'):
        fig = go.Figure()

        # Exclude outlier points if include_outliers_label is 'No'
        if outlier_points is not None and include_outliers_label == 'No':
            df_without_outliers = df[~df.index.isin(outlier_points.index)]
        else:
            df_without_outliers = df
            
        print(df_without_outliers)

        # Plot the main data points
        fig.add_trace(go.Scatter(   
            x=df['No_Of_Ownership'],
            y=df[metric],
            mode='markers',
            name='Data Points',
            marker=dict(color='blue')
        ))

        # Highlight outlier points if available
        if outlier_points is not None and not outlier_points.empty:
            fig.add_trace(go.Scatter(
                x=outlier_points['No_Of_Ownership'],
                y=outlier_points[metric],
                mode='markers',
                name='Outliers',
                marker=dict(color='red', size=10)
            ))

        # Add linear trendline for data excluding outliers
        z = np.polyfit(df_without_outliers['No_Of_Ownership'], df_without_outliers[metric], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=df['No_Of_Ownership'],  # Use the original dataframe for x-axis values
            y=p(df['No_Of_Ownership']),
            mode='lines',
            name='Trendline',
            line=dict(color='green', dash='dash')
        ))

        # Calculate R-squared
        y_pred = p(df_without_outliers['No_Of_Ownership'])
        r_squared = 1 - (np.sum((df_without_outliers[metric] - y_pred) ** 2) / np.sum((df_without_outliers[metric] - np.mean(df_without_outliers[metric])) ** 2))

        # Prepare equation string
        equation = f"y = {z[0]:.4f}x + {z[1]:.4f}"

        # Set the layout of the figure
        fig.update_layout(
            title=f'{title} (Outliers Included: {include_outliers_label})<br>{equation}, R² = {r_squared:.4f}',
            xaxis_title='No_Of_Ownership',
            yaxis_title=ylabel,
            xaxis=dict(tickmode='array', tickvals=df['No_Of_Ownership'].unique(), ticktext=df['No_Of_Ownership'].unique()),
            showlegend=True
        )

        return fig, equation, r_squared

    # Usage
    coeff_full_df = pd.DataFrame(coeff_list_2nd_degree)
    coeff_sample_df = pd.DataFrame(coeff_list_2nd_degree_sample_points)

    outliers_full = coeff_full_df[(
        (coeff_full_df['Intercept (2nd Degree)'] > coeff_full_df['Intercept (2nd Degree)'].quantile(0.75) + 1.5 * (coeff_full_df['Intercept (2nd Degree)'].quantile(0.75) - coeff_full_df['Intercept (2nd Degree)'].quantile(0.25))) |
        (coeff_full_df['Coefficient 1 (2nd Degree)'] > coeff_full_df['Coefficient 1 (2nd Degree)'].quantile(0.75) + 1.5 * (coeff_full_df['Coefficient 1 (2nd Degree)'].quantile(0.75) - coeff_full_df['Coefficient 1 (2nd Degree)'].quantile(0.25))) |
        (coeff_full_df['Coefficient 2 (2nd Degree)'] > coeff_full_df['Coefficient 2 (2nd Degree)'].quantile(0.75) + 1.5 * (coeff_full_df['Coefficient 2 (2nd Degree)'].quantile(0.75) - coeff_full_df['Coefficient 2 (2nd Degree)'].quantile(0.25)))
    )]
    
    outliers_sample = coeff_sample_df[(
        (coeff_sample_df['Intercept (sample points)'] > coeff_sample_df['Intercept (sample points)'].quantile(0.75) + 1.5 * (coeff_sample_df['Intercept (sample points)'].quantile(0.75) - coeff_sample_df['Intercept (sample points)'].quantile(0.25))) |
        (coeff_sample_df['Coefficient 1 (sample points)'] > coeff_sample_df['Coefficient 1 (sample points)'].quantile(0.75) + 1.5 * (coeff_sample_df['Coefficient 1 (sample points)'].quantile(0.75) - coeff_sample_df['Coefficient 1 (sample points)'].quantile(0.25))) |
        (coeff_sample_df['Coefficient 2 (sample points)'] > coeff_sample_df['Coefficient 2 (sample points)'].quantile(0.75) + 1.5 * (coeff_sample_df['Coefficient 2 (sample points)'].quantile(0.75) - coeff_sample_df['Coefficient 2 (sample points)'].quantile(0.25)))
    )]
    
    # print(coeff_sample_df)

    # Create DataFrame to store equations and R-squared values
    results = []

    # Create figures for full data with and without outliers
    fig_intercept_full_with_outliers, eq_intercept_full_with_outliers, r2_intercept_full_with_outliers = plot_metric_vs_year(coeff_full_df, 'Intercept (2nd Degree)', 'Intercept vs No Of Ownership (Full Data)', 'Intercept (In Lakhs)', outliers_full, 'Yes')
    results.append({"Metric": "Intercept (Full Data, With Outliers)", "Equation": eq_intercept_full_with_outliers, "R²": r2_intercept_full_with_outliers})

    fig_coefficient1_full_with_outliers, eq_coefficient1_full_with_outliers, r2_coefficient1_full_with_outliers = plot_metric_vs_year(coeff_full_df, 'Coefficient 1 (2nd Degree)', 'Coefficient 1 vs No Of Ownership (Full Data)', 'Coefficient 1 (Rs)', outliers_full, 'Yes')
    results.append({"Metric": "Coefficient 1 (Full Data, With Outliers)", "Equation": eq_coefficient1_full_with_outliers, "R²": r2_coefficient1_full_with_outliers})

    fig_coefficient2_full_with_outliers, eq_coefficient2_full_with_outliers, r2_coefficient2_full_with_outliers = plot_metric_vs_year(coeff_full_df, 'Coefficient 2 (2nd Degree)', 'Coefficient 2 vs No Of Ownership (Full Data)', 'Coefficient 2 (Rs)', outliers_full, 'Yes')
    results.append({"Metric": "Coefficient 2 (Full Data, With Outliers)", "Equation": eq_coefficient2_full_with_outliers, "R²": r2_coefficient2_full_with_outliers})

    fig_intercept_full_without_outliers, eq_intercept_full_without_outliers, r2_intercept_full_without_outliers = plot_metric_vs_year(coeff_full_df, 'Intercept (2nd Degree)', 'Intercept vs No Of Ownership (Full Data)', 'Intercept (In Lakhs)', outliers_full, 'No')
    results.append({"Metric": "Intercept (Full Data, Without Outliers)", "Equation": eq_intercept_full_without_outliers, "R²": r2_intercept_full_without_outliers})

    fig_coefficient1_full_without_outliers, eq_coefficient1_full_without_outliers, r2_coefficient1_full_without_outliers = plot_metric_vs_year(coeff_full_df, 'Coefficient 1 (2nd Degree)', 'Coefficient 1 vs No Of Ownership (Full Data)', 'Coefficient 1 (Rs)', outliers_full, 'No')
    results.append({"Metric": "Coefficient 1 (Full Data, Without Outliers)", "Equation": eq_coefficient1_full_without_outliers, "R²": r2_coefficient1_full_without_outliers})

    fig_coefficient2_full_without_outliers, eq_coefficient2_full_without_outliers, r2_coefficient2_full_without_outliers = plot_metric_vs_year(coeff_full_df, 'Coefficient 2 (2nd Degree)', 'Coefficient 2 vs No Of Ownership (Full Data)', 'Coefficient 2 (Rs)', outliers_full, 'No')
    results.append({"Metric": "Coefficient 2 (Full Data, Without Outliers)", "Equation": eq_coefficient2_full_without_outliers, "R²": r2_coefficient2_full_without_outliers})

    fig_intercept_sample_with_outliers, eq_intercept_sample_with_outliers, r2_intercept_sample_with_outliers = plot_metric_vs_year(coeff_sample_df, 'Intercept (sample points)', 'Intercept vs No Of Ownership (Sample Data)', 'Intercept (In Lakhs)', outliers_sample, 'Yes')
    results.append({"Metric": "Intercept (Sample Data, With Outliers)", "Equation": eq_intercept_sample_with_outliers, "R²": r2_intercept_sample_with_outliers})

    fig_coefficient1_sample_with_outliers, eq_coefficient1_sample_with_outliers, r2_coefficient1_sample_with_outliers = plot_metric_vs_year(coeff_sample_df, 'Coefficient 1 (sample points)', 'Coefficient 1 vs No Of Ownership (Sample Data)', 'Coefficient 1 (Rs)', outliers_sample, 'Yes')
    results.append({"Metric": "Coefficient 1 (Sample Data, With Outliers)", "Equation": eq_coefficient1_sample_with_outliers, "R²": r2_coefficient1_sample_with_outliers})

    fig_coefficient2_sample_with_outliers, eq_coefficient2_sample_with_outliers, r2_coefficient2_sample_with_outliers = plot_metric_vs_year(coeff_sample_df, 'Coefficient 2 (sample points)', 'Coefficient 2 vs No Of Ownership (Sample Data)', 'Coefficient 2 (Rs)', outliers_sample, 'Yes')
    results.append({"Metric": "Coefficient 2 (Sample Data, With Outliers)", "Equation": eq_coefficient2_sample_with_outliers, "R²": r2_coefficient2_sample_with_outliers})

    fig_intercept_sample_without_outliers, eq_intercept_sample_without_outliers, r2_intercept_sample_without_outliers = plot_metric_vs_year(coeff_sample_df, 'Intercept (sample points)', 'Intercept vs No Of Ownership (Sample Data)', 'Intercept (In Lakhs)', outliers_sample, 'No')
    results.append({"Metric": "Intercept (Sample Data, Without Outliers)", "Equation": eq_intercept_sample_without_outliers, "R²": r2_intercept_sample_without_outliers})

    fig_coefficient1_sample_without_outliers, eq_coefficient1_sample_without_outliers, r2_coefficient1_sample_without_outliers = plot_metric_vs_year(coeff_sample_df, 'Coefficient 1 (sample points)', 'Coefficient 1 vs No Of Ownership (Sample Data)', 'Coefficient 1 (Rs)', outliers_sample, 'No')
    results.append({"Metric": "Coefficient 1 (Sample Data, Without Outliers)", "Equation": eq_coefficient1_sample_without_outliers, "R²": r2_coefficient1_sample_without_outliers})

    fig_coefficient2_sample_without_outliers, eq_coefficient2_sample_without_outliers, r2_coefficient2_sample_without_outliers = plot_metric_vs_year(coeff_sample_df, 'Coefficient 2 (sample points)', 'Coefficient 2 vs No Of Ownership (Sample Data)', 'Coefficient 2 (Rs)', outliers_sample, 'No')
    results.append({"Metric": "Coefficient 2 (Sample Data, Without Outliers)", "Equation": eq_coefficient2_sample_without_outliers, "R²": r2_coefficient2_sample_without_outliers})


    # results = create_plots(coeff_full_df, coeff_sample_df, outliers_full, outliers_sample)
    
    st.sidebar.title('Metric vs Ownership')

    # Outliers option: With or Without Outliers
    outliers_option = st.sidebar.radio('Select Outliers Option:', ['With Outliers', 'Without Outliers'])
    
    # Plot type selection: Intercept, Coeff1, or Coeff2 vs YOM
    plot_type = st.sidebar.radio('Select Plot Type:', ['Intercept vs Ownership', 'Coeff1 vs Ownership', 'Coeff2 vs Ownership'])
    
    # fig_full = fig_intercept_full_with_outliers
    
    # Determine the correct figures based on user selection
    if outliers_option == 'With Outliers':
        if plot_type == 'Intercept vs Ownership':
            fig_full = fig_intercept_full_with_outliers
            fig_sample = fig_intercept_sample_with_outliers
        elif plot_type == 'Coeff1 vs Ownership':
            fig_full = fig_coefficient1_full_with_outliers
            fig_sample = fig_coefficient1_sample_with_outliers
        elif plot_type == 'Coeff2 vs Ownership':
            fig_full = fig_coefficient2_full_with_outliers
            fig_sample = fig_coefficient2_sample_with_outliers
    else: 
        if plot_type == 'Intercept vs Ownership':
            fig_full = fig_intercept_full_without_outliers
            fig_sample = fig_intercept_sample_without_outliers
        elif plot_type == 'Coeff1 vs Ownership':
            fig_full = fig_coefficient1_full_without_outliers
            fig_sample = fig_coefficient1_sample_without_outliers
        elif plot_type == 'Coeff2 vs Ownership':
            fig_full = fig_coefficient2_full_without_outliers
            fig_sample = fig_coefficient2_sample_without_outliers


    return fig_full, fig_sample


