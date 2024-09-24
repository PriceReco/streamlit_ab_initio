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
from price_vs_odometer_by_owner import plot_price_vs_odo_by_owner
from metric_vs_yom import metric_vs_yom_plot


def filter_data(data: pd.DataFrame, selected_make: str, selected_model: str, selected_variant: str, selected_fuel_type: str, selected_no_of_owners: str) -> pd.DataFrame:
    if selected_no_of_owners == 'All':
        filtered_data = data[(data['Make'] == selected_make) &
                             (data['Model'] == selected_model) &
                             (data['Variant'] == selected_variant) &
                             (data['Fuel_Type'] == selected_fuel_type)]
    else:
        filtered_data = data[(data['Make'] == selected_make) &
                             (data['Model'] == selected_model) &
                             (data['Variant'] == selected_variant) &
                             (data['Fuel_Type'] == selected_fuel_type) &
                             (data['No_Of_Ownership'] == int(selected_no_of_owners))]
    return filtered_data


def configure_sidebar(filtered_data: pd.DataFrame) -> tuple:
    st.sidebar.markdown("<h3>Select KM Range & No. of Points for Sample Data</h3>", unsafe_allow_html=True)
    odometer_min = int(filtered_data['Odometer_Reading'].min())
    odometer_max = int(filtered_data['Odometer_Reading'].max())
    selected_odometer_range = st.sidebar.slider('Select Odometer Reading Range for 2nd Degree Fit (Sample Points):',
                                                min_value=odometer_min,
                                                max_value=odometer_max,
                                                value=(odometer_min, odometer_max))
    
    num_samples = st.sidebar.slider('Select Number of Samples for 2nd Degree Fit (Sample Points):',
                                    min_value=1,
                                    max_value=30,
                                    value=5)
    return selected_odometer_range, num_samples


def main():
    data = utils.load_data('data/Dataset_15Jan 2.xlsx')
    selected_make, selected_model, selected_variant, selected_fuel_type, selected_no_of_owners = sidebar_components.create_sidebar(data)
    filtered_data = filter_data(data, selected_make, selected_model, selected_variant, selected_fuel_type, selected_no_of_owners)
    # filtered_data_no_outliers = utils.remove_outliers(filtered_data, 'Odometer_Reading')
    selected_odometer_range, num_samples = configure_sidebar(filtered_data)
    odometer_filtered_data = filtered_data[(filtered_data['Odometer_Reading'] >= selected_odometer_range[0]) &
                                                       (filtered_data['Odometer_Reading'] <= selected_odometer_range[1])]
    
    fig_comparison, coeff_df_2nd_degree, coeff_list_2nd_degree_sample_points = polynomial_regression(odometer_filtered_data, num_samples, selected_make, selected_model, selected_variant, selected_fuel_type, selected_no_of_owners)
    st.sidebar.title('Settings')
    option_set = st.sidebar.radio('Select fits:', ['Price vs Odometer Reading', 'Price vs Odo(by Owner)', 'Metric vs YOM'])
    
    if  option_set == 'Price vs Odometer Reading':
        st.sidebar.title('Price vs Odometer Reading')
        show_plot = st.sidebar.checkbox('Show Plot', value=True)
        show_equation = st.sidebar.checkbox('Show Equation', value=True)
        
        if show_plot:
            st.plotly_chart(fig_comparison)
        if show_equation:
            st.write("### 2nd Degree Polynomial Regression Coefficients and R²:")
            st.dataframe(coeff_df_2nd_degree)
            st.write("### Sample Points Polynomial Regression Coefficients and R²:")
            st.dataframe(coeff_list_2nd_degree_sample_points)
            
    if option_set == 'Price vs Odo(by Owner)':
        fig, coeff_list_2nd_degree_wrt_ownership, coeff_df_2nd_degree_sample_wrt_ownership  = plot_price_vs_odo_by_owner(odometer_filtered_data, num_samples, selected_make, selected_model, selected_variant, selected_fuel_type, selected_no_of_owners)
        st.plotly_chart(fig)

        st.write("#### 2nd Degree Polynomial Regression Coefficients and R² w.r.t Ownership:")
        # st.dataframe(coeff_df_2nd_degree[coeff_df_2nd_degree['Year of Manufacture'] == selected_Mfg_Year])
        
        st.dataframe(coeff_list_2nd_degree_wrt_ownership)
        st.write("#### Sample Points Polynomial Regression Coefficients and R² w.r.t Ownership:")
        st.dataframe(coeff_df_2nd_degree_sample_wrt_ownership)
    

    if option_set == 'Metric vs YOM':    

        fig_full, fig_sample = metric_vs_yom_plot(coeff_df_2nd_degree, coeff_list_2nd_degree_sample_points)
        # Display the selected plots side by side
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_full)
        with col2:
            st.plotly_chart(fig_sample)
        
        
if __name__ == '__main__':
    main()
    

