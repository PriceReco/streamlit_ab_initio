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
from metric_vs_ownership import metric_vs_ownership_plot
from price_vs_odometer_for_all_ownership_manf import plot_price_vs_odo_by_owner_for_all_manf
from future_price import polynomial_regression_future_price
from metric_vs_ownership_with_no_graphs import metric_vs_yom_no_plot
from new_car_vs_ibb_price import new_price_vs_ibb_price_ctdms_price


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


def main():
    # data = utils.load_data('data/Dataset_15Jan 2.xlsx')
    data = utils.load_data('data/combined_data.xlsx')
    selected_make, selected_model, selected_variant, selected_fuel_type, selected_no_of_owners = sidebar_components.create_sidebar(data)
    filtered_data = filter_data(data, selected_make, selected_model, selected_variant, selected_fuel_type, selected_no_of_owners)
    filtered_data_no_outliers = utils.remove_outliers(filtered_data, 'Odometer_Reading')
    # selected_odometer_range = sidebar_components.configure_sidebar_odometer_reading(filtered_data_no_outliers)
    # odometer_filtered_data = filtered_data[(filtered_data['Odometer_Reading'] >= selected_odometer_range[0]) &
    #                                                    (filtered_data['Odometer_Reading'] <= selected_odometer_range[1])]
    
    option_set = sidebar_components.radio_button()
    odometer_filtered_data = filtered_data
    
    if  option_set == 'Price vs Odometer Reading':

        selected_odometer_range = sidebar_components.configure_sidebar_odometer_reading(filtered_data_no_outliers)
        odometer_filtered_data = filtered_data[(filtered_data['Odometer_Reading'] >= selected_odometer_range[0]) &
                                                       (filtered_data['Odometer_Reading'] <= selected_odometer_range[1])]
    
        
        num_samples = sidebar_components.configure_sidebar_number_of_samples(filtered_data_no_outliers)
        fig_comparison, coeff_df_2nd_degree, coeff_list_2nd_degree_sample_points = polynomial_regression(odometer_filtered_data, num_samples, selected_make, selected_model, selected_variant, selected_fuel_type, selected_no_of_owners)

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
        selected_odometer_range = sidebar_components.configure_sidebar_odometer_reading(filtered_data_no_outliers)
        odometer_filtered_data = filtered_data[(filtered_data['Odometer_Reading'] >= selected_odometer_range[0]) &
                                                (filtered_data['Odometer_Reading'] <= selected_odometer_range[1])]
    
        
        num_samples = sidebar_components.configure_sidebar_number_of_samples(filtered_data_no_outliers)
        fig, coeff_list_2nd_degree_wrt_ownership, coeff_df_2nd_degree_sample_wrt_ownership  = plot_price_vs_odo_by_owner(odometer_filtered_data, num_samples, selected_make, selected_model, selected_variant, selected_fuel_type, selected_no_of_owners)
        st.plotly_chart(fig)

        st.write("#### 2nd Degree Polynomial Regression Coefficients and R² w.r.t Ownership:")
        # st.dataframe(coeff_df_2nd_degree[coeff_df_2nd_degree['Year of Manufacture'] == selected_Mfg_Year])
        
        st.dataframe(coeff_list_2nd_degree_wrt_ownership)
        st.write("#### Sample Points Polynomial Regression Coefficients and R² w.r.t Ownership:")
        st.dataframe(coeff_df_2nd_degree_sample_wrt_ownership)
    

    if option_set == 'Metric vs YOM':
        
        selected_odometer_range = sidebar_components.configure_sidebar_odometer_reading(filtered_data_no_outliers)
        odometer_filtered_data = filtered_data[(filtered_data['Odometer_Reading'] >= selected_odometer_range[0]) &
                                                (filtered_data['Odometer_Reading'] <= selected_odometer_range[1])]
    
        num_samples = sidebar_components.configure_sidebar_number_of_samples(filtered_data_no_outliers)

        fig_comparison, coeff_df_2nd_degree, coeff_list_2nd_degree_sample_points = polynomial_regression(odometer_filtered_data, num_samples, selected_make, selected_model, selected_variant, selected_fuel_type, selected_no_of_owners)

        fig_full, fig_sample, equation_arr = metric_vs_yom_plot(coeff_df_2nd_degree, coeff_list_2nd_degree_sample_points)
        # Display the selected plots side by side
            
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_full)
        with col2:
            st.plotly_chart(fig_sample)
            
                        
    if option_set == 'Metric vs Ownership': 
        
        selected_odometer_range = sidebar_components.configure_sidebar_odometer_reading(filtered_data_no_outliers)

        odometer_filtered_data = filtered_data[(filtered_data['Odometer_Reading'] >= selected_odometer_range[0]) &
                                                (filtered_data['Odometer_Reading'] <= selected_odometer_range[1])]


        num_samples = sidebar_components.configure_sidebar_number_of_samples(filtered_data_no_outliers)

        fig, coeff_list_2nd_degree_wrt_ownership, coeff_df_2nd_degree_sample_wrt_ownership  = plot_price_vs_odo_by_owner(odometer_filtered_data, num_samples, selected_make, selected_model, selected_variant, selected_fuel_type, selected_no_of_owners)

        fig_full_1, fig_sample_1 = metric_vs_ownership_plot(coeff_list_2nd_degree_wrt_ownership, coeff_df_2nd_degree_sample_wrt_ownership)
        # Display the selected plots side by side
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_full_1)
        with col2:
            st.plotly_chart(fig_sample_1)
            
    
    if option_set == 'Price vs Odo(by Owner) for all Year':
        
        selected_odometer_range = sidebar_components.configure_sidebar_odometer_reading(filtered_data_no_outliers)

        
        # num_samples = sidebar_components.configure_sidebar_number_of_samples(filtered_data_no_outliers)
        odometer_filtered_data = sidebar_components.get_filtered_year_data(odometer_filtered_data, selected_make, selected_model, selected_variant, selected_fuel_type, selected_no_of_owners)        

        odometer_filtered_data = odometer_filtered_data[(odometer_filtered_data['Odometer_Reading'] >= selected_odometer_range[0]) &
                                                (odometer_filtered_data['Odometer_Reading'] <= selected_odometer_range[1])]                
        
        fig, coeff_list_2nd_degree_wrt_ownership  = plot_price_vs_odo_by_owner_for_all_manf(odometer_filtered_data, selected_make, selected_model, selected_variant, selected_fuel_type, selected_no_of_owners)
        st.plotly_chart(fig)

        st.write("##### 2nd Degree Polynomial Regression Coefficients and R² w.r.t Ownership:")
        # st.dataframe(coeff_df_2nd_degree[coeff_df_2nd_degree['Year of Manufacture'] == selected_Mfg_Year])        
        st.dataframe(coeff_list_2nd_degree_wrt_ownership)
    
    
    if  option_set == 'Future Price':
        
        selected_odometer_range = sidebar_components.configure_sidebar_odometer_reading(filtered_data_no_outliers)

        odometer_filtered_data = filtered_data[(filtered_data['Odometer_Reading'] >= selected_odometer_range[0]) &
                                                (filtered_data['Odometer_Reading'] <= selected_odometer_range[1])]

        num_samples = 5
        fig_comparison, coeff_df_2nd_degree, coeff_list_2nd_degree_sample_points = polynomial_regression(odometer_filtered_data, num_samples, selected_make, selected_model, selected_variant, selected_fuel_type, selected_no_of_owners)
                
        equation_arr = metric_vs_yom_no_plot(coeff_df_2nd_degree, coeff_list_2nd_degree_sample_points)
        
        fig_comparison, coeff_df_2nd_degree, coeff_df_2nd_degree_future_price = polynomial_regression_future_price(odometer_filtered_data, num_samples, selected_make, selected_model, selected_variant, selected_fuel_type, selected_no_of_owners, equation_arr)
        
        st.plotly_chart(fig_comparison)
            
        st.write("##### 2nd Degree Polynomial Regression Coefficients and R²:")
        st.dataframe(coeff_df_2nd_degree)
        
        st.write("##### 2nd Degree Polynomial Regression Coefficients and R² For Future Price:")
        st.dataframe(coeff_df_2nd_degree_future_price)
        
    
    if  option_set == 'New Car Price v/s Ibb Price':
        
        new_price_vs_ibb_price_ctdms_price()
        

if __name__ == '__main__':
    main()


 
