import streamlit as st
import pandas as  pd

st.set_page_config(
    page_title="AB Initio Model fit",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_sidebar(data, radio_button=None):
    st.sidebar.title("Filter Options")

    make_options = data['Make'].unique()
    selected_make = st.sidebar.selectbox('Select Make:', make_options)

    model_options = data[data['Make'] == selected_make]['Model'].unique()
    selected_model = st.sidebar.selectbox('Select Model:', model_options)

    variant_options = data[(data['Make'] == selected_make) & (data['Model'] == selected_model)]['Variant'].unique()
    selected_variant = st.sidebar.selectbox('Select Variant:', variant_options)

    fuel_type_options = data[(data['Make'] == selected_make) & (data['Model'] == selected_model) & (data['Variant'] == selected_variant)]['Fuel_Type'].unique()
    selected_fuel_type = st.sidebar.selectbox('Select Fuel Type:', fuel_type_options)

    # min_no_of_owners = data[(data['Make'] == selected_make) & (data['Model'] == selected_model) & (data['Variant'] == selected_variant) & (data['Fuel_Type'] == selected_fuel_type)]['No_Of_Ownership'].min()
    # max_no_of_owners = data[(data['Make'] == selected_make) & (data['Model'] == selected_model) & (data['Variant'] == selected_variant) & (data['Fuel_Type'] == selected_fuel_type)]['No_Of_Ownership'].max()

    selected_no_of_owners = st.sidebar.selectbox('Select No. of Owners:', [1, 2, 3, 'All'], index=3)

    if radio_button is None:
        return selected_make, selected_model, selected_variant, selected_fuel_type, selected_no_of_owners
    else:
        # Get unique manufacturing years for the selected make, model, variant, and fuel type
        Year_options = data[(data['Make'] == selected_make) & 
                            (data['Model'] == selected_model) & 
                            (data['Variant'] == selected_variant) & 
                            (data['Fuel_Type'] == selected_fuel_type)]['Mfg_Year'].unique()
        
        Year_options = list(map(str, Year_options)) + ['All']    
        
        # Allow the user to select a manufacturing year
        selected_Mfg_Year = st.sidebar.selectbox('Select Mfg Year:', Year_options, index=len(Year_options)-1)
        # selected_Mfg_Year = int(selected_Mfg_Year)  # Convert back to integer


        if selected_Mfg_Year == 'All':
            filtered_data_year = data[(data['Make'] == selected_make) &
                                (data['Model'] == selected_model) &
                                (data['Variant'] == selected_variant) &
                                (data['Fuel_Type'] == selected_fuel_type)]
        else:
            filtered_data_year = data[(data['Make'] == selected_make) &
                                (data['Model'] == selected_model) &
                                (data['Variant'] == selected_variant) &
                                (data['Fuel_Type'] == selected_fuel_type) &
                                (data['Mfg_Year'] == int(selected_Mfg_Year))]
                                
                                
        return selected_make, selected_model, selected_variant, selected_fuel_type, selected_no_of_owners, filtered_data_year


def configure_sidebar_odometer_reading(filtered_data: pd.DataFrame) -> tuple:
    st.sidebar.markdown("<h3>Select KM Range & No. of Points for Sample Data</h3>", unsafe_allow_html=True)
    # odometer_min = int(filtered_data['Odometer_Reading'].min())
    odometer_min = 0
    odometer_max = int(filtered_data['Odometer_Reading'].max())
    selected_odometer_range = st.sidebar.slider('Select Odometer Reading Range for 2nd Degree Fit (Sample Points):',
                                                min_value=odometer_min,
                                                max_value=odometer_max,
                                                value=(odometer_min, odometer_max))
    
    return selected_odometer_range


def configure_sidebar_number_of_samples(filtered_data: pd.DataFrame) -> tuple:
    
    num_samples = st.sidebar.slider('Select Number of Samples for 2nd Degree Fit (Sample Points):',
                                min_value=1,
                                max_value=30,
                                value=5)
    
    return num_samples


def radio_button():
    st.sidebar.title('Settings')
    option_set = st.sidebar.radio('Select fits:', ['Price vs Odo(by Owner) for all Year', 'future price', 'New Car Price v/s Ibb Price', 'Price vs Odometer Reading', 'Metric vs YOM', 'Price vs Odo(by Owner)', 'Metric vs Ownership'])
    
    return option_set



def get_filtered_year_data(odometer_filtered_data, selected_make, selected_model, selected_variant, selected_fuel_type, selected_no_of_owners ):
    
    # Get unique manufacturing years for the selected make, model, variant, and fuel type
    Year_options = odometer_filtered_data[(odometer_filtered_data['Make'] == selected_make) & 
                        (odometer_filtered_data['Model'] == selected_model) & 
                        (odometer_filtered_data['Variant'] == selected_variant) & 
                        (odometer_filtered_data['Fuel_Type'] == selected_fuel_type)]['Mfg_Year'].unique()
    
    Year_options.sort()
    print(Year_options)
    
    Year_options = list(map(str, Year_options)) + ['All']    
    
    # Allow the user to select a manufacturing year
    selected_Mfg_Year = st.sidebar.selectbox('Select Mfg Year:', Year_options, index=len(Year_options)-1)
    # selected_Mfg_Year = int(selected_Mfg_Year)  # Convert back to integer


    if selected_Mfg_Year == 'All':
        filtered_data_year = odometer_filtered_data[(odometer_filtered_data['Make'] == selected_make) &
                             (odometer_filtered_data['Model'] == selected_model) &
                             (odometer_filtered_data['Variant'] == selected_variant) &
                             (odometer_filtered_data['Fuel_Type'] == selected_fuel_type)]
    else:
        filtered_data_year = odometer_filtered_data[(odometer_filtered_data['Make'] == selected_make) &
                             (odometer_filtered_data['Model'] == selected_model) &
                             (odometer_filtered_data['Variant'] == selected_variant) &
                             (odometer_filtered_data['Fuel_Type'] == selected_fuel_type) &
                             (odometer_filtered_data['Mfg_Year'] == int(selected_Mfg_Year))]
                             
                             
    return filtered_data_year
    
    
    