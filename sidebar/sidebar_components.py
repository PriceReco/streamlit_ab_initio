import streamlit as st
import pandas as  pd

st.set_page_config(
    page_title="AB Initio Model fit",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_sidebar(data):
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

    return selected_make, selected_model, selected_variant, selected_fuel_type, selected_no_of_owners


def configure_sidebar(filtered_data: pd.DataFrame) -> tuple:
    st.sidebar.markdown("<h3>Select KM Range & No. of Points for Sample Data</h3>", unsafe_allow_html=True)
    # odometer_min = int(filtered_data['Odometer_Reading'].min())
    odometer_min = 0
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