import streamlit as st

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

    selected_no_of_owners = st.sidebar.selectbox('Select No. of Owners:', [1, 2, 3, 'All'])

    return selected_make, selected_model, selected_variant, selected_fuel_type, selected_no_of_owners

