import pandas as pd
import plotly.graph_objs as go
import streamlit as st


def new_price_vs_ibb_price_ctdms_price():
    # Load the data
    df = pd.read_excel('data/toyota_on_road_price_for_crysta_2016-2020.xlsx')
    
    years = df['Mfg_Year'].unique()
    selected_year = st.sidebar.selectbox('Select Year', sorted(years))

    # Filter the DataFrame based on the selected year
    filtered_df = df[df['Mfg_Year'] == selected_year]


    # Group by Variant and calculate mean prices
    variant_prices = filtered_df.groupby('Variant')[['Tradein_MarketPrice', 'Private_MarketPrice', 'Retail_MarketPrice', 'CPO_MarketPrice', 'New_Car_Price']].mean()

    # Create a figure with multiple lines
    # fig = go.Figure()
    # for col in variant_prices.columns:
        # fig.add_trace(go.Scatter(x=variant_prices.index, y=variant_prices[col]/100000, mode='markers', name=col))

    # Customize the plot
    # fig.update_layout(title='Price Comparison of New Car v/s IBB of Toyota Innova Crysta Variants',
    #                   xaxis_title='Variant',
    #                   yaxis_title='Price',
    #                     yaxis=dict(
    #                     # tickformat=".1f",
    #                     ticksuffix='L'
    #                 )
    # )

    # Display the plot in Streamlit
    # st.plotly_chart(fig, use_container_width=True)

    # Display the variant prices DataFrame in Streamlit
    st.write(f"##### Toyota Innova Crysta Car Price For {selected_year} Manufacturing Year ")
    
    # To ungroup (reset the index)
    ungrouped_df = variant_prices.reset_index()
    
    # Rename the 'index' column to 'Variant'
    ungrouped_df.rename(columns={'index': 'Variant'}, inplace=True)
    
    st.dataframe(variant_prices)