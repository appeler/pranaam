import streamlit as st
import pandas as pd
from pranaam import pranaam
import base64

def download_file(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="results.csv">Download results</a>'
    st.markdown(href, unsafe_allow_html=True)

def app():
    # Set app title
    st.title("pranaam: predict religion based on name")

    # Generic info.
    st.write('Pranaam uses the Bihar Land Records data, plot-level land records, to build machine learning models that predict religion and caste from the name.')
    st.write('[Github](https://github.com/appeler/pranaam)')

    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    # Load data
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data loaded successfully!")
    else:
        st.stop()

    lname_col = st.selectbox("Select column with last name", df.columns)
    lang = st.selectbox("Select the language", ["eng", "hin"])
    function = sidebar_options[selected_function]
    if st.button('Run'):
        transformed_df = pranaam.pred_rel(df, namecol=lname_col, lang = lang)
        st.dataframe(transformed_df)
        download_file(transformed_df)

# Run the app
if __name__ == "__main__":
    app()
