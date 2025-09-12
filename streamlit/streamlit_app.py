import streamlit as st
import pandas as pd
import pranaam
import base64

def download_file(df):
    """Create download link for DataFrame as CSV."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="results.csv">Download results</a>'
    st.markdown(href, unsafe_allow_html=True)

def app():
    # Set app title
    st.title("🔮 pranaam: predict religion based on name")
    
    # Add sidebar info
    with st.sidebar:
        st.header("About")
        st.write("Pranaam uses Bihar Land Records data (4M+ records) to predict religion from names using ML models.")
        st.write("**Accuracy**: ~98% on out-of-sample data")
        st.write("[GitHub Repository](https://github.com/appeler/pranaam)")
        st.write("[Documentation](https://pranaam.readthedocs.io/)")

    # Description
    st.write("""
    This app predicts whether a name is **Muslim** or **not-Muslim** based on machine learning models 
    trained on Bihar Land Records data covering 35,626+ villages and 4M+ unique records.
    """)

    # Input methods
    input_method = st.radio("Choose input method:", ["Enter names manually", "Upload CSV file"])
    
    if input_method == "Enter names manually":
        # Manual input
        st.subheader("Enter Names")
        
        # Language selection
        lang = st.selectbox("Select language:", 
                          ["eng", "hin"], 
                          format_func=lambda x: "English" if x == "eng" else "Hindi")
        
        # Name input
        if lang == "eng":
            example = "Shah Rukh Khan, Amitabh Bachchan, Salman Khan"
            names_input = st.text_area(
                "Enter names (one per line or comma-separated):", 
                placeholder=example,
                height=100
            )
        else:
            example = "शाहरुख खान, अमिताभ बच्चन"
            names_input = st.text_area(
                "Enter names in Hindi (one per line or comma-separated):", 
                placeholder=example,
                height=100
            )
        
        if st.button("Predict Religion"):
            if names_input.strip():
                # Parse names
                if '\n' in names_input:
                    names = [name.strip() for name in names_input.split('\n') if name.strip()]
                else:
                    names = [name.strip() for name in names_input.split(',') if name.strip()]
                
                with st.spinner('Making predictions...'):
                    try:
                        result = pranaam.pred_rel(names, lang=lang)
                        
                        st.subheader("Results")
                        st.dataframe(result, use_container_width=True)
                        
                        # Summary
                        muslim_count = (result['pred_label'] == 'muslim').sum()
                        total_count = len(result)
                        st.write(f"**Summary**: {muslim_count} Muslim, {total_count - muslim_count} non-Muslim out of {total_count} names")
                        
                        # Download button
                        download_file(result)
                        
                    except Exception as e:
                        st.error(f"Error making predictions: {str(e)}")
            else:
                st.warning("Please enter at least one name.")
    
    else:
        # CSV Upload
        st.subheader("Upload CSV File")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("**Data loaded successfully!**")
                st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
                
                # Preview data
                with st.expander("Preview data"):
                    st.dataframe(df.head(), use_container_width=True)
                
                # Column selection
                name_col = st.selectbox("Select column containing names:", df.columns)
                lang = st.selectbox("Select language:", 
                                  ["eng", "hin"], 
                                  format_func=lambda x: "English" if x == "eng" else "Hindi")
                
                if st.button("Predict Religion for All Names"):
                    with st.spinner('Processing names...'):
                        try:
                            names_list = df[name_col].dropna().astype(str).tolist()
                            result = pranaam.pred_rel(names_list, lang=lang)
                            
                            # Merge with original data
                            result_df = df.copy()
                            result_df = result_df.merge(
                                result, 
                                left_on=name_col, 
                                right_on='name', 
                                how='left'
                            )
                            
                            st.subheader("Results")
                            st.dataframe(result_df, use_container_width=True)
                            
                            # Summary
                            muslim_count = (result_df['pred_label'] == 'muslim').sum()
                            total_count = len(result_df)
                            st.write(f"**Summary**: {muslim_count} Muslim, {total_count - muslim_count} non-Muslim out of {total_count} names")
                            
                            # Download
                            download_file(result_df)
                            
                        except Exception as e:
                            st.error(f"Error processing file: {str(e)}")
            
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        else:
            st.info("Please upload a CSV file to continue.")

    # Footer
    st.markdown("---")
    st.markdown("""
    **Note**: This tool is for research and educational purposes. The predictions are based on statistical patterns 
    and should not be used for discriminatory purposes.
    """)

# Run the app
if __name__ == "__main__":
    app()
