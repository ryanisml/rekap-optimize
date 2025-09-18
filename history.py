import streamlit as st
import pandas as pd

def show():
    st.title("History Page")
    st.write("Sample optimization history:")

    # Sample data
    data = {
        "File Name": ["data1.xlsx", "data2.xlsx", "data3.xlsx"],
        "Date": ["2025-07-10", "2025-07-11", "2025-07-12"],
        "Status": ["Optimized", "Optimized", "Optimized"]
    }
    df = pd.DataFrame(data)

    for idx, row in df.iterrows():
        cols = st.columns(len(df.columns) + 1)
        for i, col_name in enumerate(df.columns):
            cols[i].write(row[col_name])
        if cols[-1].button("View History Page", key=f"view_{idx}"):
            st.info(f"Viewing history for {row['File Name']} (Date: {row['Date']})")