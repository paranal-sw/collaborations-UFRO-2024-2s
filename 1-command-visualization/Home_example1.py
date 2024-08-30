import streamlit as st
from src.streamlit_include import *

st.sidebar.write("""
# Instructions:
                 
This sample app loads the observations list in df_meta, and given a trace_id it loads all the traces from the instrument and the subsystems.
                 
The last section is left empty to be filled with the command visualization.
""")

st.title("Command Visualization Tool")
st.write("Version: First example")

st.header("Observation List")
col1, col2 = st.columns(2)
with col1:
    instrument = st.selectbox("Instrument", ["PIONIER", "MATISSE", "GRAVITY"])
with col2:
    period = st.selectbox("Period", ["1d", "1w", "1m", "6m"])

df_meta = load_meta(instrument, period)

st.write(df_meta)

st.header("Trace selection")
trace_id=st.selectbox("Trace ID", df_meta.index)
df_trace = load_trace(instrument, period, trace_id)
st.write(df_trace[['@timestamp', 'system', 'procid', 'logtext']])

st.header(f"Commands in trace {instrument}-{period}#{trace_id}", divider=True)
st.write("Write here the visualization")