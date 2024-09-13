import streamlit as st
import pandas as pd
from src.streamlit_include import *

st.sidebar.write("""
# Instructions:
                 
This sample app loads the observations list in df_meta, and given a trace_id it loads all the traces from the instrument and the subsystems.
                 
The last section is left empty to be filled with the command visualization.
""")

st.title("Command Visualization Tool")
st.write("Version: 2 Joaquin_Gonzalez")

st.header("Observation List")
col1, col2 = st.columns(2)
with col1:
    instrument = st.selectbox("Instrument", ["PIONIER", "MATISSE", "GRAVITY"])
with col2:
    period = st.selectbox("Period", ["1d", "1w", "1m", "6m"])

df_meta = load_meta(instrument, period)

st.write(df_meta)

st.header("Trace selection")
trace_id = st.selectbox("Trace ID", df_meta.index)
df_trace = load_trace(instrument, period, trace_id)

# Marcar los registros con timeout
df_trace['timeout'] = df_trace['logtext'].str.contains('timeout', case=False, na=False)

# Mostrar los registros con la columna de timeout
st.write(df_trace[['@timestamp', 'system', 'procid', 'logtext', 'timeout']])

# Selectbox para seleccionar periodo, instrumento y trace ID
st.header("Select Specific Trace")
col3, col4, col5 = st.columns(3)
with col3:
    specific_instrument = st.selectbox("Specific Instrument", ["PIONIER", "MATISSE", "GRAVITY"])
with col4:
    specific_period = st.selectbox("Specific Period", ["1d", "1w", "1m", "6m"])

# Cargar los datos de observación basados en las selecciones del usuario
df_meta_specific = load_meta(specific_instrument, specific_period)

with col5:
    specific_trace_id = st.selectbox("Specific Trace ID", df_meta_specific.index)

# Cargar y mostrar los comandos en el trazo específico basado en las selecciones del selectbox
if specific_instrument and specific_period and specific_trace_id is not None:
    st.header(f"Commands in trace {specific_instrument}-{specific_period}#{specific_trace_id}", divider=True)
    df_trace_specific = load_trace(specific_instrument, specific_period, specific_trace_id)
    
    st.write(df_trace_specific[['@timestamp', 'system', 'procid', 'logtext']])
   