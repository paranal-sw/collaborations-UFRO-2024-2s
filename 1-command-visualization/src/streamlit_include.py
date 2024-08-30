# Functions to be used in streamlit
import os
import pandas as pd
# os.environ['STREAMLIT_SUPPRESS_LOGS'] = 'True'
import streamlit as st

from urllib.request import urlretrieve
REPO_URL='https://huggingface.co/datasets/Paranal/parlogs-observations/resolve/main/data'
PATH='../data/' # Convenient name to be Colab compatible

@st.cache_data
def load_dataset(INSTRUMENT, RANGE):

    fname = f'{INSTRUMENT}-{RANGE}-traces.parket'
    if not os.path.exists(f'{PATH}/{fname}'):
        urlretrieve(f'{REPO_URL}/{fname}', f'{PATH}/{fname}')
    df_inst=pd.read_parquet(f'{PATH}/{fname}')

    fname = f'{INSTRUMENT}-{RANGE}-traces-SUBSYSTEMS.parket'
    if not os.path.exists(f'{PATH}/{fname}'):
        urlretrieve(f'{REPO_URL}/{fname}', f'{PATH}/{fname}')
    df_subs=pd.read_parquet(f'{PATH}/{fname}')

    fname = f'{INSTRUMENT}-{RANGE}-traces-TELESCOPES.parket'
    if not os.path.exists(f'{PATH}/{fname}'):
        urlretrieve(f'{REPO_URL}/{fname}', f'{PATH}/{fname}')
    df_tele=pd.read_parquet(f'{PATH}/{fname}')

    all_traces = [df_inst, df_subs, df_tele]

    df_all = pd.concat(all_traces)
    df_all.sort_values('@timestamp', inplace=True)
    df_all.reset_index(drop=True, inplace=True)

    return df_all

def load_trace(INSTRUMENT, RANGE, trace_id):
    df_all = load_dataset(INSTRUMENT, RANGE)
    df_all = df_all[ df_all['trace_id']==trace_id ]
    return df_all

@st.cache_data
def load_meta(INSTRUMENT, RANGE):
    fname = f'{INSTRUMENT}-{RANGE}-meta.parket'
    if not os.path.exists(f'{PATH}/{fname}'):
        urlretrieve(f'{REPO_URL}/{fname}', f'{PATH}/{fname}')
    df_meta=pd.read_parquet(f'{PATH}/{fname}')

    return df_meta


