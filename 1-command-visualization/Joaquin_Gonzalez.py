import os
import pandas as pd
from urllib.request import urlretrieve
import streamlit as st
import tempfile
import plantuml
from PIL import Image

REPO_URL = 'https://huggingface.co/datasets/Paranal/parlogs-observations/resolve/main/data'

if 'COLAB_RELEASE_TAG' in os.environ.keys():
    PATH = 'data/'  # Convenient name to be Colab compatible
else:
    PATH = 'D:/Ufro/11 nivel/Lab modelacion 2/Trabajo_paramal/collaborations-UFRO-2024-2s/data'  # Local directory to your system

def load_dataset(INSTRUMENT, RANGE):
    fname = f'{INSTRUMENT}-{RANGE}-traces.parket'
    if not os.path.exists(f'{PATH}/{fname}'):
        urlretrieve(f'{REPO_URL}/{fname}', f'{PATH}/{fname}')
    df_inst = pd.read_parquet(f'{PATH}/{fname}')

    fname = f'{INSTRUMENT}-{RANGE}-traces-SUBSYSTEMS.parket'
    if not os.path.exists(f'{PATH}/{fname}'):
        urlretrieve(f'{REPO_URL}/{fname}', f'{PATH}/{fname}')
    df_subs = pd.read_parquet(f'{PATH}/{fname}')

    fname = f'{INSTRUMENT}-{RANGE}-traces-TELESCOPES.parket'
    if not os.path.exists(f'{PATH}/{fname}'):
        urlretrieve(f'{REPO_URL}/{fname}', f'{PATH}/{fname}')
    df_tele = pd.read_parquet(f'{PATH}/{fname}')

    all_traces = [df_inst, df_subs, df_tele]

    df_all = pd.concat(all_traces)
    df_all.sort_values('@timestamp', inplace=True)
    df_all.reset_index(drop=True, inplace=True)

    return df_all

def load_trace(INSTRUMENT, RANGE, trace_id):
    df_all = load_dataset(INSTRUMENT, RANGE)
    df_all = df_all[df_all['trace_id'] == trace_id]
    return df_all

def load_meta(INSTRUMENT, RANGE):
    fname = f'{INSTRUMENT}-{RANGE}-meta.parket'
    if not os.path.exists(f'{PATH}/{fname}'):
        urlretrieve(f'{REPO_URL}/{fname}', f'{PATH}/{fname}')
    df_meta = pd.read_parquet(f'{PATH}/{fname}')
    return df_meta

# Función para generar el código PlantUML para un solo usuario
def generate_plantuml_code_single_user(meta_row, trace_row):
    uml_code = "@startuml\n"
    uml_code += "actor User\n"
    uml_code += "participant System\n"
    uml_code += "participant Instrument\n"
    uml_code += "participant Subsystem\n\n"

    start_time = meta_row['START']
    end_time = meta_row['END']
    procname = meta_row['procname']
    status = 'success' if not meta_row['ERROR'] else 'failure'
    logtext = trace_row['logtext']

    uml_code += f"User -> System: Send Command ({procname})\n"
    uml_code += "activate System\n"
    uml_code += f"System -> Instrument: Forward Command ({procname})\n"
    uml_code += "activate Instrument\n"
    uml_code += f"Instrument -> Subsystem: Execute Command ({procname})\n"
    uml_code += "activate Subsystem\n\n"

    if status == 'success':
        uml_code += f"Subsystem -> Instrument: Return Success ({start_time}, {end_time})\n"
    else:
        uml_code += f"Subsystem -> Instrument: Return Failure ({start_time}, {end_time})\n"
    
    uml_code += "deactivate Subsystem\n\n"
    uml_code += f"Instrument -> System: Return Status ({status})\n"
    uml_code += "deactivate Instrument\n\n"
    uml_code += f"System -> User: Return Status ({status})\n"
    uml_code += "deactivate System\n\n"
    uml_code += f"note right of System: Log Entry - {logtext}\n"

    uml_code += "@enduml"
    return uml_code

# Streamlit UI
st.title("VLTI Log Visualization")

# Selección de parámetros
instrument = st.selectbox("Select Instrument", ["PIONIER", "GRAVITY", "MATISSE"])
time_range = st.selectbox("Select Time Range", ["1d", "1w", "1m", "6m"])
trace_id = st.number_input("Enter Trace ID", min_value=0, step=1)

if st.button("Generate Diagram"):
    df_meta = load_meta(instrument, time_range)
    df_trace = load_trace(instrument, time_range, trace_id)

    # Verificar las columnas de los DataFrames
    st.write("Columnas en df_meta:", df_meta.columns)
    st.write("Columnas en df_trace:", df_trace.columns)

    # Seleccionar una fila del DataFrame para el usuario
    if trace_id < len(df_meta):
        meta_row = df_meta.iloc[trace_id]
    else:
        st.error("El índice de trace_id está fuera de los límites de df_meta.")
        st.stop()

    if trace_id < len(df_trace):
        trace_row = df_trace.iloc[0]
    else:
        st.error("El índice de trace_id está fuera de los límites de df_trace.")
        st.stop()

    # Generar el código PlantUML para un solo usuario
    uml_code = generate_plantuml_code_single_user(meta_row, trace_row)
    
    # Usar archivos temporales para el código PlantUML y la imagen generada
    temp_dir = 'D:/Ufro/11 nivel/Lab modelacion 2/Trabajo_paramal/collaborations-UFRO-2024-2s/1-command-visualization'  # Cambia esto a la ruta deseada para los archivos temporales
    os.makedirs(temp_dir, exist_ok=True)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".puml", dir=temp_dir) as puml_file:
        puml_file.write(uml_code.encode('utf-8'))
        puml_file_path = puml_file.name
    
    # Mostrar la ruta de los archivos temporales
    st.write(f"Archivo PlantUML temporal: {puml_file_path}")
    
    # Renderizar el diagrama usando la biblioteca plantuml
    plantuml_server = PlantUML(url='http://www.plantuml.com/plantuml/png')
    plantuml_server.processes_file(puml_file_path)
    
    # Obtener la ruta del archivo PNG generado
    png_file_path = puml_file_path.replace('.puml', '.png')
    
    # Verificar si el archivo se ha generado correctamente
    if os.path.exists(png_file_path):
        # Mostrar el diagrama generado
        with open(png_file_path, 'rb') as img_file:
            st.image(img_file.read())
        # Guardar el archivo PNG en una ubicación específica
        output_png_path = 'D:/Ufro/11 nivel/Lab modelacion 2/Trabajo_paramal/collaborations-UFRO-2024-2s/diagram.png'
        os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
        with open(output_png_path, 'wb') as out_file:
            out_file.write(open(png_file_path, 'rb').read())
        st.write(f"Diagrama guardado en: {output_png_path}")
    else:
        st.error("El archivo PNG no se ha generado correctamente.")

   
 




            