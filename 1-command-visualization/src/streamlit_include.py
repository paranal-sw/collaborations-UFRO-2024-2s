import os
import pandas as pd
import subprocess
from urllib.request import urlretrieve
from PIL import Image
import re

# URL of the repository containing the datasets
REPO_URL = 'https://huggingface.co/datasets/Paranal/parlogs-observations/resolve/main/data'

# Determine the path based on the environment (Colab or local)
if 'COLAB_RELEASE_TAG' in os.environ.keys():
    PATH = 'data/'  # Convenient name to be Colab compatible
else:
    PATH = 'D:/Ufro/11 nivel/Lab modelacion 2/Trabajo_paramal/collaborations-UFRO-2024-2s/data'  # Local directory to your system

# Function to load the dataset for a given instrument and range
def load_dataset(INSTRUMENT, RANGE):
    # Load instrument traces
    fname = f'{INSTRUMENT}-{RANGE}-traces.parket'
    if not os.path.exists(f'{PATH}/{fname}'):
        urlretrieve(f'{REPO_URL}/{fname}', f'{PATH}/{fname}')
    df_inst = pd.read_parquet(f'{PATH}/{fname}')

    # Load subsystem traces
    fname = f'{INSTRUMENT}-{RANGE}-traces-SUBSYSTEMS.parket'
    if not os.path.exists(f'{PATH}/{fname}'):
        urlretrieve(f'{REPO_URL}/{fname}', f'{PATH}/{fname}')
    df_subs = pd.read_parquet(f'{PATH}/{fname}')

    # Load telescope traces
    fname = f'{INSTRUMENT}-{RANGE}-traces-TELESCOPES.parket'
    if not os.path.exists(f'{PATH}/{fname}'):
        urlretrieve(f'{REPO_URL}/{fname}', f'{PATH}/{fname}')
    df_tele = pd.read_parquet(f'{PATH}/{fname}')

    # Combine all traces into a single DataFrame
    all_traces = [df_inst, df_subs, df_tele]
    df_all = pd.concat(all_traces)
    df_all.sort_values('@timestamp', inplace=True)
    df_all.reset_index(drop=True, inplace=True)

    return df_all

# Function to load a specific trace by trace_id
def load_trace(INSTRUMENT, RANGE, trace_id):
    df_all = load_dataset(INSTRUMENT, RANGE)
    df_all = df_all[df_all['trace_id'] == trace_id]
    return df_all

# Function to load metadata for a given instrument and range
def load_meta(INSTRUMENT, RANGE):
    fname = f'{INSTRUMENT}-{RANGE}-meta.parket'
    if not os.path.exists(f'{PATH}/{fname}'):
        urlretrieve(f'{REPO_URL}/{fname}', f'{PATH}/{fname}')
    df_meta = pd.read_parquet(f'{PATH}/{fname}')
    return df_meta

# Function to generate PlantUML code for a single user
def generate_plantuml_code_single_user(meta_row, trace_row, Lista_system, Lista_procname, Lista_time, lista_log_text):
    uml_code = "@startuml\n"
    uml_code += 'actor User\n'
    
    # Create a dictionary to track unique participants and their procname
    participants = {}
    
    # Add the first transition from User to the first system
    if Lista_system:
        first_system = Lista_system[0]
        first_procname = Lista_procname[0]
        uml_code += f'participant {first_system}\n'
        uml_code += f'User -> {first_system}: {lista_log_text[0]}\n'
        uml_code += f'note right of User: {str(Lista_time[0])}\n'
        uml_code += f'note right of User: {lista_log_text[0]}\n'
        participants[first_system] = [first_procname]
    
    # Generate interactions between systems and add log_text notes
    for i in range(len(Lista_system)):
        system = Lista_system[i]
        procname = Lista_procname[i]
        
        # Add the participant if not already added
        if system not in participants:
            participants[system] = [procname]
            uml_code += f'participant {system}\n'
        else:
            # If procname is not in the system's procname list, add it
            if procname not in participants[system]:
                participants[system].append(procname)
                uml_code += f'note right of {system}: {procname}\n'
        
        from_participant = system
        if i < len(Lista_system) - 1:
            next_system = Lista_system[i+1]
            next_procname = Lista_procname[i+1]
            to_participant = next_system
            if system != next_system:
                uml_code += f'{from_participant} -> {to_participant}: {lista_log_text[i]}\n'
                uml_code += f'note right of {from_participant}: {str(Lista_time[i])}\n'
                uml_code += f'note right of {from_participant}: {lista_log_text[i]}\n'
            else:
                uml_code += f'note right of {from_participant}: {str(Lista_time[i])}\n'
                uml_code += f'note right of {from_participant}: {lista_log_text[i]}\n'
        else:
            uml_code += f'{from_participant} -> User: {lista_log_text[i]}\n'
            uml_code += f'note right of {from_participant}: {str(Lista_time[i])}\n'
            uml_code += f'note right of {from_participant}: {lista_log_text[i]}\n'

    uml_code += "@enduml"
    return uml_code

# Function to visualize the command sequence
def visualization(INSTRUMENT, RANGE, TRACE_ID, COMMANDLIST):
    df_meta = load_meta(INSTRUMENT, RANGE)
    df_trace = load_trace(INSTRUMENT, RANGE, TRACE_ID)
    
    # Verify the columns of the DataFrames
    print("Columns in df_meta:", df_meta.columns)
    print("Columns in df_trace:", df_trace.columns)
    
    # Select a row from the DataFrame for the user
    meta_row = df_meta.iloc[TRACE_ID]

    if 'trace_id' in df_trace.columns:
        trace_row = df_trace.iloc[0]
    else:
        print("The column 'trace_id' is not present in df_trace. Verify the column name.")

    # Filter and process the data from df_trace
    df_trace = df_trace[['@timestamp', 'system', 'procname', 'logtext', 'trace_id']]
    df_tratamiento = df_trace[df_trace['logtext'].str.contains(COMMANDLIST)][:]

    lista = ['Send', 'Received command', 'Waiting', 'Succesfully completed', 'Last reply']
    Lista_system = []

    for x in lista:
        df_cc = df_tratamiento[df_tratamiento['logtext'].str.contains(x)]['system']
        Lista_system.extend(df_cc.tolist())

    print(Lista_system)

    lista1 = ['Send', 'Received command', 'Waiting', 'Succesfully completed', 'Last reply']
    Lista_procname = []

    for x in lista1:
        df_cc = df_tratamiento[df_tratamiento['logtext'].str.contains(x)]['procname']
        Lista_procname.extend(df_cc.tolist())

    print(Lista_procname)

    lista2 = ['Send', 'Received command', 'Waiting', 'Succesfully completed', 'Last reply']
    Lista_time = []
    for x in lista2:
        df_cc = df_tratamiento[df_tratamiento['logtext'].str.contains(x)]['@timestamp']
        Lista_time.extend(df_cc.tolist())

    print(Lista_time)

    lista_log_text = df_trace[df_trace['logtext'].str.contains(COMMANDLIST)][:]['logtext'].tolist()
    
    # Generate PlantUML code for a single user
    uml_code = generate_plantuml_code_single_user(meta_row, trace_row, Lista_system, Lista_procname, Lista_time, lista_log_text)

    # Save the PlantUML code to a file
    uml_file = 'dynamic_diagram.puml'
    with open(uml_file, 'w') as file:
        file.write(uml_code)

    # Path to the plantuml.jar file
    plantuml_path = 'c:/Users/joaco/anaconda3/envs/Paramal/Library/lib/plantuml.jar'  # Update this path where the plantuml.jar is located

    # .puml file to be converted
    input_file = uml_file

    # Run the Java command to convert the .puml file to .png with higher resolution
    try:
        subprocess.run(['java', '-DPLANTUML_LIMIT_SIZE=8192', '-jar', plantuml_path, input_file], check=True)
        print(f"File {input_file} converted to .png successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error converting the file: {e}")

    # Verify if the PNG file was generated successfully
    png_file = 'dynamic_diagram.png'
    if os.path.exists(png_file):
        print(f"UML diagram generated and saved as '{png_file}'")
    else:
        print(f"Error: Could not generate the file '{png_file}'")

    return png_file

# Function to extract commands from logtext
def extract_commands(df_trace):
    # Extract logtext entries that contain the word "command"
    cmd = df_trace[df_trace['logtext'].str.contains("command")]['logtext']


    # Define a regex pattern to clean the logtext
    pattern = re.compile(r"[:,' ]+")

    # Clean the logtext using the regex pattern
    cmd = cmd.apply(lambda x: pattern.sub(" ", x))

    # Define a regex pattern to extract commands
    command_pattern = re.compile(r"\bcommand (\w+)\b")

    # Extract commands using the regex pattern
    cmdlist = cmd.apply(lambda x: command_pattern.findall(x)).explode()

    # Filter out unwanted commands
    cmdlist = cmdlist[~cmdlist.isin(["...", "done", "to"])]

    
    # Drop NaN values
    cmdlist = cmdlist.dropna()

    # Convert to a set to get unique commands
    unique_commands = set(cmdlist)
    unique_commands = list(unique_commands)
    return unique_commands





