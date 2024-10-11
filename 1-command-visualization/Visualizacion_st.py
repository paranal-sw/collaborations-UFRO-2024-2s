import os
import streamlit as st
from src.streamlit_include import load_meta, load_trace, visualization, extract_commands
from PIL import Image

# CSS for better image appearance and zoom functionality
st.markdown("""
    <style>
    .image-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 20px;
        position: relative;
        overflow: hidden;
        max-width: 100%; /* Ensure the container does not exceed the width of the viewport */
        max-height: 80vh; /* Ensure the container does not exceed the height of the viewport */
    }
    .image-container img {
        border: 2px solid #ddd;
        border-radius: 4px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        transition: transform 0.25s ease;
        cursor: grab;
    }
    .image-container img:active {
        cursor: grabbing;
    }
    .image-container img.zoomed {
        transform: scale(2); /* Zoom effect */
    }
    </style>
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const img = document.querySelector('.image-container img');
        let isZoomed = false;
        img.addEventListener('click', function() {
            isZoomed = !isZoomed;
            if (isZoomed) {
                img.classList.add('zoomed');
            } else {
                img.classList.remove('zoomed');
            }
        });
        img.addEventListener('mousedown', function(e) {
            if (isZoomed) {
                const startX = e.pageX - img.offsetLeft;
                const startY = e.pageY - img.offsetTop;
                function onMouseMove(e) {
                    img.style.left = `${e.pageX - startX}px`;
                    img.style.top = `${e.pageY - startY}px`;
                }
                document.addEventListener('mousemove', onMouseMove);
                document.addEventListener('mouseup', function() {
                    document.removeEventListener('mousemove', onMouseMove);
                }, { once: true });
            }
        });
    });
    </script>
    """, unsafe_allow_html=True)

# Sidebar instructions
st.sidebar.write("""
# Instructions:
                 
This sample app loads the observations list in df_meta, and given a trace_id it loads all the traces from the instrument and the subsystems.
                 
The last section is filled with the command visualization.
""")

# Main title of the app
st.title("Command Visualization Tool")
st.write("Version: Second version")

# Header for the observation list section
st.header("Observation List")
col1, col2 = st.columns(2)
with col1:
    # Dropdown for selecting the instrument
    instrument = st.selectbox("Instrument", ["PIONIER", "MATISSE", "GRAVITY"])
with col2:
    # Dropdown for selecting the period
    period = st.selectbox("Period", ["1d", "1w", "1m", "6m"])

# Load metadata based on the selected instrument and period
df_meta = load_meta(instrument, period)

# Display the metadata
st.write(df_meta)

# Header for the trace selection section
st.header("Trace selection")
# Dropdown for selecting the trace ID
trace_id = st.selectbox("Trace ID", df_meta.index)
# Load trace data based on the selected instrument, period, and trace ID
df_trace = load_trace(instrument, period, trace_id)
# Display specific columns of the trace data
st.write(df_trace[['@timestamp', 'system', 'procid', 'logtext']])

# Header for the commands in trace section
st.header(f"Commands in trace {instrument}-{period}#{trace_id}", divider=True)

# Extract commands from the trace data
unique_commands = extract_commands(df_trace)

# Display the unique commands
st.write("Unique Commands:", unique_commands)

# Input for entering the command list
command_list = st.selectbox("Select a command", options=unique_commands, index=0, key='command_list')

# Button to generate the UML diagram
if st.button("Generate UML Diagram"):
    # Generate the UML diagram and get the path to the PNG file
    png_file = visualization(instrument, period, trace_id, command_list)
    if os.path.exists(png_file):
        # Load the image with Pillow
        img = Image.open(png_file)
        
        # Resize the image to improve visual quality
        new_width = 7000  # Adjust this value as needed
        new_height = int((float(img.size[1]) * (new_width / float(img.size[0]))))
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Display the image with Streamlit and apply CSS for zoom
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(img_resized, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Display an error message if the UML diagram could not be generated
        st.error("Error generating UML diagram. Please check the logs.")



            