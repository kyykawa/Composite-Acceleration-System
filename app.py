# app.py
import io
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import streamlit as st
from scipy.signal import find_peaks

# --- Page config ---
st.set_page_config(page_title="Composite Acceleration System", layout="wide")

# --- CSS for white background & black fonts ---
st.markdown("""
<style>
body { background-color: white; color: black; }
h1, h2, h3, h4 { color: black; }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("Composite Acceleration ($A_{com}$) System")
st.write("Upload CSV(s), compute A_com, choose duration, visualize, and download graph.")

# --- Helper Function for Peak Detection ---
def detect_protrusions(data_series, prominence=0.5, distance=10):
    data_array = data_series.to_numpy()
    peaks_indices, _ = find_peaks(data_array, prominence=prominence, distance=distance)
    return peaks_indices

# --- Core Processing Function ---
@st.cache_data
def process_file(uploaded_file, file_label):
    try:
        uploaded_bytes = uploaded_file.read()
        df = None
        for enc in ("utf-8", "cp932", "cp1252", "utf-8-sig", "latin1"):
            try:
                df = pd.read_csv(io.BytesIO(uploaded_bytes), skiprows=9, encoding=enc)
                break
            except Exception:
                df = None
        if df is None:
            return None, "Could not read CSV. Try UTF-8 or Shift-JIS encoding."

        df.columns = df.columns.str.strip()
        required_cols = ['加速度X', '加速度Y', '加速度Z']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            return None, f"Missing required columns: {missing_cols}"

        df['A_com'] = np.sqrt(df['加速度X']**2 + df['加速度Y']**2 + df['加速度Z']**2)

        if 'Time' in df.columns:
            time_col = 'Time'
        elif 'time' in df.columns:
            time_col = 'time'
        else:
            sampling_freq = 200
            df['Time'] = (df.index + 1) / sampling_freq
            time_col = 'Time'

        df[time_col] = df[time_col].astype(float)
        df['file_label'] = file_label
        return df, None
    except Exception as e:
        return None, f"Unexpected error processing {file_label}: {e}"

# --- Sidebar ---
st.sidebar.header("Analysis Options")
analysis_mode = st.sidebar.radio(
    "Select Mode",
    ("Single File Analysis", "Two File Comparison"),
    key='analysis_mode'
)
st.sidebar.markdown("---")
st.sidebar.header("Upload Files")

dfs = {}
errors = {}
uploaded_files = []

if analysis_mode == "Single File Analysis":
    uploaded_file_1 = st.sidebar.file_uploader("Upload CSV file (File 1)", type=["csv"], key="uploader_1")
    if uploaded_file_1:
        uploaded_files.append((uploaded_file_1, "File 1"))
else:
    uploaded_file_1 = st.sidebar.file_uploader("Upload CSV file (File 1) - LEFT Foot", type=["csv"], key="uploader_1")
    uploaded_file_2 = st.sidebar.file_uploader("Upload CSV file (File 2) - RIGHT Foot", type=["csv"], key="uploader_2")
    if uploaded_file_1:
        uploaded_files.append((uploaded_file_1, "File 1"))
    if uploaded_file_2:
        uploaded_files.append((uploaded_file_2, "File 2"))

if not uploaded_files:
    st.info(f"Upload {len(uploaded_files) + 1} CSV file(s) using the sidebar to start.")
    st.markdown("### Example CSV format to test")
    st.code(
        "Sequence,Time,加速度X,加速度Y,加速度Z\n"
        "1,0,0.12,0.05,0.98\n"
        "2,1,0.15,0.06,0.97\n"
        "3,2,0.14,0.07,0.99\n"
        "4,3,0.13,0.05,0.96\n"
        "5,4,0.11,0.06,0.95\n",
        language="csv"
    )
    st.stop()

# --- Process uploaded files ---
for uploaded_file, label in uploaded_files:
    df, error = process_file(uploaded_file, label)
    if error:
        errors[label] = error
    elif df is not None:
        dfs[label] = df

if errors:
    for label, error in errors.items():
        st.error(f"Error for {label}: {error}")

if not dfs:
    st.stop()

# --- Combine data ---
all_dfs = list(dfs.values())
combined_df = pd.concat(all_dfs, ignore_index=True)
time_col = 'Time'
min_time = combined_df[time_col].min()
max_time = combined_df[time_col].max()
total_duration = max_time - min_time

st.subheader("Combined File Duration")
st.write(f"**Overall Start:** {min_time:.3f} sec")
st.write(f"**Overall End:** {max_time:.3f} sec")
st.write(f"**Maximum Duration:** {total_duration:.3f} sec")
st.write(f"**Approximate Sampling Frequency (File 1):** {1/dfs['File 1'][time_col].diff().mean():.1f} Hz")
st.markdown("---")

# --- Duration input ---
st.sidebar.subheader("Enter Duration for Graph (seconds)")
start_time_input = st.sidebar.text_input("Start time (sec)", value=str(min_time))
end_time_input = st.sidebar.text_input("End time (sec)", value=str(max_time))

try:
    start_time = float(start_time_input)
    end_time = float(end_time_input)
    if start_time < min_time or end_time > max_time or start_time >= end_time:
        st.sidebar.warning(f"Invalid range! Using full duration ({min_time:.2f}-{max_time:.2f})")
        start_time = min_time
        end_time = max_time
except ValueError:
    st.sidebar.warning("Invalid input! Using full duration.")
    start_time = min_time
    end_time = max_time

# --- Peak detection settings ---
prominence_value = 0.50  # FIXED PROMINENCE
time_distance = 0.45
MIN_ACCELERATION_PROTRUSION = 2.0

# --- Automatic foot assignment ---
if len(dfs) == 2:
    left_foot_file = "File 1"
    right_foot_file = "File 2"
    file_color_map = {left_foot_file: 'black', right_foot_file: 'green'}
    foot_labels = {left_foot_file: "Left Foot", right_foot_file: "Right Foot"}
else:
    single_file = list(dfs.keys())[0]
    file_color_map = {}
    foot_labels = {single_file: "Foot"}

# --- Plotting ---
st.subheader(f"Composite Acceleration Graph ({start_time:.2f}-{end_time:.2f} sec)")
FIXED_GRAPH_WIDTH = 800
fig = px.line(
    combined_df,
    x=time_col,
    y='A_com',
    color='file_label',
    title=f"A_com Comparison vs Time (sec) (Raw Data)",
    render_mode="svg",
    height=600,
    width=FIXED_GRAPH_WIDTH
)

# Apply colors
for trace in fig.data:
    if trace.name in file_color_map:
        trace.line.color = file_color_map[trace.name]

# --- Peaks plotting ---
all_peaks_df = []
for i, (label, df) in enumerate(dfs.items()):
    sampling_freq = 1 / df[time_col].diff().mean() if df[time_col].diff().mean() > 0 else 200
    peak_distance = max(1, int(time_distance * sampling_freq))
    peak_indices = detect_protrusions(df['A_com'], prominence=prominence_value, distance=peak_distance)

    df_peaks = df.iloc[peak_indices].copy()
    df_peaks = df_peaks[df_peaks['A_com'] > MIN_ACCELERATION_PROTRUSION]

    if not df_peaks.empty:
        df_peaks['Peak A_com'] = df_peaks['A_com']
        df_peaks = df_peaks[[time_col, 'Peak A_com', 'file_label']]
        all_peaks_df.append(df_peaks)

        fig.add_scatter(
            x=df_peaks[time_col],
            y=df_peaks['Peak A_com'],
            mode='markers',
            name=f'{label} Peaks',
            showlegend=True,
            visible='legendonly' if len(dfs) > 1 else True,
            marker=dict(color=file_color_map.get(label, None), size=8, symbol='circle', line=dict(width=1, color='black')),
            hoverinfo='text',
            hovertext=[f'{label} ({foot_labels[label]}) Peak A_com: {v:.3f}<br>Time: {t:.3f}s' 
                       for v, t in zip(df_peaks['Peak A_com'], df_peaks[time_col])]
        )

fig.update_xaxes(range=[start_time, end_time])
fig.update_layout(legend_title_text='Foot Assignment')
st.plotly_chart(fig)

if len(dfs) == 2:
    st.markdown(f"**Line Colors:** 🖤 Black = Left Foot ({left_foot_file}), 💚 Green = Right Foot ({right_foot_file})")

# --- Peak Summary with split tables ---
if all_peaks_df:
    final_peaks_df = pd.concat(all_peaks_df, ignore_index=True)

    if len(dfs) == 1:
        st.info(f"Detected **{len(final_peaks_df)}** significant protrusions (peaks, A_com > {MIN_ACCELERATION_PROTRUSION}).")
        st.subheader("Protrusion Peak Data")
        st.dataframe(final_peaks_df[[time_col, 'Peak A_com', 'file_label']], use_container_width=True)
        st.caption("Peaks filtered by minimum acceleration and separation distance.")
    else:
        for label, foot in foot_labels.items():
            df_table = final_peaks_df[final_peaks_df['file_label'] == label].copy()
            st.subheader(f"{foot} Peak Protrusions ({label})")
            st.dataframe(df_table[[time_col, 'Peak A_com']], use_container_width=True)
        st.markdown("Peaks filtered by minimum acceleration and separation distance.")
        info_text = "Detected significant protrusions:\n"
        for label, df in dfs.items():
            count = len(final_peaks_df[final_peaks_df['file_label'] == label])
            info_text += f"* **{foot_labels[label]} ({label})**: **{count}** peaks\n"
        st.info(info_text.strip() + "\n\nClick the peak color in the legend to toggle the markers on/off.")
else:
    st.info(f"No significant protrusions detected with current settings.")

st.markdown("---")

# --- Download PNG ---
try:
    buf = io.BytesIO()
    pio.write_image(fig, buf, format="png")  # Only this line, no standalone pio.write_image
    file_prefix = "A_com_comparison" if len(dfs) > 1 else "A_com_single"
    st.download_button(
        label="Download Graph as PNG",
        data=buf,
        file_name=f"{file_prefix}_{int(start_time)}-{int(end_time)}s.png",
        mime="image/png"
    )
except Exception:
    st.warning("Cannot export graph as PNG. Install Kaleido.")
