import os
import json
import tempfile
from io import BytesIO
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer
from Backend.detect import (
    analyze_image,
    analyze_video
)
import cv2
import threading
import streamlit as st
# API_URL = "https://your-api.onrender.com"

def load_bpm():
    try:
        return float(
            np.load(
                "Backend/results/bpm.npy"
            )[0]
        )
    except:
        return None


def load_alertness():
    try:
        return float(
            np.load(
                "Backend/results/alertness_score.npy"
            )[0]
        )
    except:
        return None

# ==================================================
# CONFIG
# ==================================================

st.set_page_config(
    page_title="AI Classroom Monitoring",
    page_icon="🎓",
    layout="wide"
)


# ==================================================
# HELPERS
# ==================================================



# ==================================================
# LOAD DATA
# ==================================================
dashboard = st.session_state.get(
    "analysis_result",
    {}
)
student_df = pd.DataFrame()

analytics = dashboard.get(
    "counts",
    {}
)

detections = analytics

# Teacher Alert
final_score = dashboard.get(
    "final_score",
    0
)

if final_score >= 80:
    teacher_alert = {
        "alert_color": "GREEN",
        "score": final_score,
        "status": "Classroom Performing Well"
    }

elif final_score >= 60:
    teacher_alert = {
        "alert_color": "YELLOW",
        "score": final_score,
        "status": "Moderate Attention"
    }

else:
    teacher_alert = {
        "alert_color": "RED",
        "score": final_score,
        "status": "Immediate Intervention Required"
    }

# Students
students_file = "Backend/results/students.json"
teacher_file = "Backend/results/teacher.json"

if (
    os.path.exists(teacher_file)
    and os.path.getsize(teacher_file) > 0
):
    try:
        with open(teacher_file, "r") as f:
            teacher = json.load(f)
    except json.JSONDecodeError:
        teacher = {}
else:
    teacher = {}
    
if os.path.exists(students_file):
    with open(students_file, "r") as f:
        students = json.load(f)
else:
    students = {}

# History
history_file = "Backend/results/attention_history.json"

if os.path.exists(history_file):
    with open(history_file, "r") as f:
        history = json.load(f)
else:
    history = {
        "labels": [],
        "attention": [],
        "engagement": [],
        "pulse": []
    }
# ==================================================
# HEADER
# ==================================================


st.title("AI Classroom Monitoring")



st.caption(
    "Classroom Attention and Engagement Analytics"
)



# ==================================================
# REFRESH
# ==================================================



if st.button(
    "Refresh Dashboard",
    key="refresh_dashboard"
):



    st.rerun()



# ==================================================
# TABS
# ==================================================
dashboard_tab, students_tab, monitoring_tab, teacher_tab, reports_tab, register_tab = st.tabs(
    [
        "Dashboard",
        "Students",
        "Monitoring",
        "Teacher",
        "Reports",
        "Register Student"
    ]
)

with dashboard_tab:


    st.subheader("Classroom Status")

    status = dashboard.get(
        "status",
        "Class Not Analysed Yet"
    )

    if "Highly" in status:
        st.success(status)

    elif "Moderately" in status:
        st.warning(status)

    else:
        st.error(status)

    gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=dashboard.get(
            "final_score",
            0
        ),
        title={
            "text": "Classroom Performance"
        },
        gauge={
            "axis": {
                "range": [0, 100]
            },
            "bar": {
                "color": "royalblue"
            }
        }
    )
)

    st.plotly_chart(
    gauge,
    use_container_width=True,
    key="dashboard_gauge"
)
    
    bpm = load_bpm()
rppg_alertness = load_alertness()
attention_score = dashboard.get(
    "attention_score",
    0
)

alertness_score = dashboard.get(
    "alertness_score",
    0
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Attention Score",
        f"{attention_score:.2f}%"
    )

with col2:
    st.metric(
        "Alertness Score",
        f"{alertness_score:.2f}%"
    )

with col3:
    st.metric(
        "Heart Rate",
        f"{bpm:.0f} BPM"
        if bpm
        else "N/A"
    )

with col4:
    st.metric(
        "rPPG Alertness",
        f"{rppg_alertness:.1f}%"
        if rppg_alertness
        else "N/A"
    )

if bpm is not None:

    if bpm < 60:
        pulse_status = "Low Pulse"

    elif bpm <= 100:
        pulse_status = "Normal Pulse"

    else:
        pulse_status = "Elevated Pulse"

    st.info(
        f"Pulse Status: {pulse_status} ({bpm:.0f} BPM)"
    )
# Normalize analytics keys
analytics = {
    str(k).lower(): v
    for k, v in analytics.items()
}

chart_df = pd.DataFrame({
    "Behavior": [
        "Attentive",
        "Distracted",
        "Sleepy",
        "Phone Use"
    ],
    "Count": [
        analytics.get("attentive", 0),
        analytics.get("distracted", 0),
        analytics.get("sleepy", 0),
        analytics.get("phone", 0)
    ]
})

fig = px.bar(
    chart_df,
    x="Behavior",
    y="Count",
    text="Count",
    color="Behavior",
    title="Classroom Behaviour Distribution"
)

fig.update_traces(
    textposition="outside"
)

st.plotly_chart(
    fig,
    use_container_width=True,
    key="behavior_chart"
)

st.divider()

st.subheader("🧪 Pulse Debug")

st.write("BPM:", bpm)

st.write(
    "rPPG Alertness:",
    rppg_alertness
)

st.write(
    "Pulse File Exists:",
    os.path.exists(
        "Backend/results/bpm.npy"
    )
)

st.write(
    "Alertness File Exists:",
    os.path.exists(
        "Backend/results/alertness_score.npy"
    )
)
# STUDENTS TAB
# ==================================================
with students_tab:

    st.subheader("Student Analytics")

    rows = []

    for student_id, data in students.items():

        rows.append({
            "Student ID": student_id,
            "Name": data.get("name", "Unknown"),
            "Attention Rate": data.get("attention_rate", 0),
            "Status": data.get("status", "Unknown")
        })

    student_df = pd.DataFrame(rows)

    if not student_df.empty:

        st.dataframe(
            student_df,
            use_container_width=True,
            hide_index=True
        )

        ranking_chart = px.bar(
            student_df,
            x="Student ID",
            y="Attention Rate",
            color="Attention Rate",
            text="Attention Rate",
            title="Student Attention Ranking"
        )

        st.plotly_chart(
            ranking_chart,
            use_container_width=True,
            key="student_ranking_chart"
        )
if not student_df.empty:

    heatmap_df = pd.DataFrame(
        [student_df["Attention Rate"].tolist()]
    )

    heatmap = px.imshow(
        heatmap_df,
        text_auto=True,
        color_continuous_scale="RdYlGn",
        title="Attention Heatmap"
    )

    st.plotly_chart(
        heatmap,
        use_container_width=True,
        key="attention_heatmap"
    )
if not student_df.empty:

    leaderboard = student_df.sort_values(
        by="Attention Rate",
        ascending=False
    )

    st.subheader("🏆 Attention Leaderboard")

    st.dataframe(
        leaderboard,
        use_container_width=True,
        hide_index=True
    )

    leaderboard_chart = px.bar(
        leaderboard,
        x="Name",
        y="Attention Rate",
        color="Attention Rate",
        text="Attention Rate",
        title="Student Attention Leaderboard"
    )

    st.plotly_chart(
        leaderboard_chart,
        use_container_width=True,
        key="leaderboard_chart"
    )
    if not student_df.empty:

        best_student = student_df.loc[
        student_df["Attention Rate"].idxmax()
    ]

    worst_student = student_df.loc[
        student_df["Attention Rate"].idxmin()
    ]

    col1, col2 = st.columns(2)

    with col1:
        st.success(
            f"🥇 Most Attentive: "
            f"{best_student['Name']} "
            f"({best_student['Attention Rate']}%)"
        )

    with col2:
        st.error(
            f"⚠ Least Attentive: "
            f"{worst_student['Name']} "
            f"({worst_student['Attention Rate']}%)"
        )
    st.subheader("❤️ Pulse Alerts")

for student_id, data in students.items():

    pulse = data.get(
        "pulse",
        0
    )

    if pulse and pulse < 50:

        st.error(
            f"{data.get('name')} - {pulse} BPM"
        )
signal_file = "Backend/results/green_signal.npy"

if os.path.exists(signal_file):

    signal = np.load(signal_file)

    pulse_df = pd.DataFrame({
        "Frame": range(len(signal)),
        "Pulse Signal": signal
    })

    st.subheader(
        "rPPG Pulse Waveform"
    )

    fig = px.line(
        pulse_df,
        x="Frame",
        y="Pulse Signal",
        title="Extracted Pulse Signal"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="pulse_waveform"
    )

else:

        st.info(
            "No students registered yet."
        )
        
st.subheader("❤️ Physiological Monitoring")

bpm = load_bpm()
rppg_alertness = load_alertness()

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Heart Rate",
        f"{bpm:.0f} BPM" if bpm else "N/A"
    )

with col2:
    st.metric(
        "Physiological Alertness",
        f"{rppg_alertness:.1f}%"
        if rppg_alertness
        else "N/A"
    )
teacher_image = (
    "Backend/data/teacher/teacher.jpg"
)

if os.path.exists(
    teacher_image
):
    st.image(
        teacher_image,
        width=250
    )
teacher_score = round(
    (
        final_score * 0.7
    ) +
    (
        dashboard.get(
            "attention_score",
            0
        ) * 0.3
    ),
    2
)

st.metric(
    "Teaching Effectiveness",
    f"{teacher_score}%"
)

# ==================================================
# register TAB
# ==================================================
     
        
        
with register_tab:

    st.subheader("Student Registration")

    student_id = st.text_input("Student ID")

    student_name = st.text_input("Student Name")

    photo = st.file_uploader(
        "Student Photo",
        type=["jpg", "jpeg", "png"]
    )

    if st.button(
        "Register Student",
        key="register_student_btn"
    ):

        if photo is not None:

            photo_dir = "Backend/data/student_photos"

            os.makedirs(
                photo_dir,
                exist_ok=True
            )

            students_file = "Backend/results/students.json"
            teacher_file = "Backend/results/teacher.json"

            if os.path.exists(teacher_file):
                with open(teacher_file, "r") as f:
                    teacher = json.load(f)
            else:
                teacher = {}
            
            if (
                os.path.exists(students_file)
                and os.path.getsize(students_file) > 0
            ):
                with open(
                    students_file,
                    "r"
                ) as f:
                    students = json.load(f)
            else:
                students = {}

            students[student_id] = {
            "name": student_name,
            "photo": f"{student_id}.jpg",
            "attention_rate": 0,
            "pulse": 72,
            "rppg_alertness": 88,
            "status": "Normal"
}


            with open(
                students_file,
                "w"
            ) as f:
                json.dump(
                    students,
                    f,
                    indent=4
                )

            with open(
                f"{photo_dir}/{student_id}.jpg",
                "wb"
            ) as f:
                f.write(
                    photo.getbuffer()
                )

            st.success(
                "Student registered successfully."
            )

        else:
            st.error(
                "Please upload a photo."
            )
            
# ==================================================
# MONITORING TAB
# ==================================================
with monitoring_tab:

    st.subheader("Classroom Monitoring")

    source = st.radio(
        "Select Source",
        [
            "Upload Image",
            "Upload Video",
            "Camera Snapshot"
        ]
    )

    st.subheader("Live Webcam")

    webrtc_streamer(
        key="webcam",
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )

    st.divider()

    # ===============================
    # IMAGE
    # ===============================

    if source == "Upload Image":

        uploaded_image = st.file_uploader(
            "Upload Classroom Image",
            type=["jpg", "jpeg", "png"],
            key="image_upload"
        )

        if uploaded_image is not None:

            st.image(
                uploaded_image,
                use_container_width=True
            )

            if st.button(
                "Analyze Image",
                key="analyze_image"
            ):

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".jpg"
                ) as tmp_file:

                    tmp_file.write(
                        uploaded_image.getbuffer()
                    )

                    temp_path = tmp_file.name
                    
                result = analyze_image(temp_path)

                st.session_state["analysis_result"] = result

                st.rerun()
                st.session_state[
                    "analysis_result"
                ] = result

                st.success(
                    "Analysis Complete"
                )

                st.json(result)

    # ===============================
    # VIDEO
    # ===============================
    elif source == "Upload Video":

        uploaded_video = st.file_uploader(
            "Upload Classroom Video",
            type=["mp4", "avi", "mov"],
            key="video_upload"
        )

        if uploaded_video is not None:

            st.video(uploaded_video)

            if st.button(
                "Analyze Video",
                key="analyze_video"
            ):

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ) as tmp_video:

                    tmp_video.write(
                        uploaded_video.read()
                    )

                    video_path = tmp_video.name

                result = analyze_video(
                    video_path
                )

                st.session_state[
                    "analysis_result"
                ] = result

                st.success(
                    "Video Analysis Complete"
                )

                st.rerun()

    # ===============================
    # CAMERA
    # ===============================

    elif source == "Camera Snapshot":

        snapshot = st.camera_input(
            "Take Photo"
        )

        if snapshot is not None:

            if st.button(
                "Analyze Snapshot",
                key="analyze_snapshot"
            ):

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".jpg"
                ) as tmp_file:

                    tmp_file.write(
                        snapshot.getbuffer()
                    )

                    temp_path = tmp_file.name

                result = analyze_image(
                    temp_path
                )

                st.session_state[
                    "analysis_result"
                ] = result

                st.success(
                    "Snapshot Analysis Complete"
                )

                st.json(result)

    st.divider()

    # ===============================
    # CCTV
    # ===============================

    st.subheader("CCTV / RTSP Monitoring")

    rtsp_url = st.text_input(
        "RTSP URL",
        key="rtsp_url"
    )

    if st.button(
        "Start CCTV Feed",
        key="start_cctv"
    ):

        cap = cv2.VideoCapture(rtsp_url)

        frame_placeholder = st.empty()

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                st.error(
                    "Unable to read CCTV stream"
                )
                break

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frame_placeholder.image(
                rgb,
                channels="RGB",
                use_container_width=True
            )

        cap.release()

# ==================================================
# teacher TAB
# ==================================================
with teacher_tab:

    st.subheader("👨‍🏫 Teacher Analytics")

    teacher_name = st.text_input(
        "Teacher Name",
        key="teacher_name"
    )

    teacher_photo = st.file_uploader(
        "Teacher Photo",
        type=["jpg", "jpeg", "png"],
        key="teacher_photo"
    )

    if st.button(
        "Register Teacher",
        key="register_teacher"
    ):
        ...
        ...

    if teacher:

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Teacher Presence",
                teacher.get("presence", 0)
            )

        with col2:
            st.metric(
                "Class Engagement",
                final_score
            )

        with col3:
            st.metric(
                "Heart Rate",
                teacher.get("heart_rate", 0)
            )

        with col4:
            st.metric(
                "Status",
                teacher.get("status", "Unknown")
            )

    st.subheader(
        "AI Teaching Recommendations"
    )
    if final_score < 60:
        st.error(
            "Students are disengaged. Increase interaction."
        )
    elif final_score < 80:
        st.warning(
            "Moderate engagement detected."
        )
    else:
        st.success(
            "Excellent classroom engagement."
        )
# ==================================================
# REPORTS TAB
# ==================================================
with reports_tab:

    st.subheader("Reports")

    if not student_df.empty:

        csv = student_df.to_csv(
            index=False
        )

        st.download_button(
            "Download CSV",
            csv,
            "student_report.csv",
            "text/csv"
        )

        buffer = BytesIO()

        with pd.ExcelWriter(
            buffer,
            engine="openpyxl"
        ) as writer:

            student_df.to_excel(
                writer,
                index=False
            )

        st.download_button(
            "Download Excel",
            buffer.getvalue(),
            "student_report.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    if history:

        trend_df = pd.DataFrame(
            {
                "Time": history.get(
                    "labels",
                    []
                ),
                "Attention": history.get(
                    "attention",
                    []
                ),
                "Engagement": history.get(
                    "engagement",
                    []
                )
            }
        )

        if not trend_df.empty:

            st.subheader(
                "Attention Trend"
            )

            fig = px.line(
                trend_df,
                x="Time",
                y=[
                    "Attention",
                    "Engagement"
                ],
                markers=True
            )

            st.plotly_chart(
            fig,
            use_container_width=True,
            key="attention_trend_chart"
            )
    st.info(
        "Historical reports and trend analytics will appear here."
    )
    
    