import os
import json
import tempfile
from io import BytesIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer
from Backend.detect import analyze_image
import cv2
import threading
import streamlit as st
# API_URL = "https://your-api.onrender.com"



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

dashboard_tab, students_tab, monitoring_tab, reports_tab, register_tab = st.tabs(
    [
        "Dashboard",
        "Students",
        "Monitoring",
        "Reports",
        "Register Student"
    ]
)

with dashboard_tab:

    col1, col2, col3, col4 = st.columns(4)

    pulse_data = history.get("pulse", [0])

    current_pulse = (
        pulse_data[-1]
        if pulse_data
        else 0
    )
    col1.metric(
    "Attention Score",
    f"{dashboard.get('attention_score',0):.2f}%"
)

    col2.metric(
    "Alertness Score",
    f"{dashboard.get('alertness_score',0):.2f}%"
)

    col3.metric(
    "Final Score",
    f"{dashboard.get('final_score',0):.2f}%"
)
    col4.metric(
    "Pulse (BPM)",
    current_pulse
)

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
else:

        st.info(
            "No students registered yet."
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
            "pulse": 0,
            "status": "Unknown",
            "sleepy_count": 0,
            "distracted_count": 0
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