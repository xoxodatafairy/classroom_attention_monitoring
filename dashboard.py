import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

API_URL = "http://127.0.0.1:8000"

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

def fetch_data(endpoint, default=None):

    try:

        response = requests.get(
            f"{API_URL}{endpoint}",
            timeout=10
        )
        
        

        if response.status_code == 200:

            return response.json()

    except Exception:

        pass

    return default


# ==================================================
# LOAD DATA
# ==================================================

dashboard = fetch_data("/dashboard", {})
analytics = fetch_data("/analytics", {})
detections = fetch_data("/detections", {})
teacher_alert = fetch_data("/teacher-alert", {})
students = fetch_data("/students", {})
history = fetch_data(
    "/attention-history",
    {}
)

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

if st.button("Refresh Dashboard"):

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

# ==================================================
# DASHBOARD TAB
# ==================================================

with dashboard_tab:

    col1, col2, col3 = st.columns(3)

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

    st.subheader("Classroom Status")

    status = dashboard.get(
        "status",
        "Unknown"
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
                "text":
                "Classroom Performance"
            },
            gauge={
                "axis": {
                    "range": [0, 100]
                },
                "bar": {
                    "color":
                    "royalblue"
                }
            }
        )
    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

    st.subheader(
        "Behavior Distribution"
    )

    chart_df = pd.DataFrame({

        "Behavior": [
            "Attentive",
            "Distracted",
            "Sleepy",
            "Phone Use"
        ],

        "Count": [

            analytics.get(
                "attentive_students",
                0
            ),

            analytics.get(
                "distracted_students",
                0
            ),

            analytics.get(
                "sleepy_students",
                0
            ),

            analytics.get(
                "phone_usage",
                0
            )
        ]
    })

    fig = px.bar(
        chart_df,
        x="Behavior",
        y="Count",
        color="Behavior"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Live Detections"
    )

    st.json(detections)

    st.subheader(
        "Teacher Alert"
    )

    alert_color = teacher_alert.get(
        "alert_color",
        "RED"
    )

    score = teacher_alert.get(
        "score",
        0
    )

    teacher_status = teacher_alert.get(
        "status",
        "Unknown"
    )

    if alert_color == "GREEN":

        st.success(
            f"{teacher_status} | Score: {score}"
        )

    elif alert_color == "YELLOW":

        st.warning(
            f"{teacher_status} | Score: {score}"
        )

    else:

        st.error(
            f"{teacher_status} | Score: {score}"
        )

# ==================================================
# STUDENTS TAB
# ==================================================

with students_tab:

    st.subheader(
        "Student Analytics"
    )

    rows = []

    for student_id, data in students.items():

        rows.append({

            "Student ID":
            student_id,

            "Name":
            data.get(
                "name",
                "Unknown"
            ),

            "Attention Rate":
            data.get(
                "attention_rate",
                0
            ),

            "Status":
            data.get(
                "status",
                "Unknown"
            )
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
            use_container_width=True
        )
with register_tab:

    st.subheader(
        "Student Registration"
    )

    student_id = st.text_input(
        "Student ID"
    )

    student_name = st.text_input(
        "Student Name"
    )

    photo = st.file_uploader(
        "Student Photo",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if st.button(
        "Register Student"
    ):

        if photo:

            files = {

                "file": (

                    photo.name,

                    photo.getvalue(),

                    photo.type
                )
            }

            data = {

                "student_id":
                student_id,

                "student_name":
                student_name
            }

            response = requests.post(

                f"{API_URL}/register-student",

                files=files,

                data=data
            )

            st.success(
                response.json()
            )

        else:

            st.error(
                "Please upload a photo."
            )
# ==================================================
# MONITORING TAB
# ==================================================

with monitoring_tab:

    st.subheader(
        "Camera Configuration"
    )

    source = st.selectbox(

        "Camera Source",

        [
            "Laptop Webcam",
            "iPhone Stream",
            "CCTV RTSP"
        ]
    )

    if source == "iPhone Stream":

        st.text_input(
            "iPhone Stream URL"
        )

    elif source == "CCTV RTSP":

        st.text_input(
            "RTSP URL"
        )

    c1, c2 = st.columns(2)

    with c1:

        st.button(
            "Start Monitoring"
        )

    with c2:

        st.button(
            "Stop Monitoring"
        )

    st.divider()

    st.subheader(
        "Image Analysis"
    )

    uploaded_image = st.file_uploader(
        "Upload Classroom Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_image:

        st.image(
            uploaded_image,
            use_container_width=True
        )

        if st.button(
            "Analyze Image"
        ):

            files = {

                "file": (

                    uploaded_image.name,

                    uploaded_image.getvalue(),

                    uploaded_image.type
                )
            }

            result = requests.post(

                f"{API_URL}/analyze-image",

                files=files,

                timeout=180
            ).json()

            st.success(
                "Analysis Complete"
            )

            st.json(result)

    st.divider()

    st.subheader(
        "Camera Snapshot"
    )

    snapshot = st.camera_input(
        "Take Photo"
    )

    if snapshot:

        if st.button(
            "Analyze Snapshot"
        ):

            files = {

                "file": (

                    "snapshot.jpg",

                    snapshot.getvalue(),

                    "image/jpeg"
                )
            }

            result = requests.post(

                f"{API_URL}/analyze-image",

                files=files,

                timeout=180
            ).json()

            st.json(result)
if history:

    trend_df = pd.DataFrame({

        "Time":
        history.get(
            "labels",
            []
        ),

        "Attention":
        history.get(
            "attention",
            []
        ),

        "Engagement":
        history.get(
            "engagement",
            []
        )
    })

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
            width="stretch"
        )

# ==================================================
# REPORTS TAB
# ==================================================

with reports_tab:

    st.subheader("Reports")

    st.button(
        "Export PDF"
    )

    st.button(
        "Export CSV"
    )

    st.info(
        "Historical reports and trend analytics will appear here."
    )