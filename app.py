"""
app.py  –  Smart Attendance System using Face Recognition
==========================================================
Main Streamlit application.  Run with:
    streamlit run app.py

Pages
-----
1. 🏠 Dashboard         – live stats + daily summary
2. 👤 Register Student  – add student + capture face images
3. 🗄️  Manage Dataset   – view / delete student image data
4. 🔢 Generate Encodings– build / refresh face encoding file
5. 📷 Mark Attendance   – webcam / photo recognition
6. 📋 Attendance Records– browse, filter, export records
7. 📊 Analytics         – Plotly charts
8. 🔒 Admin             – login / session management
"""

import os
import io
import time
import logging
from datetime import date, datetime, timedelta

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ── Local modules ─────────────────────────────────────────────────────────────
from database import (
    init_database, verify_admin,
    add_student, get_all_students, get_student_by_roll, delete_student,
    get_student_count,
    mark_attendance, get_all_attendance, get_attendance_by_date,
    get_attendance_by_roll, get_attendance_count, get_today_attendance_count,
    get_monthly_attendance, get_attendance_stats,
)
from register_student import (
    capture_from_uploaded_image, get_dataset_info, delete_student_dataset,
)
from face_encoder import generate_encodings, load_encodings, encodings_exist
from attendance   import recognise_from_image_bytes, FaceRecognitionEngine
from utils        import (
    inject_css, render_header, stat_card, alert, csv_download_button,
    ensure_directories, today_str, ENCODINGS_FILE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
#  Streamlit page config
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Bootstrap ────────────────────────────────────────────────────────────────
inject_css()
ensure_directories()
init_database()


# ════════════════════════════════════════════════════════════════════════════
#  Session state defaults
# ════════════════════════════════════════════════════════════════════════════

if "logged_in"    not in st.session_state: st.session_state.logged_in    = False
if "username"     not in st.session_state: st.session_state.username     = ""
if "page"         not in st.session_state: st.session_state.page         = "🏠 Dashboard"
if "cam_running"  not in st.session_state: st.session_state.cam_running  = False
if "att_log"      not in st.session_state: st.session_state.att_log      = []


# ════════════════════════════════════════════════════════════════════════════
#  Sidebar navigation
# ════════════════════════════════════════════════════════════════════════════

PAGES = [
    "🏠 Dashboard",
    "👤 Register Student",
    "🗄️ Manage Dataset",
    "🔢 Generate Encodings",
    "📷 Mark Attendance",
    "📋 Attendance Records",
    "📊 Analytics",
    "🔒 Admin",
]

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem;">
        <div style="font-size:2.5rem;">🎓</div>
        <div style="font-weight:800; font-size:1rem; color:#00e5ff; letter-spacing:0.03em;">
            Smart Attendance
        </div>
        <div style="font-size:0.72rem; color:#8899aa; margin-top:2px;">
            Face Recognition System
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        PAGES,
        index=PAGES.index(st.session_state.page),
        label_visibility="collapsed",
    )
    st.session_state.page = page

    st.markdown("---")

    # Login status indicator
    if st.session_state.logged_in:
        st.markdown(f"""
        <div style="background:rgba(0,255,157,0.1);border:1px solid rgba(0,255,157,0.3);
                    border-radius:8px;padding:0.6rem 1rem;font-size:0.82rem;">
            ✅ Logged in as <strong>{st.session_state.username}</strong>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username  = ""
            st.rerun()
    else:
        st.markdown("""
        <div style="background:rgba(255,184,48,0.1);border:1px solid rgba(255,184,48,0.3);
                    border-radius:8px;padding:0.6rem 1rem;font-size:0.82rem;color:#ffb830;">
            ⚠️ Not logged in (Admin only)
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem;color:#445566;text-align:center;line-height:1.6;">
        B.Tech AI/ML Final Year Project<br>
        Smart Attendance System v1.0<br>
        Powered by face_recognition + OpenCV
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  Guard: pages that require admin login
# ════════════════════════════════════════════════════════════════════════════

PROTECTED = {"👤 Register Student", "🗄️ Manage Dataset", "🔢 Generate Encodings"}


def require_login() -> bool:
    if not st.session_state.logged_in:
        st.warning("🔒 Please log in from the **Admin** page to access this feature.")
        return False
    return True


# ════════════════════════════════════════════════════════════════════════════
#  Page 1 – Dashboard
# ════════════════════════════════════════════════════════════════════════════

def page_dashboard():
    render_header(
        "Smart Attendance System",
        "AI-powered face recognition attendance – B.Tech AI/ML Final Year Project",
    )

    # ── Key stats ─────────────────────────────────────────────────────────
    students_total   = get_student_count()
    attendance_total = get_attendance_count()
    today_count      = get_today_attendance_count()
    enc_status       = "✅ Ready" if encodings_exist() else "⚠️ Not generated"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(stat_card(students_total, "Registered Students", "cyan"),
                    unsafe_allow_html=True)
    with c2:
        st.markdown(stat_card(attendance_total, "Total Attendance Records", "green"),
                    unsafe_allow_html=True)
    with c3:
        st.markdown(stat_card(today_count, "Present Today", "purple"),
                    unsafe_allow_html=True)
    with c4:
        st.markdown(stat_card(enc_status, "Encoding Status", "amber"),
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Today's attendance table ──────────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 Today\'s Attendance</div>',
                    unsafe_allow_html=True)
        today_df = get_attendance_by_date(today_str())
        if today_df.empty:
            st.info("No attendance marked today yet.")
        else:
            display_df = today_df[["name", "roll_no", "time", "status"]].rename(columns={
                "name": "Name", "roll_no": "Roll No",
                "time": "Time", "status": "Status",
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Quick Analytics</div>',
                    unsafe_allow_html=True)

        # 7-day trend
        trend_data = []
        for i in range(6, -1, -1):
            d     = (date.today() - timedelta(days=i)).strftime("%Y-%m-%d")
            count = len(get_attendance_by_date(d))
            trend_data.append({"Date": d[-5:], "Count": count})

        trend_df = pd.DataFrame(trend_data)
        fig = px.bar(
            trend_df, x="Date", y="Count",
            color_discrete_sequence=["#00e5ff"],
            template="plotly_dark",
        )
        fig.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            height=220,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Recent registrations ──────────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎓 Recently Registered Students</div>',
                unsafe_allow_html=True)
    s_df = get_all_students()
    if s_df.empty:
        st.info("No students registered yet.")
    else:
        disp = s_df[["name", "roll_no", "branch", "year", "registered_at"]].head(5)
        disp.columns = ["Name", "Roll No", "Branch", "Year", "Registered At"]
        st.dataframe(disp, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  Page 2 – Register Student
# ════════════════════════════════════════════════════════════════════════════

def page_register_student():
    if not require_login():
        return

    render_header("Register New Student", "Capture face images and add to the system")

    col_form, col_cam = st.columns([1, 1])

    with col_form:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📝 Student Details</div>',
                    unsafe_allow_html=True)

        with st.form("register_form", clear_on_submit=False):
            name    = st.text_input("Full Name", placeholder="e.g. Arjun Sharma")
            roll_no = st.text_input("Roll Number", placeholder="e.g. CS2021001")
            branch  = st.selectbox("Branch", [
                "Computer Science", "Information Technology",
                "Electronics & Communication", "Mechanical Engineering",
                "Civil Engineering", "Electrical Engineering", "AI & ML",
            ])
            year    = st.selectbox("Year", ["1st Year", "2nd Year", "3rd Year", "4th Year"])
            submit  = st.form_submit_button("✅ Register Student", use_container_width=True)

        if submit:
            if not name.strip() or not roll_no.strip():
                st.error("Name and Roll Number are required.")
            else:
                res = add_student(name, roll_no, branch, year)
                if res["success"]:
                    st.success(res["message"])
                    st.session_state["reg_name"]    = name
                    st.session_state["reg_roll"]    = roll_no
                    st.session_state["reg_ready"]   = True
                else:
                    st.error(res["message"])

        st.markdown("</div>", unsafe_allow_html=True)

    with col_cam:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📸 Capture Face Images</div>',
                    unsafe_allow_html=True)

        st.info(
            "Upload a clear, well-lit photo of the student's face. "
            "Multiple uploads improve recognition accuracy."
        )

        reg_name = st.session_state.get("reg_name", "")
        reg_roll = st.session_state.get("reg_roll", "")

        if not st.session_state.get("reg_ready", False):
            st.warning("Register the student first (left panel) before capturing images.")
        else:
            st.success(f"Capturing for: **{reg_name}** ({reg_roll})")

            uploaded_files = st.file_uploader(
                "Upload face photos (JPG / PNG)",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="face_upload",
            )

            # Camera input (browser webcam)
            cam_photo = st.camera_input("Or take a photo with your webcam")

            saved_total = 0

            if uploaded_files:
                for idx, uf in enumerate(uploaded_files):
                    res = capture_from_uploaded_image(
                        reg_roll, reg_name, uf.read(), img_index=idx + 1
                    )
                    if res["success"]:
                        saved_total += res["saved"]
                if saved_total:
                    alert(
                        f"✅ Saved {saved_total} image variants for {reg_name}. "
                        "Now go to **Generate Encodings** to update the model.",
                        "success",
                    )

            if cam_photo is not None:
                res = capture_from_uploaded_image(
                    reg_roll, reg_name, cam_photo.getvalue(),
                    img_index=100,
                )
                if res["success"]:
                    alert(f"📸 Webcam photo saved ({res['saved']} variants).", "success")

        st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  Page 3 – Manage Dataset
# ════════════════════════════════════════════════════════════════════════════

def page_manage_dataset():
    if not require_login():
        return

    render_header("Face Dataset Management", "View and manage student face image data")

    # ── Dataset overview ──────────────────────────────────────────────────
    dataset_info = get_dataset_info()
    students_df  = get_all_students()

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🗂️ Dataset Overview</div>',
                unsafe_allow_html=True)

    if not dataset_info:
        st.info("Dataset is empty. Register students and upload face images first.")
    else:
        info_df = pd.DataFrame(dataset_info)
        info_df.columns = ["Roll No", "Name", "Image Count", "Folder"]
        info_df = info_df[["Roll No", "Name", "Image Count"]]
        st.dataframe(info_df, use_container_width=True, hide_index=True)

        total_images = sum(d["image_count"] for d in dataset_info)
        col1, col2, col3 = st.columns(3)
        col1.metric("Students in Dataset", len(dataset_info))
        col2.metric("Total Images",        total_images)
        col3.metric("Avg Images/Student",
                    f"{total_images // max(len(dataset_info), 1)}")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Delete student ────────────────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🗑️ Delete Student Record</div>',
                unsafe_allow_html=True)

    if students_df.empty:
        st.info("No students in the database.")
    else:
        options    = [f"{r['roll_no']} – {r['name']}" for _, r in students_df.iterrows()]
        selected   = st.selectbox("Select student to delete", options)
        roll_del   = selected.split(" – ")[0] if selected else ""

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🗑️ Delete from Database", use_container_width=True):
                res = delete_student(roll_del)
                if res["success"]:
                    st.success(res["message"])
                    st.rerun()
                else:
                    st.error(res["message"])

        with col_btn2:
            if st.button("🗂️ Delete Image Dataset", use_container_width=True):
                res = delete_student_dataset(roll_del)
                if res["success"]:
                    st.success(res["message"])
                    st.rerun()
                else:
                    st.warning(res["message"])

    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  Page 4 – Generate Encodings
# ════════════════════════════════════════════════════════════════════════════

def page_generate_encodings():
    if not require_login():
        return

    render_header("Face Encoding Generator", "Build recognition model from the dataset")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔢 Encoding Status</div>',
                unsafe_allow_html=True)

    if encodings_exist():
        mtime = os.path.getmtime(ENCODINGS_FILE)
        mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        data  = load_encodings()
        n_enc = len(data["encodings"])
        n_stu = len(set(data["rolls"]))
        c1, c2, c3 = st.columns(3)
        c1.metric("Faces Encoded",     n_enc)
        c2.metric("Students Covered",  n_stu)
        c3.metric("Last Updated",      mtime_str)
        alert("✅ Encodings file exists and is ready for recognition.", "success")
    else:
        alert("⚠️ No encodings file found. Click Generate to create it.", "info")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚡ Generate / Refresh Encodings</div>',
                unsafe_allow_html=True)

    st.write(
        "This process scans the dataset folder, detects faces, "
        "and saves numeric encodings that the recognition engine uses."
    )

    dataset_info = get_dataset_info()
    if not dataset_info:
        st.warning("Dataset is empty. Register students and upload face images first.")
    else:
        st.write(f"**{len(dataset_info)} student folder(s)** found in dataset.")

        if st.button("🔢 Generate Encodings", use_container_width=True):
            progress_bar = st.progress(0)
            status_text  = st.empty()
            status_text.text("Processing images…")

            def on_progress(p):
                progress_bar.progress(p)

            with st.spinner("Generating encodings – this may take a moment…"):
                result = generate_encodings(progress_callback=on_progress)

            progress_bar.progress(1.0)

            if result["success"]:
                status_text.empty()
                st.success(result["message"])
                st.balloons()
            else:
                st.error(result["message"])

    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  Page 5 – Mark Attendance
# ════════════════════════════════════════════════════════════════════════════

def page_mark_attendance():
    render_header("Mark Attendance", "Real-time face recognition for attendance")

    if not encodings_exist():
        alert("⚠️ No encodings found. Please generate encodings first.", "error")
        return

    data    = load_encodings()
    n_faces = len(data["encodings"])
    alert(f"✅ Loaded {n_faces} face encodings. Ready for recognition.", "success")

    tab1, tab2 = st.tabs(["📸 Upload / Webcam Photo", "📷 Live Webcam (Local)"])

    # ── Tab 1: single image recognition ──────────────────────────────────
    with tab1:
        col_img, col_res = st.columns([1, 1])

        with col_img:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Upload Photo or Use Webcam</div>',
                        unsafe_allow_html=True)

            input_method = st.radio(
                "Input method", ["Upload Image", "Webcam Capture"], horizontal=True
            )

            image_bytes = None
            if input_method == "Upload Image":
                uf = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
                if uf:
                    image_bytes = uf.read()
                    st.image(image_bytes, caption="Uploaded Image", use_container_width=True)
            else:
                cam_photo = st.camera_input("Take a photo")
                if cam_photo:
                    image_bytes = cam_photo.getvalue()

            if image_bytes and st.button("🔍 Recognise & Mark Attendance",
                                          use_container_width=True):
                with st.spinner("Recognising faces…"):
                    ann_bytes, results = recognise_from_image_bytes(image_bytes)

                with col_res:
                    st.markdown('<div class="section-card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">Recognition Results</div>',
                                unsafe_allow_html=True)
                    st.image(ann_bytes, caption="Annotated", use_container_width=True)

                    if results:
                        for r in results:
                            if r.get("roll_no"):
                                status = "✅ Marked" if r.get("marked") else "ℹ️ Already marked"
                                st.markdown(f"""
                                <div class="section-card" style="margin-bottom:0.6rem;">
                                    <strong>{r['name']}</strong>
                                    &nbsp;<span class="badge badge-cyan">{r['roll_no']}</span>
                                    &nbsp;<span class="badge badge-green">{r['confidence']:.1f}%</span>
                                    <br><small style="color:#8899aa;">{status}</small>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(
                                    '<div class="badge badge-red">Unknown Face</div>',
                                    unsafe_allow_html=True,
                                )
                    else:
                        st.info("No faces detected in the image.")
                    st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    # ── Tab 2: instructions for live webcam ───────────────────────────────
    with tab2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📷 Live Webcam Recognition</div>',
                    unsafe_allow_html=True)
        st.info(
            "Live webcam recognition requires running the application **locally** "
            "(not on Streamlit Cloud). Use the script below or the Upload Photo tab."
        )
        st.code("""
# Run this standalone script for live webcam attendance:
python run_webcam.py
        """, language="bash")

        st.markdown("""
**Instructions for local live recognition:**
1. Ensure `face_recognition` and `opencv-python` are installed
2. Run `streamlit run app.py` on your local machine
3. Use the *Upload Photo* tab, or run `python run_webcam.py` for a standalone window
        """)
        st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  Page 6 – Attendance Records
# ════════════════════════════════════════════════════════════════════════════

def page_attendance_records():
    render_header("Attendance Records", "View, filter and export attendance data")

    tab_all, tab_date, tab_student = st.tabs([
        "📋 All Records", "📅 By Date", "👤 By Student"
    ])

    # ── All records ───────────────────────────────────────────────────────
    with tab_all:
        df = get_all_attendance()
        if df.empty:
            st.info("No attendance records yet.")
        else:
            st.markdown(f"**{len(df)} total records**")
            st.dataframe(
                df[["name", "roll_no", "date", "time", "status"]].rename(columns={
                    "name": "Name", "roll_no": "Roll No",
                    "date": "Date", "time": "Time", "status": "Status",
                }),
                use_container_width=True, hide_index=True,
            )
            csv_download_button(df, "all_attendance.csv")

    # ── By date ───────────────────────────────────────────────────────────
    with tab_date:
        selected_date = st.date_input("Select Date", value=date.today())
        date_str      = selected_date.strftime("%Y-%m-%d")
        df_date       = get_attendance_by_date(date_str)

        st.markdown(f"**{len(df_date)} students present on {date_str}**")
        if df_date.empty:
            st.info("No records for this date.")
        else:
            st.dataframe(
                df_date[["name", "roll_no", "time", "status"]].rename(columns={
                    "name": "Name", "roll_no": "Roll No",
                    "time": "Time", "status": "Status",
                }),
                use_container_width=True, hide_index=True,
            )
            csv_download_button(df_date, f"attendance_{date_str}.csv")

    # ── By student ────────────────────────────────────────────────────────
    with tab_student:
        students = get_all_students()
        if students.empty:
            st.info("No students registered.")
        else:
            opts     = [f"{r['roll_no']} – {r['name']}"
                        for _, r in students.iterrows()]
            sel      = st.selectbox("Select Student", opts)
            roll_sel = sel.split(" – ")[0]
            df_stu   = get_attendance_by_roll(roll_sel)

            if df_stu.empty:
                st.info("No attendance records for this student.")
            else:
                total_days     = get_attendance_count()  # rough proxy
                present        = len(df_stu)
                pct            = round(present / max(total_days, 1) * 100, 1) if total_days else 0

                c1, c2, c3 = st.columns(3)
                c1.metric("Days Present", present)
                c2.metric("Attendance %", f"{present}/{total_days} working days")
                c3.metric("First Marked", df_stu["date"].min())

                st.dataframe(
                    df_stu[["date", "time", "status"]].rename(columns={
                        "date": "Date", "time": "Time", "status": "Status"
                    }),
                    use_container_width=True, hide_index=True,
                )
                csv_download_button(df_stu, f"attendance_{roll_sel}.csv")


# ════════════════════════════════════════════════════════════════════════════
#  Page 7 – Analytics
# ════════════════════════════════════════════════════════════════════════════

def page_analytics():
    render_header("Analytics Dashboard", "Visual insights into attendance patterns")

    all_df = get_all_attendance()
    if all_df.empty:
        st.info("No attendance data available yet.")
        return

    all_df["date"] = pd.to_datetime(all_df["date"])

    # ── Row 1: Attendance % per student  +  Daily trend ──────────────────
    col1, col2 = st.columns(2)

    with col1:
        stats_df = get_attendance_stats()
        if not stats_df.empty:
            fig = px.bar(
                stats_df, x="name", y="attendance_pct",
                title="Attendance % per Student",
                labels={"name": "Student", "attendance_pct": "Attendance %"},
                color="attendance_pct",
                color_continuous_scale=["#ff4b4b", "#ffb830", "#00e5ff", "#00ff9d"],
                template="plotly_dark",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
                xaxis_tickangle=-30,
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        daily = (
            all_df.groupby(all_df["date"].dt.date)
            .size()
            .reset_index(name="count")
        )
        fig2 = px.line(
            daily, x="date", y="count",
            title="Daily Attendance Trend",
            labels={"date": "Date", "count": "Students Present"},
            line_shape="spline",
            color_discrete_sequence=["#00e5ff"],
            template="plotly_dark",
        )
        fig2.update_traces(fill="tozeroy",
                           fillcolor="rgba(0,229,255,0.08)")
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=350,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: Monthly heatmap  +  Present vs Absent pie ─────────────────
    col3, col4 = st.columns(2)

    with col3:
        all_df["month"] = all_df["date"].dt.strftime("%b %Y")
        monthly = (
            all_df.groupby("month")
            .size()
            .reset_index(name="count")
        )
        fig3 = px.bar(
            monthly, x="month", y="count",
            title="Monthly Attendance Volume",
            labels={"month": "Month", "count": "Total Records"},
            color="count",
            color_continuous_scale=["#1a2235", "#9d6bff", "#00e5ff"],
            template="plotly_dark",
        )
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            height=300,
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        total_stu   = get_student_count()
        total_days  = int(all_df["date"].nunique())
        total_slots = total_stu * total_days
        total_pres  = len(all_df)
        total_abs   = max(total_slots - total_pres, 0)

        fig4 = go.Figure(go.Pie(
            labels=["Present", "Absent"],
            values=[total_pres, total_abs],
            hole=0.55,
            marker_colors=["#00ff9d", "#ff4b4b"],
            textfont_size=13,
        ))
        fig4.update_layout(
            title="Present vs Absent (Overall)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            template="plotly_dark",
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Row 3: Top attenders ──────────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏆 Top Attenders</div>', unsafe_allow_html=True)
    stats_df2 = get_attendance_stats()
    if not stats_df2.empty:
        top5 = stats_df2.head(5)
        for _, row in top5.iterrows():
            pct   = float(row["attendance_pct"])
            color = "#00ff9d" if pct >= 75 else "#ffb830" if pct >= 50 else "#ff4b4b"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.6rem;">
                <div style="width:150px;font-weight:600;">{row['name']}</div>
                <div style="flex:1;background:#1a2235;border-radius:20px;height:10px;">
                    <div style="width:{min(pct,100):.0f}%;background:{color};
                               height:100%;border-radius:20px;"></div>
                </div>
                <div style="width:60px;text-align:right;font-family:'JetBrains Mono',monospace;
                            color:{color};font-weight:700;">{pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  Page 8 – Admin Login
# ════════════════════════════════════════════════════════════════════════════

def page_admin():
    render_header("Admin Panel", "Secure login for administrative access")

    if st.session_state.logged_in:
        st.success(f"✅ Logged in as **{st.session_state.username}**")

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">⚙️ System Information</div>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"- **Students registered:** {get_student_count()}")
            st.markdown(f"- **Total attendance records:** {get_attendance_count()}")
            st.markdown(f"- **Today's attendance:** {get_today_attendance_count()}")
        with col2:
            st.markdown(f"- **Encodings ready:** {'Yes' if encodings_exist() else 'No'}")
            from utils import DATASET_DIR, ENCODINGS_FILE, DB_DIR
            st.markdown(f"- **Dataset path:** `{DATASET_DIR}`")
            st.markdown(f"- **DB path:** `{DB_DIR}/attendance.db`")

        st.markdown("</div>", unsafe_allow_html=True)

        # Export full DB
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📤 Export Data</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            all_att = get_all_attendance()
            if not all_att.empty:
                csv_download_button(all_att, "full_attendance_export.csv",
                                    "⬇️ Export All Attendance")
        with c2:
            all_stu = get_all_students()
            if not all_stu.empty:
                csv_download_button(all_stu, "students_export.csv",
                                    "⬇️ Export Student List")
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🔓 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username  = ""
            st.rerun()

    else:
        col_mid = st.columns([1, 1.2, 1])[1]
        with col_mid:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🔒 Admin Login</div>',
                        unsafe_allow_html=True)
            st.markdown(
                "Default credentials: **admin** / **admin123**",
                unsafe_allow_html=True,
            )
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="admin")
                password = st.text_input("Password", type="password",
                                          placeholder="••••••••")
                login_btn = st.form_submit_button("🔑 Login", use_container_width=True)

            if login_btn:
                if verify_admin(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username  = username
                    st.success("Login successful! Redirecting…")
                    time.sleep(0.8)
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

            st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  Router
# ════════════════════════════════════════════════════════════════════════════

PAGE_ROUTER = {
    "🏠 Dashboard":          page_dashboard,
    "👤 Register Student":   page_register_student,
    "🗄️ Manage Dataset":     page_manage_dataset,
    "🔢 Generate Encodings": page_generate_encodings,
    "📷 Mark Attendance":    page_mark_attendance,
    "📋 Attendance Records": page_attendance_records,
    "📊 Analytics":          page_analytics,
    "🔒 Admin":              page_admin,
}

current_page = st.session_state.get("page", "🏠 Dashboard")
if current_page in PAGE_ROUTER:
    PAGE_ROUTER[current_page]()
else:
    page_dashboard()
