# 🎓 Smart Attendance System using Face Recognition

> **B.Tech AI/ML Final Year Project**  
> Real-time automated attendance marking powered by face recognition, OpenCV, and Streamlit

---

## 📌 Project Overview

| Item | Detail |
|------|--------|
| **Project Name** | Smart Attendance System using Face Recognition |
| **Domain** | Artificial Intelligence / Computer Vision |
| **Tech Stack** | Python · Streamlit · OpenCV · face_recognition · SQLite · Plotly |
| **Purpose** | Automate classroom attendance using webcam-based facial recognition |

---

## 🏗️ Folder Structure

```
attendance_system/
│
├── app.py                  ← Main Streamlit application (router + all pages)
├── database.py             ← SQLite CRUD layer (students + attendance)
├── attendance.py           ← Face recognition engine + attendance marking
├── register_student.py     ← Face capture, augmentation, dataset management
├── face_encoder.py         ← Encoding generation & pickle persistence
├── utils.py                ← Shared helpers, CSS, UI components
├── run_webcam.py           ← Standalone live webcam script (local only)
├── seed_demo_data.py       ← Populate DB with realistic demo data
├── requirements.txt        ← Python dependencies
│
├── dataset/                ← Per-student face images  (auto-created)
│   └── CS2021001_Arjun_Sharma/
│       ├── CS2021001_1.jpg
│       └── …
│
├── encodings/              ← Pickle file with face encodings  (auto-created)
│   └── face_encodings.pkl
│
├── reports/                ← Exported CSV reports  (auto-created)
│
└── database/               ← SQLite database  (auto-created)
    └── attendance.db
```

---

## ✨ Features

### 1. 🏠 Home Dashboard
- Live statistics: registered students, total records, today's count
- Encoding status indicator
- 7-day attendance bar chart (Plotly)
- Recent registrations table
- Today's attendance summary

### 2. 👤 Student Registration
- Form: Name, Roll No, Branch, Year
- Upload face photos (multiple) **or** use in-browser webcam
- Automatic image augmentation (flip, brightness variants) for robustness
- Instant database insert with duplicate-roll-no protection

### 3. 🗄️ Dataset Management
- View all registered students with image counts
- Delete student from database (cascade deletes attendance)
- Delete image dataset separately
- Dataset statistics (total images, average per student)

### 4. 🔢 Face Encoding Generator
- Scans all dataset images
- Uses `face_recognition` (dlib) to extract 128-d face embeddings
- Progress bar during generation
- Saves to `encodings/face_encodings.pkl`
- Shows last-updated timestamp and coverage stats

### 5. 📷 Attendance Marking
- **Upload photo / webcam snapshot** → annotated result with name + roll + confidence
- **Live webcam** (local) via `run_webcam.py` with real-time bounding boxes
- Cooldown prevents duplicate marking in quick succession
- Unknown face detection

### 6. 📋 Attendance Records
- View all records, filter by date or student
- CSV export for every view
- Individual student attendance history

### 7. 📊 Analytics Dashboard
- Attendance % per student (bar chart)
- Daily trend line chart
- Monthly volume bar chart
- Present vs Absent donut chart
- Top-5 attenders with progress bars

### 8. 🔒 Admin Panel
- SHA-256 hashed password authentication
- Session-based login state
- System info: file paths, record counts
- Export full DB as CSV
- Default: `admin` / `admin123`

---

## 🚀 Local Setup & Installation

### Prerequisites
- Python 3.9 – 3.11 (3.12 may have dlib compatibility issues)
- `cmake` installed (required by dlib)
- Webcam (for live recognition)

### Step-by-step

```bash
# 1. Clone / download the project
git clone <your-repo-url>
cd attendance_system

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install system deps (Ubuntu / Debian)
sudo apt-get update
sudo apt-get install -y cmake build-essential libboost-all-dev

# macOS (Homebrew)
brew install cmake boost

# 4. Install Python packages
pip install -r requirements.txt

# 5. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 🎬 Quick Demo (seed fake data first)

```bash
python seed_demo_data.py   # populates DB with 8 students + 30 days of attendance
streamlit run app.py
```

---

## 📸 How to Add a Real Student

1. Go to **🔒 Admin** → login (`admin` / `admin123`)
2. Go to **👤 Register Student** → fill in the form → submit
3. Upload 3–5 clear face photos (different angles, lighting)
4. Go to **🔢 Generate Encodings** → click Generate
5. Go to **📷 Mark Attendance** → upload a photo → click Recognise

---

## 📷 Live Webcam (Local Only)

```bash
python run_webcam.py
```

- Opens full-screen camera window
- Draws bounding box + name + confidence in real time
- Marks attendance automatically; press **Q** to quit

---

## 🗄️ Database Schema

```sql
-- Students
CREATE TABLE students (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL,
    roll_no      TEXT    NOT NULL UNIQUE,
    branch       TEXT    NOT NULL,
    year         TEXT    NOT NULL,
    registered_at TEXT   DEFAULT (datetime('now','localtime'))
);

-- Attendance (one row per student per day)
CREATE TABLE attendance (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    roll_no TEXT    NOT NULL,
    name    TEXT    NOT NULL,
    date    TEXT    NOT NULL,          -- YYYY-MM-DD
    time    TEXT    NOT NULL,          -- HH:MM:SS
    status  TEXT    NOT NULL DEFAULT 'Present',
    UNIQUE(roll_no, date)              -- prevents duplicates
);

-- Admins
CREATE TABLE admins (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT    NOT NULL UNIQUE,
    password TEXT    NOT NULL          -- SHA-256 hash
);
```

---

## ☁️ Streamlit Community Cloud Deployment

> **Note:** `face_recognition` depends on `dlib` which requires native compilation.
> Streamlit Cloud's build environment supports it, but the webcam features work
> only locally. The Upload Photo recognition tab works on cloud.

### Steps

1. Push the project to a **public GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New App**
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Add a `packages.txt` in the repo root:

```
# packages.txt  (system packages for Streamlit Cloud)
cmake
build-essential
libboost-all-dev
libopenblas-dev
```

6. Click **Deploy** — build takes ~5 minutes

### Caveats on Cloud
| Feature | Cloud | Local |
|---------|-------|-------|
| Dashboard, Records, Analytics | ✅ | ✅ |
| Student Registration (upload) | ✅ | ✅ |
| Photo-based recognition | ✅ | ✅ |
| Live webcam recognition | ❌ | ✅ |
| Dataset persistence across deploys | ❌* | ✅ |

*Use a persistent store (S3, GCS) or SQLite hosted externally for production.

---

## 🔧 Tuning Parameters

| Parameter | File | Default | Effect |
|-----------|------|---------|--------|
| `TOLERANCE` | `attendance.py` | `0.52` | Lower → stricter match |
| `RECOGNITION_WAIT` | `attendance.py` | `3.0 s` | Cooldown between markings |
| `CAPTURE_COUNT` | `register_student.py` | `20` | Images per webcam session |

---

## 🧩 Project Workflow

```
Register Student
    │  upload photos
    ▼
Dataset folder
    │  face_encoder.py
    ▼
encodings/face_encodings.pkl
    │  attendance.py
    ▼
Webcam / uploaded image  ──►  Face detected?
                                │  Yes → match encodings
                                │         confidence > threshold?
                                │              │ Yes → mark attendance (SQLite)
                                │              └ No  → Unknown face
                                └  No  → skip frame
```

---

## 📦 requirements.txt

```
streamlit==1.32.0
opencv-python-headless==4.9.0.80
face-recognition==1.3.0
face-recognition-models==0.3.0
numpy==1.26.4
pandas==2.2.1
Pillow==10.2.0
plotly==5.20.0
scikit-learn==1.4.1.post1
dlib==19.24.4
imutils==0.5.4
```

---

## 👨‍💻 Author

**[Your Name]**  
B.Tech – Artificial Intelligence & Machine Learning  
[Your College Name]  
[Your Roll Number]

---

## 📄 License

MIT License — free to use, modify and distribute with attribution.
