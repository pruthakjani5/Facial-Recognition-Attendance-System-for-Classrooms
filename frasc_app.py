import streamlit as st
import os
import gc
import time

# Set page configuration first
st.set_page_config(
    page_title="FRASC: Face Recognition Attendance System for Classes",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create directory for training images if it doesn't exist
def create_directories():
    if not os.path.exists('Training_images'):
        os.makedirs('Training_images')
    if not os.path.exists('temp'):
        os.makedirs('temp')
    return True

# Import regular dependencies first
import cv2
import numpy as np
from datetime import datetime
import csv
import pandas as pd
from PIL import Image
import shutil
from io import BytesIO
import base64
import tempfile
import zipfile

# Import face_recognition with better error handling
try:
    # Show loading message
    loading_message = st.empty()
    loading_message.info("Loading face recognition system... Please wait.")
    
    # Clean memory before importing
    gc.collect()
    
    # Try importing without timeouts (which cause issues in Streamlit Cloud)
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    loading_message.success("Face recognition loaded successfully!")
    time.sleep(1)  # Give users time to see success
    loading_message.empty()
except ImportError as e:
    st.error(f"Face recognition library not available: {e}")
    FACE_RECOGNITION_AVAILABLE = False
except Exception as e:
    st.error(f"Unexpected error importing face_recognition: {str(e)}")
    FACE_RECOGNITION_AVAILABLE = False

# For webcam on Streamlit Cloud - import with error handling
try:
    # Install required packages if not available
    import subprocess
    import sys
    
    def install_package(package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    try:
        import streamlit_webrtc
    except ImportError:
        st.info("Installing streamlit-webrtc package...")
        install_package("streamlit-webrtc")
        
    try:
        import av
    except ImportError:
        st.info("Installing av package...")
        install_package("av")
    
    # Now import after installation
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
    import av
    WEBRTC_AVAILABLE = True
except Exception as e:
    WEBRTC_AVAILABLE = False
    st.warning(f"WebRTC not available - webcam functionality disabled: {e}")

# Custom CSS for a professional UI with dark mode compatibility
st.markdown("""
<style>
    /* General reset for better dark mode compatibility */
    .stApp {
        --text-color: var(--text-color, #2c3e50);
        --bg-color: var(--background-color, #ffffff);
        --accent-color: #3498db;
        --accent-hover: #2980b9;
        --border-color: rgba(49, 51, 63, 0.2);
        --card-bg: rgba(255, 255, 255, 0.1);
    }
    
    /* Main header with dark mode support */
    .main-header {
        font-size: 32px;
        font-weight: bold;
        color: var(--text-color, #2c3e50);
        text-align: center;
        background-color: rgba(52, 152, 219, 0.1);
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 30px;
        border-bottom: 3px solid #3498db;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Section headers with better visibility in both modes */
    .section-header {
        padding: 15px;
        background-color: rgba(52, 152, 219, 0.1);
        border-left: 5px solid #3498db;
        margin-bottom: 20px;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Ensure all headings in section headers are visible */
    .section-header h1, .section-header h2, .section-header h3, 
    .section-header h4, .section-header h5, .section-header h6 {
        color: var(--text-color, #2c3e50) !important;
        font-weight: 600;
        margin: 0;
    }
    
    /* Enhanced buttons */
    .stButton > button {
        background-color: #3498db;
        color: white !important;
        font-weight: 500;
        border: none;
        padding: 10px 24px;
        border-radius: 6px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    /* Improved input fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div,
    .stMultiselect > div > div > div {
        border-radius: 6px;
        border: 1px solid var(--border-color);
        padding: 8px 12px;
        transition: all 0.2s;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus,
    .stMultiselect > div > div > div:focus {
        border-color: #3498db;
        box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
    }
    
    /* Attendance data card */
    .attendance-data {
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 15px;
        background-color: var(--card-bg);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    /* Status indicators */
    .status-info {
        background-color: rgba(52, 152, 219, 0.1);
        border-left: 4px solid #3498db;
        padding: 10px 15px;
        border-radius: 0 4px 4px 0;
        margin: 10px 0;
    }
    
    .status-success {
        background-color: rgba(46, 204, 113, 0.1);
        border-left: 4px solid #2ecc71;
        padding: 10px 15px;
        border-radius: 0 4px 4px 0;
        margin: 10px 0;
    }
    
    .status-warning {
        background-color: rgba(241, 196, 15, 0.1);
        border-left: 4px solid #f1c40f;
        padding: 10px 15px;
        border-radius: 0 4px 4px 0;
        margin: 10px 0;
    }
    
    .status-error {
        background-color: rgba(231, 76, 60, 0.1);
        border-left: 4px solid #e74c3c;
        padding: 10px 15px;
        border-radius: 0 4px 4px 0;
        margin: 10px 0;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        background-color: transparent;
        border-bottom: 2px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(52, 152, 219, 0.1) !important;
        border-bottom: 2px solid #3498db !important;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-collapse: collapse !important;
        border: none;
        border-radius: 8px;
        overflow: hidden;
        width: 100%;
    }
    
    .dataframe th {
        background-color: rgba(52, 152, 219, 0.3);
        padding: 12px !important;
        color: var(--text-color, #2c3e50) !important;
    }
    
    .dataframe td {
        padding: 10px !important;
        border-bottom: 1px solid var(--border-color);
    }
    
    /* Download link styling */
    .download-button {
        background-color: #3498db;
        color: white !important;
        text-decoration: none;
        padding: 10px 16px;
        border-radius: 6px;
        font-weight: 500;
        display: inline-block;
        margin: 10px 0;
        transition: all 0.2s;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    
    .download-button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)

# Create directory for training images if it doesn't exist
@st.cache_data
def create_directories():
    try:
        # If Training_images exists but is not a directory, remove it
        if os.path.exists('Training_images') and not os.path.isdir('Training_images'):
            try:
                os.remove('Training_images')
            except Exception as e:
                st.error(f"Could not remove non-directory Training_images: {e}")
        
        # Create Training_images directory if it doesn't exist
        if not os.path.exists('Training_images'):
            try:
                os.makedirs('Training_images')
            except Exception as e:
                st.error(f"Could not create Training_images directory: {e}")
        
        # Handle temp directory the same way
        if os.path.exists('temp') and not os.path.isdir('temp'):
            try:
                os.remove('temp')
            except Exception as e:
                st.error(f"Could not remove non-directory temp: {e}")
        
        if not os.path.exists('temp'):
            try:
                os.makedirs('temp')
            except Exception as e:
                st.error(f"Could not create temp directory: {e}")
                
        return True
    except Exception as e:
        st.error(f"Error in create_directories: {e}")
        return False

# Load and encode faces from disk or memory
@st.cache_data(ttl=60)  # Cache for 1 minute, allowing frequent refreshes
def load_and_encode_faces():
    if not FACE_RECOGNITION_AVAILABLE:
        return [], []
    
    images = []
    class_names = []
    
    # Try to load from disk first
    disk_loading_succeeded = False
    try:
        if os.path.exists('Training_images') and os.path.isdir('Training_images'):
            files = os.listdir('Training_images')
            student_images = [f for f in files if os.path.isfile(os.path.join('Training_images', f)) 
                            and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            st.sidebar.info(f"Found {len(student_images)} images in Training_images directory")
            
            for img_file in student_images:
                try:
                    img_path = os.path.join('Training_images', img_file)
                    cur_img = cv2.imread(img_path)
                    if cur_img is not None:
                        images.append(cur_img)
                        class_names.append(os.path.splitext(img_file)[0])
                    else:
                        st.warning(f"Image loaded as None: {img_file}")
                except Exception as e:
                    st.warning(f"Could not load image {img_file} from disk: {e}")
            
            if images:
                disk_loading_succeeded = True
                st.sidebar.success(f"Successfully loaded {len(images)} images from disk")
    except Exception as e:
        st.warning(f"Error loading images from disk: {e}")
    
    # If disk loading failed or no images were found, try to load from memory
    if not disk_loading_succeeded and 'in_memory_training_images' in st.session_state:
        memory_image_count = len(st.session_state.in_memory_training_images)
        st.info(f"Using in-memory storage for {memory_image_count} training images")
        
        for filename, img in st.session_state.in_memory_training_images.items():
            if img is not None:
                images.append(img)
                class_names.append(os.path.splitext(filename)[0])
    
    # If we still have no images, return empty lists
    if not images:
        st.warning("No images found in both disk and memory storage")
        return [], []
    
    # Encode faces
    encode_list = []
    valid_indices = []
    
    with st.spinner(f"Encoding {len(images)} faces..."):
        for i, img in enumerate(images):
            try:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(img_rgb)
                if face_encodings:
                    encode = face_encodings[0]
                    encode_list.append(encode)
                    valid_indices.append(i)
                else:
                    st.warning(f"No face detected in image for {class_names[i]}")
            except Exception as e:
                st.error(f"Error encoding image for {class_names[i]}: {e}")
    
    # Only keep class names for successfully encoded faces
    class_names_final = [class_names[i] for i in valid_indices]
    
    if encode_list:
        st.sidebar.success(f"Successfully encoded {len(encode_list)} faces")
    else:
        st.error("No faces could be encoded. Please check your images.")
    
    return encode_list, class_names_final

def mark_attendance(name, faculty_name, lecture_name):
    filename = f"Attendance_{faculty_name}_{lecture_name}.csv"
    header = ["Date", "Time", "Faculty", "Lecture", "Name", "Attendance"]
    
    # Clean the name to remove any commas that might cause CSV issues
    cleaned_name = str(name).replace(',', '')
    
    now = datetime.now()
    # Using IST time (UTC+5:30) for Streamlit Cloud deployment
    indian_timezone = now.astimezone(datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
    today = indian_timezone.strftime("%d-%m-%Y")
    current_time = indian_timezone.strftime("%H:%M:%S")

    current_time = now.strftime("%H:%M:%S")
    
    # Use a lock mechanism to prevent duplicate entries from concurrent calls
    lock_file = f"{filename}.lock"
    
    # Create a lock file to prevent concurrent access
    try:
        # Check if lock exists and is recent (less than 5 seconds old)
        if os.path.exists(lock_file) and (time.time() - os.path.getmtime(lock_file)) < 5:
            # Wait a moment for the lock to clear
            time.sleep(0.5)
            if os.path.exists(lock_file) and (time.time() - os.path.getmtime(lock_file)) < 5:
                # Still locked, add message and return
                if 'attendance_messages' not in st.session_state:
                    st.session_state.attendance_messages = []
                st.session_state.attendance_messages.append(f'<div class="status-warning">System busy, please try again.</div>')
                return False
        
        # Create the lock file
        with open(lock_file, 'w') as f:
            f.write(str(time.time()))
        
        # Check if student already took attendance for this lecture today
        found = False
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                if not df.empty:
                    # Clean the existing names to ensure consistent comparison
                    df['Name'] = df['Name'].astype(str).str.replace(',', '')
                    
                    # Check if this student has an entry for today's lecture
                    if ((df['Name'] == cleaned_name) & (df['Date'] == today) & (df['Lecture'] == lecture_name)).any():
                        found = True
            except pd.errors.EmptyDataError:
                pass
            except Exception as e:
                st.error(f"Error reading attendance file for duplicate check: {e}")

        if found:
            if 'attendance_messages' not in st.session_state:
                st.session_state.attendance_messages = []
            st.session_state.attendance_messages.append(f'<div class="status-warning">Student \'{cleaned_name}\' already marked for {lecture_name} today.</div>')
            return False
        else:
            # Open file in append mode
            file_exists = os.path.isfile(filename) and os.stat(filename).st_size > 0
            with open(filename, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow({
                    "Date": today, 
                    "Time": current_time, 
                    "Faculty": faculty_name,
                    "Lecture": lecture_name,
                    "Name": cleaned_name, 
                    "Attendance": 1
                })
            
            if 'attendance_messages' not in st.session_state:
                st.session_state.attendance_messages = []
            st.session_state.attendance_messages.append(f'<div class="status-success">Attendance marked for {cleaned_name}.</div>')
            return True
    
    finally:
        # Always remove the lock file when done
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except:
                pass

# Function to process an attendance image
def process_attendance_image(image_file, known_encodings, class_names, faculty_name, lecture_name):
    if not FACE_RECOGNITION_AVAILABLE:
        st.error("Face recognition is not available")
        return None, []
    
    # Read the image file
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        st.error("Could not decode image")
        return None, []
    
    marked_names_in_current_image = []
    
    try:
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        faces_cur_frame = face_recognition.face_locations(imgS)
        encodes_cur_frame = face_recognition.face_encodings(imgS, faces_cur_frame)
        
        for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
            matches = face_recognition.compare_faces(known_encodings, encode_face)
            face_dis = face_recognition.face_distance(known_encodings, encode_face)
            
            name = "Unknown"
            if len(face_dis) > 0:
                match_index = np.argmin(face_dis)
                if matches[match_index]:
                    name = class_names[match_index]
            
            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            
            if name != "Unknown":
                if mark_attendance(name, faculty_name, lecture_name):
                    marked_names_in_current_image.append(name)
                    
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb, marked_names_in_current_image
    
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, []

# Video Transformer for live webcam attendance
if WEBRTC_AVAILABLE and FACE_RECOGNITION_AVAILABLE:
    class FaceRecognitionTransformer(VideoTransformerBase):
        def __init__(self, known_encodings, class_names, faculty_name, lecture_name):
            self.known_encodings = known_encodings
            self.class_names = class_names
            self.faculty_name = faculty_name
            self.lecture_name = lecture_name
            self.marked_names_session = set()

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            
            try:
                # Process the frame
                small_frame = cv2.resize(img, (0, 0), None, 0.25, 0.25)
                small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                face_locations = face_recognition.face_locations(small_frame_rgb)
                face_encodings = face_recognition.face_encodings(small_frame_rgb, face_locations)
                
                current_frame_marked = []
                
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    
                    # Draw rectangle around the face
                    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    # Compare with known faces
                    matches = face_recognition.compare_faces(self.known_encodings, face_encoding)
                    name = "Unknown"
                    
                    if len(matches) > 0:
                        face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.class_names[best_match_index]
                    
                    # Display name
                    cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    
                    if name != "Unknown" and name not in self.marked_names_session:
                        if mark_attendance(name, self.faculty_name, self.lecture_name):
                            self.marked_names_session.add(name)
                            current_frame_marked.append(name)
                
                # Update the session state for displaying marked names in the UI
                if current_frame_marked:
                    if 'live_marked_names' not in st.session_state:
                        st.session_state.live_marked_names = set()
                    st.session_state.live_marked_names.update(current_frame_marked)
            
            except Exception as e:
                # Log error but continue processing
                pass

            return av.VideoFrame.from_ndarray(img, format="bgr24")

# Function to get a download link for the attendance file
def get_csv_download_link(faculty_name, lecture_name):
    filename = f"Attendance_{faculty_name}_{lecture_name}.csv"
    if os.path.exists(filename):
        try:
            # Read file with proper error handling
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    csv_data = f.read()
            except UnicodeDecodeError:
                # Try with a different encoding if utf-8 fails
                with open(filename, 'r', encoding='latin-1') as f:
                    csv_data = f.read()
            
            # Create base64 encoded data
            b64 = base64.b64encode(csv_data.encode()).decode()
            
            # Create HTML link with better styling
            href = f'''
            <a href="data:file/csv;base64,{b64}" 
               download="{filename}" 
               class="download-button"
               style="text-decoration:none;">
                <span style="display:flex;align-items:center;justify-content:center;gap:8px;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="7 10 12 15 17 10"></polyline>
                        <line x1="12" y1="15" x2="12" y2="3"></line>
                    </svg>
                    Download Attendance CSV
                </span>
            </a>
            '''
            return href
        except Exception as e:
            st.error(f"Error creating download link: {e}")
            return None
    else:
        return None

# Main Streamlit app
def main():
    # Initialize in-memory image storage if not exists
    if 'in_memory_training_images' not in st.session_state:
        st.session_state.in_memory_training_images = {}
        
    # Set storage priority flag for the app
    if 'storage_priority' not in st.session_state:
        st.session_state.storage_priority = 'disk'  # Default to disk storage with memory as fallback
    # Application info section in sidebar
    st.sidebar.markdown("## About FRASC")
    st.sidebar.markdown("""
    **FRASC** (Face Recognition Attendance System for Classes) is an AI-powered 
    attendance management solution designed for educational institutions.
    
    This application uses state-of-the-art facial recognition technology to:
    - Automatically mark student attendance
    - Generate digital attendance records
    - Streamline administrative processes
    """)
    
    # How to use section
    st.sidebar.markdown("## How to Use")
    st.sidebar.markdown("""
    1. **Setup Tab:**
       - Enter faculty and lecture details
       - Upload student images individually or via ZIP file
       - Review the student database
    
    2. **Take Attendance Tab:**
       - Choose between image upload or webcam
       - Process images to mark student attendance
       - View real-time recognition results
    
    3. **Attendance Data:**
       - Review attendance records
       - Download CSV files for record-keeping
    """)
    
    # Tips section with collapsible content
    with st.sidebar.expander("Tips & Best Practices"):
        st.markdown("""
        - **Image Quality:** Use clear, well-lit photos for better recognition
        - **Face Angle:** Front-facing images work best
        - **Group Photos:** Ensure all faces are visible when using group images
        - **Webcam Usage:** Ensure proper lighting when using webcam mode
        - **Database:** Add multiple images per student for improved accuracy
        """)
    
    st.sidebar.markdown("---")
    # Initialize directories
    create_directories()
    
    # Check system status
    system_status = []
    if not FACE_RECOGNITION_AVAILABLE:
        system_status.append("❌ Face Recognition Library")
    else:
        system_status.append("✅ Face Recognition Library")
    
    if not WEBRTC_AVAILABLE:
        system_status.append("❌ WebRTC (Webcam disabled)")
    else:
        system_status.append("✅ WebRTC")
    
    # Display header with logo and title
    st.markdown('''
    <div class="main-header">
        <div style="display: flex; align-items: center; justify-content: center; gap: 15px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#3498db" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"></path>
                <circle cx="12" cy="7" r="4"></circle>
            </svg>
            <span>Face Recognition Attendance System for Classes</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Display system status
    st.sidebar.markdown("### System Status")
    for status in system_status:
        st.sidebar.markdown(status)
    
    if not FACE_RECOGNITION_AVAILABLE:
        st.markdown('<div class="status-error">⚠️ Face recognition functionality is not available. Please check the deployment configuration.</div>', unsafe_allow_html=True)
        st.stop()
    
    # Create tabs
    tab1, tab2 = st.tabs(["Setup", "Take Attendance"])
    
    # Initialize session state variables if they don't exist
    if 'faculty_name' not in st.session_state:
        st.session_state['faculty_name'] = ""
    if 'lecture_name' not in st.session_state:
        st.session_state['lecture_name'] = ""
    if 'attendance_messages' not in st.session_state:
        st.session_state['attendance_messages'] = []
    if 'live_marked_names' not in st.session_state:
        st.session_state['live_marked_names'] = set()

    with tab1:
        st.markdown('<div class="section-header"><h3 style="color: var(--text-color, #2c3e50);">System Setup</h3></div>', unsafe_allow_html=True)
        
        # Faculty selection
        st.subheader("Faculty and Lecture Information")
        
        col1, col2 = st.columns(2)
        with col1:
            faculty_name = st.text_input("Faculty Name", value=st.session_state['faculty_name'], key="faculty_input")
            if faculty_name:
                st.session_state['faculty_name'] = faculty_name
        
        with col2:
            lecture_name = st.text_input("Lecture/Course Name", value=st.session_state['lecture_name'], key="lecture_input")
            if lecture_name:
                st.session_state['lecture_name'] = lecture_name
        
        if st.session_state['faculty_name'] and st.session_state['lecture_name']:
            # Create attendance file if it doesn't exist
            filename = f"Attendance_{st.session_state['faculty_name']}_{st.session_state['lecture_name']}.csv"
            if not os.path.exists(filename) or os.stat(filename).st_size == 0:
                with open(filename, 'w', newline='') as f:
                    csvwriter = csv.writer(f)
                    csvwriter.writerow(["Date", "Time", "Faculty", "Lecture", "Name", "Attendance"])
            st.markdown(f'<div class="status-success">Faculty: {st.session_state["faculty_name"]} | Lecture: {st.session_state["lecture_name"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-warning">Please enter both Faculty Name and Lecture/Course Name</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-header"><h3 style="color: var(--text-color, #2c3e50);">Student Database Management</h3></div>', unsafe_allow_html=True)
        
        # Add individual student image upload
        st.subheader("Upload Individual Student Image")
        student_name = st.text_input("Student Name (will be the filename)")
        uploaded_file = st.file_uploader("Upload student image", type=["jpg", "jpeg", "png"], key="individual_uploader")
        
        if uploaded_file and student_name:
            try:
                with st.spinner("Processing image..."):
                    # Read image with PIL first
                    img = Image.open(uploaded_file)
                    img_array = np.array(img)
                    
                    # Convert to BGR for OpenCV processing if needed
                    if len(img_array.shape) == 3 and img_array.shape[2] == 3:  # If RGB
                        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    else:
                        img_cv = img_array
                    
                    # Ensure filename has appropriate extension
                    valid_extensions = ['.jpg', '.jpeg', '.png']
                    has_valid_ext = False
                    for ext in valid_extensions:
                        if student_name.lower().endswith(ext):
                            has_valid_ext = True
                            break
                    
                    if not has_valid_ext:
                        student_name = f"{student_name}.jpg"
                    
                    # Ensure directory exists
                    if not os.path.exists('Training_images'):
                        os.makedirs('Training_images')
                    
                    # Save to disk
                    save_path = os.path.join('Training_images', student_name)
                    cv2.imwrite(save_path, img_cv)
                    
                    # Also save to session state
                    st.session_state.in_memory_training_images[student_name] = img_cv
                    
                    # Clear cache for face encoding function
                    load_and_encode_faces.clear()
                    
                    st.success(f"Image for {student_name} saved successfully!")
            except Exception as e:
                st.error(f"Error processing image: {e}")
        
        st.markdown("### OR")
        
        # Bulk upload with ZIP
        st.subheader("Upload Multiple Student Images")
        uploaded_zip = st.file_uploader("Upload a ZIP file containing student images", type=["zip"], key="zip_uploader")
        if uploaded_zip:
            try:
                with st.spinner("Processing ZIP file..."):
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        # Reset file pointer
                        uploaded_zip.seek(0)
                        
                        # Save zip to temp directory
                        zip_path = os.path.join(tmp_dir, uploaded_zip.name)
                        with open(zip_path, 'wb') as f:
                            f.write(uploaded_zip.read())
                        
                        count = 0
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            # Get list of all image files in zip
                            image_files = [
                                file for file in zip_ref.namelist() 
                                if file.lower().endswith(('.jpg', '.jpeg', '.png')) 
                                and not file.startswith('__MACOSX')
                                and not os.path.isdir(file)
                            ]
                            
                            st.info(f"Found {len(image_files)} images in ZIP file")
                            
                            # Extract each image
                            for file in image_files:
                                try:
                                    zip_ref.extract(file, tmp_dir)
                                    img_path = os.path.join(tmp_dir, file)
                                    if os.path.isfile(img_path):
                                        # Read the image
                                        img = cv2.imread(img_path)
                                        if img is not None:
                                            # Get base filename
                                            filename = os.path.basename(file)
                                            
                                            # Ensure Training_images directory exists
                                            if not os.path.exists('Training_images'):
                                                os.makedirs('Training_images')
                                                
                                            # Save to disk
                                            save_path = os.path.join('Training_images', filename)
                                            cv2.imwrite(save_path, img)
                                            
                                            # Also save to session state
                                            st.session_state.in_memory_training_images[filename] = img
                                            count += 1
                                except Exception as e:
                                    st.warning(f"Could not extract {file}: {e}")
                        
                        if count > 0:
                            st.markdown(f'<div class="status-success">Extracted {count} images from ZIP file successfully!</div>', unsafe_allow_html=True)
                            # Clear cache for the function that loads and encodes faces
                            load_and_encode_faces.clear()
                        else:
                            st.warning("No valid images found in ZIP file or could not process images")
            except Exception as e:
                st.error(f"Error processing ZIP file: {e}")
                
        st.subheader("Current Student Database")
        
        try:
            # Try to get images from both disk and memory
            images_to_display = []
            image_names = []
            
            disk_images_count = 0
            memory_images_count = 0
            
            # Set up a progress bar for the loading process
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Check if disk storage is available and has images
            status_text.text("Checking disk storage...")
            if os.path.exists('Training_images') and os.path.isdir('Training_images'):
                student_images_list = [f for f in os.listdir('Training_images') 
                                    if os.path.isfile(os.path.join('Training_images', f)) 
                                    and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if student_images_list:
                    status_text.text(f"Loading {len(student_images_list)} images from disk...")
                
                for i, img_file in enumerate(student_images_list):
                    try:
                        img_path = os.path.join('Training_images', img_file)
                        
                        # Try with PIL first
                        try:
                            img = Image.open(img_path)
                            images_to_display.append(img)
                            image_names.append(os.path.splitext(img_file)[0])
                            disk_images_count += 1
                        except Exception:
                            # If PIL fails, try with OpenCV
                            try:
                                cv_img = cv2.imread(img_path)
                                if cv_img is not None:
                                    cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                                    img = Image.fromarray(cv_img_rgb)
                                    images_to_display.append(img)
                                    image_names.append(os.path.splitext(img_file)[0])
                                    disk_images_count += 1
                            except Exception:
                                st.warning(f"Could not open {img_file} with either PIL or OpenCV")
                    except Exception as e:
                        st.warning(f"Could not process {img_file} from disk: {e}")
                    
                    # Update progress
                    if student_images_list:
                        progress_bar.progress((i + 1) / len(student_images_list) * 0.5)  # First half of progress bar
            
            # Also check memory storage regardless of whether disk loading succeeded
            if 'in_memory_training_images' in st.session_state:
                memory_files = list(st.session_state.in_memory_training_images.keys())
                
                if memory_files:
                    status_text.text(f"Loading {len(memory_files)} images from memory...")
                
                for i, (filename, img_data) in enumerate(st.session_state.in_memory_training_images.items()):
                    try:
                        if img_data is not None:
                            img_rgb = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
                            pil_img = Image.fromarray(img_rgb)
                            
                            # Only add if not already loaded from disk (check by name)
                            img_name = os.path.splitext(filename)[0]
                            if img_name not in image_names:
                                images_to_display.append(pil_img)
                                image_names.append(img_name)
                                memory_images_count += 1
                    except Exception as e:
                        st.warning(f"Could not convert {filename} from memory: {e}")
                    
                    # Update progress for second half
                    if memory_files:
                        progress_bar.progress(0.5 + (i + 1) / len(memory_files) * 0.5)
            
            # Clear progress elements
            progress_bar.empty()
            status_text.empty()
            
            # Display info about loaded images
            if disk_images_count > 0:
                st.success(f"Loaded {disk_images_count} images from disk storage")
            if memory_images_count > 0:
                st.success(f"Loaded {memory_images_count} images from memory storage")
            
            # Display the images
            if images_to_display:
                cols = 4
                rows = (len(images_to_display) + cols - 1) // cols
                
                for i in range(rows):
                    row_cols = st.columns(cols)
                    for j in range(cols):
                        idx = i * cols + j
                        if idx < len(images_to_display):
                            row_cols[j].image(images_to_display[idx], caption=image_names[idx], width=150)
            else:
                st.info("No student images available. Please upload images.")
                
        except Exception as e:
            st.error(f"Error displaying student database: {e}")
            st.info("No student images could be displayed. Please try uploading images again.")
        
    with tab2:
        st.markdown('<div class="section-header"><h2 style="color: var(--text-color, #2c3e50);">Attendance Management</h2></div>', unsafe_allow_html=True)
        
        if not st.session_state['faculty_name'] or not st.session_state['lecture_name']:
            st.markdown('<div class="status-warning">Please enter Faculty Name and Lecture Name in the Setup tab first.</div>', unsafe_allow_html=True)
            return
        
        faculty_name = st.session_state['faculty_name']
        lecture_name = st.session_state['lecture_name']
        st.markdown(f'<div class="status-info">Faculty: {faculty_name} | Lecture: {lecture_name}</div>', unsafe_allow_html=True)
        
        # Load and encode known faces using the cached function
        known_encodings, class_names = load_and_encode_faces()
        
        if not known_encodings:
            st.markdown('<div class="status-warning">No student encodings available. Please upload images in the Setup tab and ensure faces are detectable.</div>', unsafe_allow_html=True)
            return
            
        st.markdown(f'<div class="status-success">Loaded and encoded {len(known_encodings)} faces for {len(class_names)} students.</div>', unsafe_allow_html=True)
        
        # Choose attendance method
        st.subheader("Choose Attendance Method")
        
        # Show available methods based on system capabilities
        available_methods = ["Upload Image"]
        if WEBRTC_AVAILABLE:
            available_methods.append("Use Webcam")
        
        attendance_method = st.radio(
            "Select method for taking attendance:",
            available_methods,
            horizontal=True,
            key="attendance_method_radio"
        )
        
        # Clear previous attendance messages when method changes
        if st.session_state.get('last_attendance_method') != attendance_method:
            st.session_state.attendance_messages = []
            st.session_state.live_marked_names = set()
        st.session_state['last_attendance_method'] = attendance_method

        attendance_messages_placeholder = st.empty()
        
        if attendance_method == "Upload Image":
            st.subheader("Upload Image for Attendance")
            attendance_image = st.file_uploader("Upload an image containing students", type=["jpg", "jpeg", "png"], key="attendance_image_uploader")
            
            if attendance_image:
                st.session_state.attendance_messages = []
                with st.spinner("Processing image and marking attendance..."):
                    processed_img, newly_marked_in_image = process_attendance_image(attendance_image, known_encodings, class_names, faculty_name, lecture_name)
                    if processed_img is not None:
                        # Fix: Remove use_container_width parameter
                        st.image(processed_img, caption="Processed Image")
                        if newly_marked_in_image:
                            st.session_state.live_marked_names.update(newly_marked_in_image)
                        else:
                            st.markdown('<div class="status-warning">No new faces detected or recognized in the uploaded image.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="status-error">Error processing the uploaded image.</div>', unsafe_allow_html=True)
                # Display attendance messages
                if st.session_state.attendance_messages:
                    for message in st.session_state.attendance_messages:
                        attendance_messages_placeholder.markdown(message, unsafe_allow_html=True)
                else:
                    attendance_messages_placeholder.markdown('<div class="status-info">No new attendance marked.</div>', unsafe_allow_html=True)
        elif attendance_method == "Use Webcam":
            st.subheader("Use Webcam for Live Attendance")
            st.markdown('<div class="status-info">Ensure your webcam is enabled and working.</div>', unsafe_allow_html=True)
            
            # Start the webcam stream
            webrtc_streamer(
                key="live_attendance",
                mode=WebRtcMode.SENDRECV,
                video_transformer_factory=lambda: FaceRecognitionTransformer(known_encodings, class_names, faculty_name, lecture_name),
                media_stream_constraints={"video": True, "audio": False},
                async_transform=True
            )
            
            # Display live marked names
            if st.session_state.live_marked_names:
                st.markdown('<div class="status-success">Live Attendance:</div>', unsafe_allow_html=True)
                for name in st.session_state.live_marked_names:
                    attendance_messages_placeholder.markdown(f'<div class="status-success">{name}</div>', unsafe_allow_html=True)
            else:
                attendance_messages_placeholder.markdown('<div class="status-info">No attendance marked yet.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-warning">Please select a method to take attendance.</div>', unsafe_allow_html=True)
        
    # Display attendance data
    st.markdown('<div class="section-header"><h3 style="color: var(--text-color, #2c3e50);">Attendance Data</h3></div>', unsafe_allow_html=True)
    
    # Only display attendance data if faculty and lecture names are set
    if st.session_state['faculty_name'] and st.session_state['lecture_name']:
        faculty_name = st.session_state['faculty_name']
        lecture_name = st.session_state['lecture_name']
        
        # Display attendance data in a table
        filename = f"Attendance_{faculty_name}_{lecture_name}.csv"
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                if not df.empty:
                    # Fix: Remove use_container_width parameter
                    st.dataframe(df)
                    # Or use this older syntax for width adjustment:
                    # st.dataframe(df, width=800)
                else:
                    st.markdown('<div class="status-warning">No attendance data available.</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error reading attendance file: {e}")
        else:
            st.markdown('<div class="status-warning">No attendance file found.</div>', unsafe_allow_html=True)
        
        # Display download link for attendance CSV
        download_link = get_csv_download_link(faculty_name, lecture_name)
        if download_link:
            st.markdown(download_link, unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-warning">No attendance file available for download.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-warning">Please enter Faculty Name and Lecture Name in the Setup tab first.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
