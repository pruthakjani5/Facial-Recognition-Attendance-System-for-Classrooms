import streamlit as st
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import csv
import pandas as pd
from PIL import Image
import shutil
import io
import base64
import tempfile
import zipfile
import time

# Set page configuration
st.set_page_config(
    page_title="FRASC: Face Recognition Attendance System for Classes",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
if not os.path.exists('Training_images'):
    os.makedirs('Training_images')

# Create directory for temp storage
if not os.path.exists('temp'):
    os.makedirs('temp')

# Function to find face encodings
def find_encodings(images_list):
    encode_list = []
    for img in images_list:
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(img)
            if face_encodings:
                encode = face_encodings[0]
                encode_list.append(encode)
        except Exception as e:
            st.error(f"Error encoding image: {e}")
    return encode_list

# Function to mark attendance
def mark_attendance(name, faculty_name, lecture_name, name_list):
    name_list.append(name)
    filename = f"Attendance_{faculty_name}_{lecture_name}.csv"
    header = ["Date", "Time", "Faculty", "Lecture", "Name", "Attendance"]
    found = False
    
    # Get current date and time
    now = datetime.now()
    today = now.strftime("%d-%m-%Y")
    current_time = now.strftime("%H:%M:%S")
    
    # Check if file exists
    if os.path.exists(filename):
        # Check if student already took attendance
        with open(filename, 'r') as f:
            reader = csv.DictReader(f, fieldnames=header)
            for row in reader:
                if row.get('Name') == name and row.get('Date') == today and row.get('Lecture') == lecture_name:
                    found = True
                    break
    
    if found:
        st.markdown(f'<div class="status-warning">Student with name \'{name}\' has already been marked for {lecture_name} today</div>', unsafe_allow_html=True)
    else:
        # Open file in append mode
        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if not file_exists or os.stat(filename).st_size == 0:
                writer.writeheader()
            for name in set(name_list):  # Use set to avoid duplicates
                writer.writerow({
                    "Date": today, 
                    "Time": current_time, 
                    "Faculty": faculty_name,
                    "Lecture": lecture_name,
                    "Name": name, 
                    "Attendance": 1
                })
        
        st.markdown(f'<div class="status-success">Attendance marked for {name}</div>', unsafe_allow_html=True)
    
    return set(name_list)  # Return unique names

# Function to process an attendance image
def process_attendance_image(image_file, known_encodings, class_names, faculty_name, lecture_name):
    # Read the image file
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Process the image
    name_list = []
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faces_cur_frame = face_recognition.face_locations(imgS)
    encodes_cur_frame = face_recognition.face_encodings(imgS, faces_cur_frame)
    
    # Draw rectangles and names on the image
    for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
        matches = face_recognition.compare_faces(known_encodings, encode_face)
        face_dis = face_recognition.face_distance(known_encodings, encode_face)
        
        if len(face_dis) > 0:  # Check if any faces were detected
            match_index = np.argmin(face_dis)
            if matches[match_index]:
                name = class_names[match_index]
                y1, x2, y2, x1 = face_loc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                
                # Mark attendance
                name_list = list(mark_attendance(name, faculty_name, lecture_name, name_list))
    
    # Convert processed image for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb, name_list

# Function to get a download link for the attendance file
def get_csv_download_link(faculty_name, lecture_name):
    filename = f"Attendance_{faculty_name}_{lecture_name}.csv"
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            csv_data = f.read()
        
        b64 = base64.b64encode(csv_data.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button">Download Attendance CSV</a>'
        return href
    else:
        return None
        
# Function to process webcam frame for attendance
def process_webcam_frame(frame, known_encodings, class_names, faculty_name, lecture_name):
    name_list = []
    
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Find faces in current frame
    face_locations = face_recognition.face_locations(small_frame_rgb)
    face_encodings = face_recognition.face_encodings(small_frame_rgb, face_locations)
    
    # Process each face found
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Scale back up locations
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        # See if face matches any known face
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"
        
        if True in matches:
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = class_names[best_match_index]
                
                # Mark attendance for recognized face
                if name != "Unknown":
                    name_list = list(mark_attendance(name, faculty_name, lecture_name, name_list))
        
        # Draw rectangle and name on face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
    
    return frame, name_list

# Main Streamlit app
def main():
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
    
    # Create tabs
    tab1, tab2 = st.tabs(["Setup", "Take Attendance"])
    
    # Initialize session state variables if they don't exist
    if 'faculty_name' not in st.session_state:
        st.session_state['faculty_name'] = ""
    if 'lecture_name' not in st.session_state:
        st.session_state['lecture_name'] = ""
        
    with tab1:
        st.markdown('<div class="section-header"><h2 style="color: var(--text-color, #2c3e50);">System Setup</h2></div>', unsafe_allow_html=True)
        
        # Faculty selection
        st.subheader("Faculty and Lecture Information")
        
        col1, col2 = st.columns(2)
        with col1:
            faculty_name = st.text_input("Faculty Name", value=st.session_state['faculty_name'])
            if faculty_name:
                st.session_state['faculty_name'] = faculty_name
        
        with col2:
            lecture_name = st.text_input("Lecture/Course Name", value=st.session_state['lecture_name'])
            if lecture_name:
                st.session_state['lecture_name'] = lecture_name
        
        if faculty_name and lecture_name:
            # Create attendance file if it doesn't exist
            with open(f'Attendance_{faculty_name}_{lecture_name}.csv', 'a+', newline='') as f:
                csvwriter = csv.writer(f)
                if os.stat(f'Attendance_{faculty_name}_{lecture_name}.csv').st_size == 0:
                    csvwriter.writerow(["Date", "Time", "Faculty", "Lecture", "Name", "Attendance"])
            st.markdown(f'<div class="status-success">Faculty: {faculty_name} | Lecture: {lecture_name}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-warning">Please enter both Faculty Name and Lecture/Course Name</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-header"><h3 style="color: var(--text-color, #2c3e50);">Student Database Management</h3></div>', unsafe_allow_html=True)
        
        # Option 1: Upload individual images
        st.subheader("Upload Student Images")
        uploaded_files = st.file_uploader("Upload individual student images", 
                                          type=["jpg", "jpeg", "png"], 
                                          accept_multiple_files=True)
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                # Save to training images folder
                if not os.path.exists('Training_images'):
                    os.makedirs('Training_images')
                    
                filename = uploaded_file.name
                cv2.imwrite(os.path.join('Training_images', filename), img)
            
            st.markdown(f'<div class="status-success">Uploaded {len(uploaded_files)} images successfully!</div>', unsafe_allow_html=True)
        
        # Option 2: Upload ZIP file containing images
        uploaded_zip = st.file_uploader("Or upload a ZIP file containing student images", type=["zip"])
        if uploaded_zip:
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Save the zip file to the temporary directory
                zip_path = os.path.join(tmp_dir, uploaded_zip.name)
                with open(zip_path, 'wb') as f:
                    f.write(uploaded_zip.read())
                
                # Extract the zip file
                count = 0
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    for file in zip_ref.namelist():
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')) and not file.startswith('__MACOSX'):
                            zip_ref.extract(file, tmp_dir)
                            # Copy to training images folder
                            img_path = os.path.join(tmp_dir, file)
                            if os.path.isfile(img_path):
                                if not os.path.exists('Training_images'):
                                    os.makedirs('Training_images')
                                filename = os.path.basename(file)
                                shutil.copy(img_path, os.path.join('Training_images', filename))
                                count += 1
                
                st.markdown(f'<div class="status-success">Extracted {count} images from ZIP file successfully!</div>', unsafe_allow_html=True)
        
        # Show all currently loaded student images
        if os.path.exists('Training_images'):
            student_images = [f for f in os.listdir('Training_images') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if student_images:
                st.subheader("Current Student Database")
                
                # Display the images in a grid
                cols = 4
                rows = (len(student_images) + cols - 1) // cols
                
                for i in range(rows):
                    row_cols = st.columns(cols)
                    for j in range(cols):
                        idx = i * cols + j
                        if idx < len(student_images):
                            img_path = os.path.join('Training_images', student_images[idx])
                            img = Image.open(img_path)
                            name = os.path.splitext(student_images[idx])[0]
                            row_cols[j].image(img, caption=name, width=150)
    
    with tab2:
        st.markdown('<div class="section-header"><h2 style="color: var(--text-color, #2c3e50);">Attendance Management</h2></div>', unsafe_allow_html=True)
        
        # Check if faculty and lecture are set
        if not st.session_state['faculty_name'] or not st.session_state['lecture_name']:
            st.markdown('<div class="status-warning">Please enter Faculty Name and Lecture Name in the Setup tab first.</div>', unsafe_allow_html=True)
            return
        
        faculty_name = st.session_state['faculty_name']
        lecture_name = st.session_state['lecture_name']
        st.markdown(f'<div class="status-info">Faculty: {faculty_name} | Lecture: {lecture_name}</div>', unsafe_allow_html=True)
        
        # Load and encode known faces
        if os.path.exists('Training_images'):
            student_images = [f for f in os.listdir('Training_images') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not student_images:
                st.markdown('<div class="status-warning">No student images available. Please upload images in the Setup tab.</div>', unsafe_allow_html=True)
                return
            
            # Load known faces
            images = []
            class_names = []
            
            with st.spinner("Loading student database..."):
                for img_file in student_images:
                    img_path = os.path.join('Training_images', img_file)
                    cur_img = cv2.imread(img_path)
                    if cur_img is not None:
                        images.append(cur_img)
                        class_names.append(os.path.splitext(img_file)[0])
            
            st.markdown(f'<div class="status-success">Loaded {len(class_names)} students: {", ".join(class_names)}</div>', unsafe_allow_html=True)
            
            # Encode faces
            with st.spinner("Encoding faces (this may take a moment)..."):
                known_encodings = find_encodings(images)
            
            st.markdown(f'<div class="status-success">Encoding complete. {len(known_encodings)} faces encoded.</div>', unsafe_allow_html=True)
            
            # Choose attendance method
            st.subheader("Choose Attendance Method")
            attendance_method = st.radio(
                "Select method for taking attendance:",
                ["Upload Image", "Use Webcam"],
                horizontal=True
            )
            
            marked_names = []
            
            if attendance_method == "Upload Image":
                # Upload image for attendance
                st.subheader("Upload Image for Attendance")
                attendance_image = st.file_uploader("Upload an image containing students", type=["jpg", "jpeg", "png"])
                
                if attendance_image:
                    with st.spinner("Processing image and marking attendance..."):
                        processed_img, marked_names = process_attendance_image(attendance_image, known_encodings, class_names, faculty_name, lecture_name)
                    
                    # Display processed image and marked attendance
                    st.subheader("Processed Image")
                    st.image(processed_img, channels="RGB", caption="Processed attendance image")
                    
                    if marked_names:
                        st.subheader("Attendance Marked")
                        st.markdown(f'<div class="status-success">Attendance marked for: {", ".join(marked_names)}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="status-warning">No known faces detected in the image.</div>', unsafe_allow_html=True)
            
            elif attendance_method == "Use Webcam":
                st.subheader("Webcam Attendance")
                
                # Add a start button for webcam
                start_webcam = st.button("Start Webcam Attendance")
                stop_webcam = st.button("Stop Webcam")
                
                stframe = st.empty()
                webcam_status = st.empty()
                attend_status = st.empty()
                
                if start_webcam and not stop_webcam:
                    webcam_status.markdown('<div class="status-info">Webcam is active. Position students in front of the camera.</div>', unsafe_allow_html=True)
                    
                    # Initialize webcam
                    cap = cv2.VideoCapture(0)
                    
                    # Set a time limit (30 seconds)
                    start_time = time.time()
                    time_limit = 30  # seconds
                    all_marked_names = []
                    
                    # Run webcam for attendance
                    while time.time() - start_time < time_limit:
                        success, frame = cap.read()
                        if not success:
                            st.error("Failed to access webcam. Please check your camera connection.")
                            break
                            
                        # Process frame for attendance
                        processed_frame, new_marked_names = process_webcam_frame(frame, known_encodings, class_names, faculty_name, lecture_name)
                        
                        # Update list of all marked names
                        all_marked_names.extend(new_marked_names)
                        all_marked_names = list(set(all_marked_names))  # Remove duplicates
                        
                        # Display processed frame
                        stframe.image(processed_frame, channels="BGR", caption="Live Webcam")
                        
                        # Display attendance status
                        if all_marked_names:
                            attend_status.markdown(f'<div class="status-success">Attendance marked for: {", ".join(all_marked_names)}</div>', unsafe_allow_html=True)
                        
                        # Check for stop button
                        if stop_webcam:
                            break
                            
                        # Add a small delay
                        time.sleep(0.1)
                    
                    # Release webcam
                    cap.release()
                    webcam_status.markdown('<div class="status-success">Webcam attendance session completed.</div>', unsafe_allow_html=True)
                    marked_names = all_marked_names
            
            # Download attendance CSV
            st.markdown('<div class="section-header"><h3 style="color: var(--text-color, #2c3e50);">Attendance Records</h3></div>', unsafe_allow_html=True)
            download_link = get_csv_download_link(faculty_name, lecture_name)
            if download_link:
                st.markdown(download_link, unsafe_allow_html=True)
                
                # Display attendance data
                try:
                    df = pd.read_csv(f"Attendance_{faculty_name}_{lecture_name}.csv")
                    st.markdown('<div class="attendance-data">', unsafe_allow_html=True)
                    st.dataframe(df, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error reading attendance file: {e}")
            else:
                st.markdown('<div class="status-warning">No attendance data available yet.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
