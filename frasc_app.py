# # # import streamlit as st
# # # import cv2
# # # import numpy as np
# # # import face_recognition
# # # import os
# # # from datetime import datetime
# # # import csv
# # # import pandas as pd
# # # from PIL import Image
# # # import shutil
# # # import io
# # # import base64
# # # import tempfile
# # # import zipfile
# # # import time

# # # # For webcam on Streamlit Cloud
# # # from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
# # # import av # Part of streamlit-webrtc dependencies

# # # # Set page configuration
# # # st.set_page_config(
# # #     page_title="FRASC: Face Recognition Attendance System for Classes",
# # #     layout="wide",
# # #     initial_sidebar_state="expanded"
# # # )

# # # # Custom CSS for a professional UI with dark mode compatibility
# # # st.markdown("""
# # # <style>
# # #     /* General reset for better dark mode compatibility */
# # #     .stApp {
# # #         --text-color: var(--text-color, #2c3e50);
# # #         --bg-color: var(--background-color, #ffffff);
# # #         --accent-color: #3498db;
# # #         --accent-hover: #2980b9;
# # #         --border-color: rgba(49, 51, 63, 0.2);
# # #         --card-bg: rgba(255, 255, 255, 0.1);
# # #     }
    
# # #     /* Main header with dark mode support */
# # #     .main-header {
# # #         font-size: 32px;
# # #         font-weight: bold;
# # #         color: var(--text-color, #2c3e50);
# # #         text-align: center;
# # #         background-color: rgba(52, 152, 219, 0.1);
# # #         padding: 20px;
# # #         border-radius: 8px;
# # #         margin-bottom: 30px;
# # #         border-bottom: 3px solid #3498db;
# # #         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
# # #     }
    
# # #     /* Section headers with better visibility in both modes */
# # #     .section-header {
# # #         padding: 15px;
# # #         background-color: rgba(52, 152, 219, 0.1);
# # #         border-left: 5px solid #3498db;
# # #         margin-bottom: 20px;
# # #         border-radius: 0 8px 8px 0;
# # #         box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
# # #     }
    
# # #     /* Ensure all headings in section headers are visible */
# # #     .section-header h1, .section-header h2, .section-header h3, 
# # #     .section-header h4, .section-header h5, .section-header h6 {
# # #         color: var(--text-color, #2c3e50) !important;
# # #         font-weight: 600;
# # #         margin: 0;
# # #     }
    
# # #     /* Enhanced buttons */
# # #     .stButton > button {
# # #         background-color: #3498db;
# # #         color: white !important;
# # #         font-weight: 500;
# # #         border: none;
# # #         padding: 10px 24px;
# # #         border-radius: 6px;
# # #         transition: all 0.2s ease;
# # #         box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
# # #     }
    
# # #     .stButton > button:hover {
# # #         background-color: #2980b9;
# # #         transform: translateY(-2px);
# # #         box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
# # #     }
    
# # #     .stButton > button:active {
# # #         transform: translateY(0px);
# # #     }
    
# # #     /* Improved input fields */
# # #     .stTextInput > div > div > input,
# # #     .stSelectbox > div > div > div,
# # #     .stMultiselect > div > div > div {
# # #         border-radius: 6px;
# # #         border: 1px solid var(--border-color);
# # #         padding: 8px 12px;
# # #         transition: all 0.2s;
# # #     }
    
# # #     .stTextInput > div > div > input:focus,
# # #     .stSelectbox > div > div > div:focus,
# # #     .stMultiselect > div > div > div:focus {
# # #         border-color: #3498db;
# # #         box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
# # #     }
    
# # #     /* Attendance data card */
# # #     .attendance-data {
# # #         border: 1px solid var(--border-color);
# # #         border-radius: 8px;
# # #         padding: 15px;
# # #         background-color: var(--card-bg);
# # #         box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
# # #     }
    
# # #     /* Status indicators */
# # #     .status-info {
# # #         background-color: rgba(52, 152, 219, 0.1);
# # #         border-left: 4px solid #3498db;
# # #         padding: 10px 15px;
# # #         border-radius: 0 4px 4px 0;
# # #         margin: 10px 0;
# # #     }
    
# # #     .status-success {
# # #         background-color: rgba(46, 204, 113, 0.1);
# # #         border-left: 4px solid #2ecc71;
# # #         padding: 10px 15px;
# # #         border-radius: 0 4px 4px 0;
# # #         margin: 10px 0;
# # #     }
    
# # #     .status-warning {
# # #         background-color: rgba(241, 196, 15, 0.1);
# # #         border-left: 4px solid #f1c40f;
# # #         padding: 10px 15px;
# # #         border-radius: 0 4px 4px 0;
# # #         margin: 10px 0;
# # #     }
    
# # #     /* Tabs styling */
# # #     .stTabs [data-baseweb="tab-list"] {
# # #         gap: 8px;
# # #     }
    
# # #     .stTabs [data-baseweb="tab"] {
# # #         border-radius: 4px 4px 0 0;
# # #         padding: 10px 16px;
# # #         background-color: transparent;
# # #         border-bottom: 2px solid transparent;
# # #     }
    
# # #     .stTabs [aria-selected="true"] {
# # #         background-color: rgba(52, 152, 219, 0.1) !important;
# # #         border-bottom: 2px solid #3498db !important;
# # #     }
    
# # #     /* Dataframe styling */
# # #     .dataframe {
# # #         border-collapse: collapse !important;
# # #         border: none;
# # #         border-radius: 8px;
# # #         overflow: hidden;
# # #         width: 100%;
# # #     }
    
# # #     .dataframe th {
# # #         background-color: rgba(52, 152, 219, 0.3);
# # #         padding: 12px !important;
# # #         color: var(--text-color, #2c3e50) !important;
# # #     }
    
# # #     .dataframe td {
# # #         padding: 10px !important;
# # #         border-bottom: 1px solid var(--border-color);
# # #     }
    
# # #     /* Download link styling */
# # #     .download-button {
# # #         background-color: #3498db;
# # #         color: white !important;
# # #         text-decoration: none;
# # #         padding: 10px 16px;
# # #         border-radius: 6px;
# # #         font-weight: 500;
# # #         display: inline-block;
# # #         margin: 10px 0;
# # #         transition: all 0.2s;
# # #         box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
# # #     }
    
# # #     .download-button:hover {
# # #         background-color: #2980b9;
# # #         transform: translateY(-2px);
# # #         box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
# # #         text-decoration: none;
# # #     }
# # # </style>
# # # """, unsafe_allow_html=True)

# # # # Create directory for training images if it doesn't exist
# # # if not os.path.exists('Training_images'):
# # #     os.makedirs('Training_images')

# # # # Create directory for temp storage (not strictly necessary but keeping for consistency)
# # # if not os.path.exists('temp'):
# # #     os.makedirs('temp')

# # # # Caching for performance: Load and encode known faces once
# # # @st.cache_resource
# # # def load_and_encode_faces():
# # #     images = []
# # #     class_names = []
    
# # #     if not os.path.exists('Training_images') or not os.listdir('Training_images'):
# # #         return [], []

# # #     student_images = [f for f in os.listdir('Training_images') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
# # #     for img_file in student_images:
# # #         img_path = os.path.join('Training_images', img_file)
# # #         cur_img = cv2.imread(img_path)
# # #         if cur_img is not None:
# # #             images.append(cur_img)
# # #             class_names.append(os.path.splitext(img_file)[0])
    
# # #     encode_list = []
# # #     for img in images:
# # #         try:
# # #             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # #             face_encodings = face_recognition.face_encodings(img)
# # #             if face_encodings:
# # #                 encode = face_encodings[0]
# # #                 encode_list.append(encode)
# # #         except Exception as e:
# # #             st.error(f"Error encoding image: {e}")
# # #             # Optionally, remove the problematic image's name to avoid errors later
# # #             # You might want to log this more robustly or provide more user feedback
    
# # #     return encode_list, class_names

# # # # Function to mark attendance
# # # def mark_attendance(name, faculty_name, lecture_name):
# # #     filename = f"Attendance_{faculty_name}_{lecture_name}.csv"
# # #     header = ["Date", "Time", "Faculty", "Lecture", "Name", "Attendance"]
    
# # #     now = datetime.now()
# # #     today = now.strftime("%d-%m-%Y")
# # #     current_time = now.strftime("%H:%M:%S")
    
# # #     # Check if student already took attendance for this lecture today
# # #     found = False
# # #     if os.path.exists(filename):
# # #         try:
# # #             df = pd.read_csv(filename)
# # #             if not df.empty:
# # #                 # Check if this student has an entry for today's lecture
# # #                 if ((df['Name'] == name) & (df['Date'] == today) & (df['Lecture'] == lecture_name)).any():
# # #                     found = True
# # #         except pd.errors.EmptyDataError:
# # #             pass # File exists but is empty, so no attendance yet
# # #         except Exception as e:
# # #             st.error(f"Error reading attendance file for duplicate check: {e}")

# # #     if found:
# # #         st.session_state.attendance_messages.append(f'<div class="status-warning">Student \'{name}\' already marked for {lecture_name} today.</div>')
# # #         return False
# # #     else:
# # #         # Open file in append mode
# # #         file_exists = os.path.isfile(filename) and os.stat(filename).st_size > 0 # Check if file exists AND is not empty
# # #         with open(filename, 'a', newline='') as f:
# # #             writer = csv.DictWriter(f, fieldnames=header)
# # #             if not file_exists: # Write header only if file is new or empty
# # #                 writer.writeheader()
            
# # #             writer.writerow({
# # #                 "Date": today, 
# # #                 "Time": current_time, 
# # #                 "Faculty": faculty_name,
# # #                 "Lecture": lecture_name,
# # #                 "Name": name, 
# # #                 "Attendance": 1
# # #             })
# # #         st.session_state.attendance_messages.append(f'<div class="status-success">Attendance marked for {name}.</div>')
# # #         return True # Return True if new attendance was marked

# # # # Function to process an attendance image
# # # def process_attendance_image(image_file, known_encodings, class_names, faculty_name, lecture_name):
# # #     # Read the image file
# # #     file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
# # #     img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
# # #     marked_names_in_current_image = [] # To store names newly marked in this image
    
# # #     imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
# # #     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
# # #     faces_cur_frame = face_recognition.face_locations(imgS)
# # #     encodes_cur_frame = face_recognition.face_encodings(imgS, faces_cur_frame)
    
# # #     for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
# # #         matches = face_recognition.compare_faces(known_encodings, encode_face)
# # #         face_dis = face_recognition.face_distance(known_encodings, encode_face)
        
# # #         name = "Unknown"
# # #         if len(face_dis) > 0:
# # #             match_index = np.argmin(face_dis)
# # #             if matches[match_index]:
# # #                 name = class_names[match_index]
        
# # #         y1, x2, y2, x1 = face_loc
# # #         y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        
# # #         color = (0, 255, 0) if name != "Unknown" else (0, 0, 255) # Green for known, Red for unknown
        
# # #         cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
# # #         cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
# # #         cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        
# # #         if name != "Unknown":
# # #             if mark_attendance(name, faculty_name, lecture_name):
# # #                 marked_names_in_current_image.append(name)
                
# # #     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # #     return img_rgb, marked_names_in_current_image

# # # # Video Transformer for live webcam attendance
# # # class FaceRecognitionTransformer(VideoTransformerBase):
# # #     def __init__(self, known_encodings, class_names, faculty_name, lecture_name):
# # #         self.known_encodings = known_encodings
# # #         self.class_names = class_names
# # #         self.faculty_name = faculty_name
# # #         self.lecture_name = lecture_name
# # #         self.marked_names_session = set() # To track marked names within the current webcam session

# # #     def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
# # #         img = frame.to_ndarray(format="bgr24")
        
# # #         # Process the frame
# # #         small_frame = cv2.resize(img, (0, 0), None, 0.25, 0.25)
# # #         small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
# # #         face_locations = face_recognition.face_locations(small_frame_rgb)
# # #         face_encodings = face_recognition.face_encodings(small_frame_rgb, face_locations)
        
# # #         current_frame_marked = []
        
# # #         for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
# # #             top *= 4
# # #             right *= 4
# # #             bottom *= 4
# # #             left *= 4
            
# # #             matches = face_recognition.compare_faces(self.known_encodings, face_encoding)
# # #             name = "Unknown"
            
# # #             if True in matches:
# # #                 face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
# # #                 best_match_index = np.argmin(face_distances)
# # #                 if matches[best_match_index]:
# # #                     name = self.class_names[best_match_index]
            
# # #             color = (0, 255, 0) if name != "Unknown" else (0, 0, 255) # Green for known, Red for unknown
            
# # #             cv2.rectangle(img, (left, top), (right, bottom), color, 2)
# # #             cv2.rectangle(img, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
# # #             cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            
# # #             if name != "Unknown" and name not in self.marked_names_session:
# # #                 if mark_attendance(name, self.faculty_name, self.lecture_name):
# # #                     self.marked_names_session.add(name)
# # #                     current_frame_marked.append(name) # Track for display on the frame
        
# # #         # Update the session state for displaying marked names in the UI
# # #         # This will only append unique new names from this frame
# # #         if current_frame_marked:
# # #             st.session_state.live_marked_names.update(current_frame_marked)

# # #         return av.VideoFrame.from_ndarray(img, format="bgr24")


# # # # Function to get a download link for the attendance file
# # # def get_csv_download_link(faculty_name, lecture_name):
# # #     filename = f"Attendance_{faculty_name}_{lecture_name}.csv"
# # #     if os.path.exists(filename):
# # #         with open(filename, 'r') as f:
# # #             csv_data = f.read()
        
# # #         b64 = base64.b64encode(csv_data.encode()).decode()
# # #         href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button">Download Attendance CSV</a>'
# # #         return href
# # #     else:
# # #         return None
        
# # # # Main Streamlit app
# # # def main():
# # #     # Display header with logo and title
# # #     st.markdown('''
# # #     <div class="main-header">
# # #         <div style="display: flex; align-items: center; justify-content: center; gap: 15px;">
# # #             <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#3498db" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
# # #                 <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"></path>
# # #                 <circle cx="12" cy="7" r="4"></circle>
# # #             </svg>
# # #             <span>Face Recognition Attendance System for Classes</span>
# # #         </div>
# # #     </div>
# # #     ''', unsafe_allow_html=True)
    
# # #     # Create tabs
# # #     tab1, tab2 = st.tabs(["Setup", "Take Attendance"])
    
# # #     # Initialize session state variables if they don't exist
# # #     if 'faculty_name' not in st.session_state:
# # #         st.session_state['faculty_name'] = ""
# # #     if 'lecture_name' not in st.session_state:
# # #         st.session_state['lecture_name'] = ""
# # #     if 'attendance_messages' not in st.session_state:
# # #         st.session_state['attendance_messages'] = []
# # #     if 'live_marked_names' not in st.session_state:
# # #         st.session_state['live_marked_names'] = set() # Use a set for unique names in live session

# # #     with tab1:
# # #         st.markdown('<div class="section-header"><h2 style="color: var(--text-color, #2c3e50);">System Setup</h2></div>', unsafe_allow_html=True)
        
# # #         # Faculty selection
# # #         st.subheader("Faculty and Lecture Information")
        
# # #         col1, col2 = st.columns(2)
# # #         with col1:
# # #             faculty_name = st.text_input("Faculty Name", value=st.session_state['faculty_name'], key="faculty_input")
# # #             if faculty_name:
# # #                 st.session_state['faculty_name'] = faculty_name
        
# # #         with col2:
# # #             lecture_name = st.text_input("Lecture/Course Name", value=st.session_state['lecture_name'], key="lecture_input")
# # #             if lecture_name:
# # #                 st.session_state['lecture_name'] = lecture_name
        
# # #         if st.session_state['faculty_name'] and st.session_state['lecture_name']:
# # #             # Create attendance file if it doesn't exist
# # #             # This logic should be robust enough to not write header multiple times
# # #             filename = f"Attendance_{st.session_state['faculty_name']}_{st.session_state['lecture_name']}.csv"
# # #             if not os.path.exists(filename) or os.stat(filename).st_size == 0:
# # #                 with open(filename, 'w', newline='') as f:
# # #                     csvwriter = csv.writer(f)
# # #                     csvwriter.writerow(["Date", "Time", "Faculty", "Lecture", "Name", "Attendance"])
# # #             st.markdown(f'<div class="status-success">Faculty: {st.session_state["faculty_name"]} | Lecture: {st.session_state["lecture_name"]}</div>', unsafe_allow_html=True)
# # #         else:
# # #             st.markdown('<div class="status-warning">Please enter both Faculty Name and Lecture/Course Name</div>', unsafe_allow_html=True)
        
# # #         st.markdown('<div class="section-header"><h3 style="color: var(--text-color, #2c3e50);">Student Database Management</h3></div>', unsafe_allow_html=True)
        
# # #         # Option 1: Upload individual images
# # #         st.subheader("Upload Student Images")
# # #         uploaded_files = st.file_uploader("Upload individual student images", 
# # #                                           type=["jpg", "jpeg", "png"], 
# # #                                           accept_multiple_files=True,
# # #                                           key="individual_uploader")
        
# # #         if uploaded_files:
# # #             for uploaded_file in uploaded_files:
# # #                 file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
# # #                 img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
# # #                 filename = uploaded_file.name
# # #                 cv2.imwrite(os.path.join('Training_images', filename), img)
            
# # #             st.markdown(f'<div class="status-success">Uploaded {len(uploaded_files)} images successfully!</div>', unsafe_allow_html=True)
# # #             # Clear cache for the function that loads and encodes faces
# # #             load_and_encode_faces.clear()
        
# # #         # Option 2: Upload ZIP file containing images
# # #         uploaded_zip = st.file_uploader("Or upload a ZIP file containing student images", type=["zip"], key="zip_uploader")
# # #         if uploaded_zip:
# # #             with tempfile.TemporaryDirectory() as tmp_dir:
# # #                 zip_path = os.path.join(tmp_dir, uploaded_zip.name)
# # #                 with open(zip_path, 'wb') as f:
# # #                     f.write(uploaded_zip.read())
                
# # #                 count = 0
# # #                 with zipfile.ZipFile(zip_path, 'r') as zip_ref:
# # #                     for file in zip_ref.namelist():
# # #                         if file.lower().endswith(('.jpg', '.jpeg', '.png')) and not file.startswith('__MACOSX'):
# # #                             zip_ref.extract(file, tmp_dir)
# # #                             img_path = os.path.join(tmp_dir, file)
# # #                             if os.path.isfile(img_path):
# # #                                 filename = os.path.basename(file)
# # #                                 shutil.copy(img_path, os.path.join('Training_images', filename))
# # #                                 count += 1
                
# # #                 st.markdown(f'<div class="status-success">Extracted {count} images from ZIP file successfully!</div>', unsafe_allow_html=True)
# # #                 # Clear cache for the function that loads and encodes faces
# # #                 load_and_encode_faces.clear()

# # #         # Show all currently loaded student images
# # #         st.subheader("Current Student Database")
# # #         if os.path.exists('Training_images'):
# # #             student_images_list = [f for f in os.listdir('Training_images') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
# # #             if student_images_list:
# # #                 cols = 4
# # #                 rows = (len(student_images_list) + cols - 1) // cols
                
# # #                 for i in range(rows):
# # #                     row_cols = st.columns(cols)
# # #                     for j in range(cols):
# # #                         idx = i * cols + j
# # #                         if idx < len(student_images_list):
# # #                             img_path = os.path.join('Training_images', student_images_list[idx])
# # #                             img = Image.open(img_path)
# # #                             name = os.path.splitext(student_images_list[idx])[0]
# # #                             row_cols[j].image(img, caption=name, width=150)
# # #             else:
# # #                 st.info("No student images uploaded yet.")
# # #         else:
# # #             st.info("No 'Training_images' directory found. Please upload images.")

# # #     with tab2:
# # #         st.markdown('<div class="section-header"><h2 style="color: var(--text-color, #2c3e50);">Attendance Management</h2></div>', unsafe_allow_html=True)
        
# # #         if not st.session_state['faculty_name'] or not st.session_state['lecture_name']:
# # #             st.markdown('<div class="status-warning">Please enter Faculty Name and Lecture Name in the Setup tab first.</div>', unsafe_allow_html=True)
# # #             return
        
# # #         faculty_name = st.session_state['faculty_name']
# # #         lecture_name = st.session_state['lecture_name']
# # #         st.markdown(f'<div class="status-info">Faculty: {faculty_name} | Lecture: {lecture_name}</div>', unsafe_allow_html=True)
        
# # #         # Load and encode known faces using the cached function
# # #         known_encodings, class_names = load_and_encode_faces()
        
# # #         if not known_encodings:
# # #             st.markdown('<div class="status-warning">No student encodings available. Please upload images in the Setup tab and ensure faces are detectable.</div>', unsafe_allow_html=True)
# # #             return
            
# # #         st.markdown(f'<div class="status-success">Loaded and encoded {len(known_encodings)} faces for {len(class_names)} students.</div>', unsafe_allow_html=True)
        
# # #         # Choose attendance method
# # #         st.subheader("Choose Attendance Method")
# # #         attendance_method = st.radio(
# # #             "Select method for taking attendance:",
# # #             ["Upload Image", "Use Webcam"],
# # #             horizontal=True,
# # #             key="attendance_method_radio"
# # #         )
        
# # #         # Clear previous attendance messages when method changes
# # #         if st.session_state.get('last_attendance_method') != attendance_method:
# # #             st.session_state.attendance_messages = []
# # #             st.session_state.live_marked_names = set() # Reset for live webcam
# # #         st.session_state['last_attendance_method'] = attendance_method

# # #         attendance_messages_placeholder = st.empty()
        
# # #         if attendance_method == "Upload Image":
# # #             st.subheader("Upload Image for Attendance")
# # #             attendance_image = st.file_uploader("Upload an image containing students", type=["jpg", "jpeg", "png"], key="attendance_image_uploader")
            
# # #             if attendance_image:
# # #                 st.session_state.attendance_messages = [] # Clear messages for new image upload
# # #                 with st.spinner("Processing image and marking attendance..."):
# # #                     processed_img, newly_marked_in_image = process_attendance_image(attendance_image, known_encodings, class_names, faculty_name, lecture_name)
                
# # #                 st.subheader("Processed Image")
# # #                 st.image(processed_img, channels="RGB", caption="Processed attendance image")
                
# # #                 if not newly_marked_in_image and not st.session_state.attendance_messages: # If no new faces marked and no previous messages
# # #                     st.markdown('<div class="status-warning">No new known faces detected in the image, or all detected were already marked.</div>', unsafe_allow_html=True)
                
# # #                 # Display all messages from this image processing
# # #                 for msg in st.session_state.attendance_messages:
# # #                     attendance_messages_placeholder.markdown(msg, unsafe_allow_html=True)

# # #         elif attendance_method == "Use Webcam":
# # #             st.subheader("Webcam Attendance")
            
# # #             # The webrtc_streamer component handles start/stop and stream
# # #             webrtc_ctx = webrtc_streamer(
# # #                 key="webcam-attendance",
# # #                 mode=WebRtcMode.SENDRECV,
# # #                 rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
# # #                 video_transformer_factory=lambda: FaceRecognitionTransformer(
# # #                     known_encodings=known_encodings,
# # #                     class_names=class_names,
# # #                     faculty_name=faculty_name,
# # #                     lecture_name=lecture_name
# # #                 ),
# # #                 media_stream_constraints={"video": True, "audio": False},
# # #                 async_transform=True, # Process frames asynchronously
# # #             )
            
# # #             # Display live attendance status
# # #             if webrtc_ctx.state.playing:
# # #                 st.markdown('<div class="status-info">Webcam is active. Position students in front of the camera.</div>', unsafe_allow_html=True)
# # #                 if st.session_state.live_marked_names:
# # #                     attendance_messages_placeholder.markdown(f'<div class="status-success">Attendance marked for: {", ".join(st.session_state.live_marked_names)}</div>', unsafe_allow_html=True)
# # #                 else:
# # #                     attendance_messages_placeholder.markdown('<div class="status-info">Awaiting faces for attendance...</div>', unsafe_allow_html=True)
# # #             else:
# # #                 st.session_state.live_marked_names = set() # Reset when webcam is off
# # #                 attendance_messages_placeholder.markdown('<div class="status-info">Webcam not active. Click "Start" below the video stream.</div>', unsafe_allow_html=True)
        
# # #         # Download attendance CSV
# # #         st.markdown('<div class="section-header"><h3 style="color: var(--text-color, #2c3e50);">Attendance Records</h3></div>', unsafe_allow_html=True)
# # #         download_link = get_csv_download_link(faculty_name, lecture_name)
# # #         if download_link:
# # #             st.markdown(download_link, unsafe_allow_html=True)
            
# # #             # Display attendance data
# # #             try:
# # #                 df = pd.read_csv(f"Attendance_{faculty_name}_{lecture_name}.csv")
# # #                 st.markdown('<div class="attendance-data">', unsafe_allow_html=True)
# # #                 st.dataframe(df, use_container_width=True)
# # #                 st.markdown('</div>', unsafe_allow_html=True)
# # #             except pd.errors.EmptyDataError:
# # #                 st.markdown('<div class="status-warning">No attendance data available yet for this lecture.</div>', unsafe_allow_html=True)
# # #             except Exception as e:
# # #                 st.error(f"Error reading attendance file: {e}")
# # #         else:
# # #             st.markdown('<div class="status-warning">No attendance data file created yet.</div>', unsafe_allow_html=True)


# # # if __name__ == "__main__":
# # #     main()

# # import streamlit as st
# # import cv2
# # import numpy as np
# # import face_recognition
# # import os
# # from datetime import datetime
# # import csv
# # import pandas as pd
# # from PIL import Image
# # import shutil
# # import io
# # import base64
# # import tempfile
# # import zipfile
# # import time

# # # For webcam on Streamlit Cloud
# # from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
# # import av # Part of streamlit-webrtc dependencies

# # # Set page configuration
# # st.set_page_config(
# #     page_title="FRASC: Face Recognition Attendance System for Classes",
# #     layout="wide",
# #     initial_sidebar_state="expanded"
# # )

# # # Custom CSS... (keep as is)

# # # Create directory for training images if it doesn't exist
# # if not os.path.exists('Training_images'):
# #     os.makedirs('Training_images')

# # if not os.path.exists('temp'):
# #     os.makedirs('temp')

# # # Temporarily comment out the @st.cache_resource decorator for initial debugging
# # # @st.cache_resource
# # def load_and_encode_faces():
# #     images = []
# #     class_names = []
    
# #     if not os.path.exists('Training_images') or not os.listdir('Training_images'):
# #         return [], []

# #     student_images = [f for f in os.listdir('Training_images') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
# #     for img_file in student_images:
# #         img_path = os.path.join('Training_images', img_file)
# #         cur_img = cv2.imread(img_path)
# #         if cur_img is not None:
# #             images.append(cur_img)
# #             class_names.append(os.path.splitext(img_file)[0])
    
# #     encode_list = []
# #     for img in images:
# #         try:
# #             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #             face_encodings = face_recognition.face_encodings(img)
# #             if face_encodings:
# #                 encode_list.append(encode)
# #         except Exception as e:
# #             # st.error(f"Error encoding image: {e}") # Don't use st.error during initial load if app crashes
# #             print(f"Error encoding image: {e}") # Use print for logs
    
# #     return encode_list, class_names

# # # mark_attendance function (keep as is)
# # def mark_attendance(name, faculty_name, lecture_name):
# #     filename = f"Attendance_{faculty_name}_{lecture_name}.csv"
# #     header = ["Date", "Time", "Faculty", "Lecture", "Name", "Attendance"]
    
# #     now = datetime.now()
# #     today = now.strftime("%d-%m-%Y")
# #     current_time = now.strftime("%H:%M:%S")
    
# #     found = False
# #     if os.path.exists(filename):
# #         try:
# #             df = pd.read_csv(filename)
# #             if not df.empty:
# #                 if ((df['Name'] == name) & (df['Date'] == today) & (df['Lecture'] == lecture_name)).any():
# #                     found = True
# #         except pd.errors.EmptyDataError:
# #             pass
# #         except Exception as e:
# #             print(f"Error reading attendance file for duplicate check: {e}")

# #     if found:
# #         # st.session_state.attendance_messages.append(f'<div class="status-warning">Student \'{name}\' already marked for {lecture_name} today.</div>')
# #         print(f"Student '{name}' already marked.") # Use print for logs
# #         return False
# #     else:
# #         file_exists = os.path.isfile(filename) and os.stat(filename).st_size > 0
# #         with open(filename, 'a', newline='') as f:
# #             writer = csv.DictWriter(f, fieldnames=header)
# #             if not file_exists:
# #                 writer.writeheader()
            
# #             writer.writerow({
# #                 "Date": today, 
# #                 "Time": current_time, 
# #                 "Faculty": faculty_name,
# #                 "Lecture": lecture_name,
# #                 "Name": name, 
# #                 "Attendance": 1
# #             })
# #         # st.session_state.attendance_messages.append(f'<div class="status-success">Attendance marked for {name}.</div>')
# #         print(f"Attendance marked for {name}.") # Use print for logs
# #         return True

# # # process_attendance_image function (keep as is)
# # def process_attendance_image(image_file, known_encodings, class_names, faculty_name, lecture_name):
# #     file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
# #     img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
# #     marked_names_in_current_image = []
    
# #     imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
# #     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
# #     faces_cur_frame = face_recognition.face_locations(imgS)
# #     encodes_cur_frame = face_recognition.face_encodings(imgS, faces_cur_frame)
    
# #     for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
# #         matches = face_recognition.compare_faces(known_encodings, encode_face)
# #         face_dis = face_recognition.face_distance(known_encodings, encode_face)
        
# #         name = "Unknown"
# #         if len(face_dis) > 0:
# #             match_index = np.argmin(face_dis)
# #             if matches[match_index]:
# #                 name = class_names[match_index]
        
# #         y1, x2, y2, x1 = face_loc
# #         y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        
# #         color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        
# #         cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
# #         cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
# #         cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        
# #         if name != "Unknown":
# #             if mark_attendance(name, faculty_name, lecture_name):
# #                 marked_names_in_current_image.append(name)
                
# #     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #     return img_rgb, marked_names_in_current_image

# # # Video Transformer for live webcam attendance (keep as is, but consider potential issues here too)
# # class FaceRecognitionTransformer(VideoTransformerBase):
# #     def __init__(self, known_encodings, class_names, faculty_name, lecture_name):
# #         self.known_encodings = known_encodings
# #         self.class_names = class_names
# #         self.faculty_name = faculty_name
# #         self.lecture_name = lecture_name
# #         self.marked_names_session = set()

# #     def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
# #         img = frame.to_ndarray(format="bgr24")
        
# #         small_frame = cv2.resize(img, (0, 0), None, 0.25, 0.25)
# #         small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
# #         face_locations = face_recognition.face_locations(small_frame_rgb)
# #         face_encodings = face_recognition.face_encodings(small_frame_rgb, face_locations)
        
# #         current_frame_marked = []
        
# #         for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
# #             top *= 4
# #             right *= 4
# #             bottom *= 4
# #             left *= 4
            
# #             matches = face_recognition.compare_faces(self.known_encodings, face_encoding)
# #             name = "Unknown"
            
# #             if True in matches:
# #                 face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
# #                 best_match_index = np.argmin(face_distances)
# #                 if matches[best_match_index]:
# #                     name = self.class_names[best_match_index]
            
# #             color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
# #             cv2.rectangle(img, (left, top), (right, bottom), color, 2)
# #             cv2.rectangle(img, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
# #             cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            
# #             if name != "Unknown" and name not in self.marked_names_session:
# #                 if mark_attendance(name, self.faculty_name, self.lecture_name):
# #                     self.marked_names_session.add(name)
# #                     current_frame_marked.append(name)
        
# #         if current_frame_marked:
# #             if 'live_marked_names' in st.session_state:
# #                 st.session_state.live_marked_names.update(current_frame_marked)

# #         return av.VideoFrame.from_ndarray(img, format="bgr24")

# # # get_csv_download_link (keep as is)
# # def get_csv_download_link(faculty_name, lecture_name):
# #     filename = f"Attendance_{faculty_name}_{lecture_name}.csv"
# #     if os.path.exists(filename):
# #         with open(filename, 'r') as f:
# #             csv_data = f.read()
        
# #         b64 = base64.b64encode(csv_data.encode()).decode()
# #         href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button">Download Attendance CSV</a>'
# #         return href
# #     else:
# #         return None
        
# # # Main Streamlit app - TEMPORARY DEBUGGING VERSION
# # def main():
# #     st.markdown('''
# #     <div class="main-header">
# #         <div style="display: flex; align-items: center; justify-content: center; gap: 15px;">
# #             <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#3498db" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
# #                 <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"></path>
# #                 <circle cx="12" cy="7" r="4"></circle>
# #             </svg>
# #             <span>Face Recognition Attendance System for Classes</span>
# #         </div>
# #     </div>
# #     ''', unsafe_allow_html=True)
    
# #     st.success("App is running! If you see this, the initial setup is OK.")
# #     st.write("Now, let's try loading the faces...")

# #     # Try loading faces
# #     try:
# #         known_encodings, class_names = load_and_encode_faces()
# #         st.success(f"Successfully loaded and encoded {len(known_encodings)} faces for {len(class_names)} students.")
# #         st.write("Known students:", ", ".join(class_names))
# #     except Exception as e:
# #         st.error(f"Error during face loading/encoding: {e}")
# #         st.info("Please check the 'Training_images' folder and ensure images are valid.")
        
# #     st.warning("This is a simplified version for debugging. Full features are disabled.")


# # if __name__ == "__main__":
# #     main()
# import streamlit as st
# import cv2  # COMMENT OUT
# import numpy as np # COMMENT OUT
# import face_recognition # COMMENT OUT
# import os
# from datetime import datetime # COMMENT OUT
# import csv # COMMENT OUT
# import pandas as pd # COMMENT OUT
# from PIL import Image # COMMENT OUT
# import shutil # COMMENT OUT
# import io # COMMENT OUT
# import base64 # COMMENT OUT
# import tempfile # COMMENT OUT
# import zipfile # COMMENT OUT
# import time # COMMENT OUT

# # # For webcam on Streamlit Cloud
# # from streamlit_webrtc import webrtc_streamer, VideoTransformerBase # COMMENT OUT
# # import av # COMMENT OUT


# # st.set_page_config(...) # Keep commented out
# # st.markdown(...) # Keep commented out

# # Create directory for training images if it doesn't exist
# if not os.path.exists('Training_images'):
#     os.makedirs('Training_images')

# if not os.path.exists('temp'):
#     os.makedirs('temp')

# # ALL FUNCTION DEFINITIONS BELOW THIS LINE REMAIN COMMENTED OUT
# # def load_and_encode_faces():
# #     ...
# # def mark_attendance(...):
# #     ...
# # def process_attendance_image(...):
# #     ...
# # class FaceRecognitionTransformer(VideoTransformerBase):
# #     ...
# # def get_csv_download_link(...):
# #     ...


# # Main Streamlit app - TEMPORARY DEBUGGING VERSION
# def main():
#     st.write("Debug App is running! (Minimal Imports)")
#     print("Debug: main function started with minimal imports.")

# if __name__ == "__main__":
#     main()
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
import io
import base64
import tempfile
import zipfile

# Import face_recognition with better error handling
# Don't use signal module since it fails in Streamlit Cloud environment
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
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    st.warning("WebRTC not available - webcam functionality disabled")

# Rest of your code remains the same...

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

# Caching for performance: Load and encode known faces once
@st.cache_data
def load_and_encode_faces():
    if not FACE_RECOGNITION_AVAILABLE:
        return [], []
    
    images = []
    class_names = []
    
    # More robust directory checking
    try:
        # First verify the directory exists and is actually a directory
        if not os.path.exists('Training_images'):
            try:
                os.makedirs('Training_images')
                return [], []  # Return empty lists since we just created the directory
            except Exception as e:
                st.error(f"Could not create Training_images directory: {e}")
                return [], []
        
        if not os.path.isdir('Training_images'):
            st.error("Training_images exists but is not a directory")
            return [], []
            
        # Check if the directory is empty
        files = os.listdir('Training_images')
        if not files:
            return [], []
            
        # Filter for image files
        student_images = [f for f in files if os.path.isfile(os.path.join('Training_images', f)) 
                          and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not student_images:
            return [], []
            
        # Process each image
        for img_file in student_images:
            try:
                img_path = os.path.join('Training_images', img_file)
                if not os.path.exists(img_path):
                    continue
                    
                cur_img = cv2.imread(img_path)
                if cur_img is not None:
                    images.append(cur_img)
                    class_names.append(os.path.splitext(img_file)[0])
            except Exception as e:
                st.warning(f"Could not load image {img_file}: {e}")
        
        # No images were loaded successfully
        if not images:
            return [], []
        
        # Encode faces
        encode_list = []
        for i, img in enumerate(images):
            try:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(img_rgb)
                if face_encodings:
                    encode = face_encodings[0]
                    encode_list.append(encode)
                else:
                    st.warning(f"No face detected in image for {class_names[i]}")
                    # Remove this name since we couldn't encode the face
                    class_names[i] = None
            except Exception as e:
                st.error(f"Error encoding image for {class_names[i]}: {e}")
                class_names[i] = None
        
        # Remove any None values from class_names where encoding failed
        encode_list_final = []
        class_names_final = []
        for i, name in enumerate(class_names):
            if name is not None:
                encode_list_final.append(encode_list[i])
                class_names_final.append(name)
        
        return encode_list_final, class_names_final
        
    except Exception as e:
        st.error(f"Error in load_and_encode_faces: {e}")
        return [], []

# Function to mark attendance
def mark_attendance(name, faculty_name, lecture_name):
    filename = f"Attendance_{faculty_name}_{lecture_name}.csv"
    header = ["Date", "Time", "Faculty", "Lecture", "Name", "Attendance"]
    
    now = datetime.now()
    today = now.strftime("%d-%m-%Y")
    current_time = now.strftime("%H:%M:%S")
    
    # Check if student already took attendance for this lecture today
    found = False
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            if not df.empty:
                # Check if this student has an entry for today's lecture
                if ((df['Name'] == name) & (df['Date'] == today) & (df['Lecture'] == lecture_name)).any():
                    found = True
        except pd.errors.EmptyDataError:
            pass
        except Exception as e:
            st.error(f"Error reading attendance file for duplicate check: {e}")

    if found:
        if 'attendance_messages' not in st.session_state:
            st.session_state.attendance_messages = []
        st.session_state.attendance_messages.append(f'<div class="status-warning">Student \'{name}\' already marked for {lecture_name} today.</div>')
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
                "Name": name, 
                "Attendance": 1
            })
        
        if 'attendance_messages' not in st.session_state:
            st.session_state.attendance_messages = []
        st.session_state.attendance_messages.append(f'<div class="status-success">Attendance marked for {name}.</div>')
        return True

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
                    
                    matches = face_recognition.compare_faces(self.known_encodings, face_encoding)
                    name = "Unknown"
                    
                    if True in matches:
                        face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.class_names[best_match_index]
                    
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    
                    cv2.rectangle(img, (left, top), (right, bottom), color, 2)
                    cv2.rectangle(img, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                    cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                    
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
            with open(filename, 'r') as f:
                csv_data = f.read()
            
            b64 = base64.b64encode(csv_data.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button">Download Attendance CSV</a>'
            return href
        except Exception as e:
            st.error(f"Error creating download link: {e}")
            return None
    else:
        return None

# Main Streamlit app
def main():
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
        st.markdown('<div class="section-header"><h2 style="color: var(--text-color, #2c3e50);">System Setup</h2></div>', unsafe_allow_html=True)
        
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
        
        # Option 1: Upload individual images
        st.subheader("Upload Student Images")
        uploaded_files = st.file_uploader("Upload individual student images", 
                                          type=["jpg", "jpeg", "png"], 
                                          accept_multiple_files=True,
                                          key="individual_uploader")
        
        # if uploaded_files:
        #     success_count = 0
        #     for uploaded_file in uploaded_files:
        #         try:
        #             file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        #             img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
        #             if img is not None:
        #                 filename = uploaded_file.name
        #                 cv2.imwrite(os.path.join('Training_images', filename), img)
        #                 success_count += 1
        #             else:
        #                 st.warning(f"Could not decode image: {uploaded_file.name}")
        #         except Exception as e:
        #             st.error(f"Error processing {uploaded_file.name}: {e}")
            
        #     if success_count > 0:
        #         st.markdown(f'<div class="status-success">Uploaded {success_count} images successfully!</div>', unsafe_allow_html=True)
        #         # Clear cache for the function that loads and encodes faces
        #         load_and_encode_faces.clear()
        # Add this near the top of your file
    if 'in_memory_training_images' not in st.session_state:
        st.session_state.in_memory_training_images = {}  # {filename: image_data}
    
    # Then modify your upload handler to also store in memory:
        if uploaded_files:
            success_count = 0
            for uploaded_file in uploaded_files:
                try:
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    if img is not None:
                        filename = uploaded_file.name
                        
                        # Try to save to disk
                        try:
                            cv2.imwrite(os.path.join('Training_images', filename), img)
                        except Exception as e:
                            st.warning(f"Could not save to disk, using memory storage: {e}")
                        
                        # Also save to memory as backup
                        st.session_state.in_memory_training_images[filename] = img
                        success_count += 1
                    else:
                        st.warning(f"Could not decode image: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                
            # Option 2: Upload ZIP file containing images
        uploaded_zip = st.file_uploader("Or upload a ZIP file containing student images", type=["zip"], key="zip_uploader")
        if uploaded_zip:
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    zip_path = os.path.join(tmp_dir, uploaded_zip.name)
                    with open(zip_path, 'wb') as f:
                        f.write(uploaded_zip.read())
                    
                    count = 0
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        for file in zip_ref.namelist():
                            if file.lower().endswith(('.jpg', '.jpeg', '.png')) and not file.startswith('__MACOSX'):
                                try:
                                    zip_ref.extract(file, tmp_dir)
                                    img_path = os.path.join(tmp_dir, file)
                                    if os.path.isfile(img_path):
                                        filename = os.path.basename(file)
                                        shutil.copy(img_path, os.path.join('Training_images', filename))
                                        count += 1
                                except Exception as e:
                                    st.warning(f"Could not extract {file}: {e}")
                    
                    if count > 0:
                        st.markdown(f'<div class="status-success">Extracted {count} images from ZIP file successfully!</div>', unsafe_allow_html=True)
                        # Clear cache for the function that loads and encodes faces
                        load_and_encode_faces.clear()
                    else:
                        st.warning("No valid images found in ZIP file")
            except Exception as e:
                st.error(f"Error processing ZIP file: {e}")

        # # Show all currently loaded student images
        # st.subheader("Current Student Database")
        # if os.path.exists('Training_images'):
        #     student_images_list = [f for f in os.listdir('Training_images') if os.path.isfile(os.path.join('Training_images', f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))] if (os.path.exists('Training_images') and os.path.isdir('Training_images')) else []
            
        #     if student_images_list:
        #         cols = 4
        #         rows = (len(student_images_list) + cols - 1) // cols
                
        #         for i in range(rows):
        #             row_cols = st.columns(cols)
        #             for j in range(cols):
        #                 idx = i * cols + j
        #                 if idx < len(student_images_list):
        #                     try:
        #                         img_path = os.path.join('Training_images', student_images_list[idx])
        #                         img = Image.open(img_path)
        #                         name = os.path.splitext(student_images_list[idx])[0]
        #                         row_cols[j].image(img, caption=name, width=150)
        #                     except Exception as e:
        #                         row_cols[j].error(f"Could not display {student_images_list[idx]}")
        #     else:
        #         st.info("No student images uploaded yet.")
        # else:
        #     st.info("No 'Training_images' directory found. Please upload images.")
        # Show all currently loaded student images
        st.subheader("Current Student Database")
        
        try:
            # Try to get images from disk first
            images_to_display = []
            image_names = []
            
            # Check if disk storage is available and has images
            if os.path.exists('Training_images') and os.path.isdir('Training_images'):
                student_images_list = [f for f in os.listdir('Training_images') 
                                      if os.path.isfile(os.path.join('Training_images', f)) 
                                      and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                for img_file in student_images_list:
                    try:
                        img_path = os.path.join('Training_images', img_file)
                        img = Image.open(img_path)
                        images_to_display.append(img)
                        image_names.append(os.path.splitext(img_file)[0])
                    except Exception as e:
                        st.warning(f"Could not open {img_file} from disk: {e}")
            
            # If no images found on disk, check memory storage
            if not images_to_display and 'in_memory_training_images' in st.session_state:
                for filename, img_data in st.session_state.in_memory_training_images.items():
                    try:
                        img_rgb = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(img_rgb)
                        images_to_display.append(pil_img)
                        image_names.append(os.path.splitext(filename)[0])
                    except Exception as e:
                        st.warning(f"Could not convert {filename} from memory: {e}")
            
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
                        st.image(processed_img, caption="Processed Image", use_container_width=True)
                        if newly_marked_in_image:
                            st.session_state.live_marked_names.update(newly_marked_in_image)
                        else:
                            st.markdown('<div class="status-warning">No faces detected in the uploaded image.</div>', unsafe_allow_html=True)
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
        # Display attendance messages
        if st.session_state.attendance_messages:
            for message in st.session_state.attendance_messages:
                attendance_messages_placeholder.markdown(message, unsafe_allow_html=True)
        else:
            attendance_messages_placeholder.markdown('<div class="status-info">No new attendance marked.</div>', unsafe_allow_html=True)
    # Display attendance data
    st.markdown('<div class="section-header"><h3 style="color: var(--text-color, #2c3e50);">Attendance Data</h3></div>', unsafe_allow_html=True)
    if 'attendance_messages' in st.session_state and st.session_state.attendance_messages:
        for message in st.session_state.attendance_messages:
            attendance_messages_placeholder.markdown(message, unsafe_allow_html=True)
    else:
        attendance_messages_placeholder.markdown('<div class="status-info">No new attendance marked.</div>', unsafe_allow_html=True)
    # Display attendance data in a table
    filename = f"Attendance_{faculty_name}_{lecture_name}.csv"
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            if not df.empty:
                st.dataframe(df, use_container_width=True)
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

if __name__ == "__main__":
    main()
