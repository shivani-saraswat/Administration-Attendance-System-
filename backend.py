#main.py
import os
from PIL import Image
import io
import pytz
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Query, Body
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import EmailStr
from models1 import *
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import atexit
from base64 import b64encode
import base64
import cv2
from email.message import EmailMessage
import smtplib
import numpy as np
from logic1 import *
from logic1 import auto_mark_all_present_on_off_days
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi import Depends, HTTPException, status, Request
import pandas as pd
import sqlite3
from fastapi import Depends, HTTPException
from pydantic import BaseModel
import dlib

logger = logging.getLogger(__name__)

# Initialize dlib models for HOG face detection and face encoding
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# At startup or schedule this with a cron job / background task

scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
scheduler.add_job(
    func = auto_mark_all_present_on_off_days,
    trigger = CronTrigger(hour=0, minute=5),
    name = "Auto-mark Attendance on Off Days"
)
scheduler.start()

atexit.register(lambda: scheduler.shutdown())

# auto_mark_all_present_on_off_days()
load_dotenv()
app = FastAPI()

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY")  # Change this in production
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT"))
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
RESET_TOKEN_EXPIRE_MINUTES = 15


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Enable CORS for frontend JS to work
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
# app.mount("/", StaticFiles(directory=".", html=True), name="static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize DB
init_db()
if not get_user_from_db(emp_id="WM-0", query=["*"]):
    hashed = pwd_context.hash("admin123")
    insert_into_userDB("","","Noida", "alt.tu-2jhdct9@yopmail.com","WM-0", "Admin User", "", hashed, "admin")

def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

def password_hash(password: str):
    return pwd_context.hash(password)


def authenticate_user(emp_id: str, password: str):
    # print(emp_id)
    user = get_user_from_db(emp_id=emp_id, query=["*"])
    # print(user)
    if not user or not verify_password(password, user.Password):
        return False
    return user


def create_reset_token(email: str) -> str:
    ist = pytz.timezone("Asia/Kolkata")
    expire = datetime.now(ist) + timedelta(minutes=RESET_TOKEN_EXPIRE_MINUTES)
    to_encode = {"sub": email, "exp": expire, "type": "psrt"}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

@app.get("/verify_token/")
def verify_jwt_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=400, detail="Invalid token")
        return {"valid": True}
    except JWTError:
        raise HTTPException(status_code=403, detail="Invalid or expired token")

def create_access_token(data: dict, expires_delta: timedelta=None):
    to_encode = data.copy()
    ist = pytz.timezone("Asia/Kolkata")
    if expires_delta:
        expire = datetime.now(ist) + expires_delta
        print(expire)
    else:
        expire = datetime.now(ist) + timedelta(minutes=15)
        print(expire)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@app.get("/check-blacklist")
def token_check(token: str=Query(...)):
    try:
        check_blacklist_status(token=token)
        return {"message": "Token Not Revoked Yet"}
    except JWTError as e:
        raise HTTPException(status_code=400, detail="Token Revoked")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exceptions = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail = "Could Not Validate Credentials",
        headers = {"WWW-Authenticate": "Bearer"},
    )
    try:
        check_blacklist_status(token)
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        emp_id: str = payload.get("sub")
        role: str = payload.get("role")
        if emp_id is None:
            raise credentials_exceptions
        token_data = TokenData(emp_id=emp_id, role=role)
    except JWTError:
        raise credentials_exceptions
    user = get_user_from_db(emp_id = token_data.emp_id, query=["*"])
    if user is None:
        raise credentials_exceptions
    return user

async def require_admin(current_user: User = Depends(get_current_user)):
    if current_user.Role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail = "Admin Privileges Required"
        )
    return current_user

@app.post("/token", response_model = Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers = {"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data = {"sub": user.Emp_id, "role": user.Role}, expires_delta = access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}



@app.get("/login")
def login_page():
    with open("login.html", "r", encoding="utf-8") as file: 
        return HTMLResponse(content=file.read())

@app.get("/welcome")
def welcome():
    with open("welcome.html", "r", encoding="utf-8") as file:
        return HTMLResponse(content=file.read())


# In main.py
@app.post("/logout")
async def logout(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        expiry = datetime.fromtimestamp(payload["exp"])

        conn = sqlite3.connect('faces.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO token_blacklist (token, expiry) VALUES (?, ?)",
            (token, expiry)
        )
        conn.commit()
        conn.close()

        return {"message": "Logged out successfully"}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/admindashboard/home")
def test(current_user: User = Depends(require_admin)):
    with open("main.html","r",encoding="utf-8") as file:
        return HTMLResponse(content=file.read())


@app.get("/admindashboard/register")
def html_register(current_user: User = Depends(require_admin)):
    with open("register_face.html", "r", encoding="utf-8") as file:
        return HTMLResponse(content=file.read())

@app.post("/admindashboard/register", response_model=RegisterResponse)
async def register_employee(data: RegisterData, current_user: Depends = Depends(require_admin)):
    try:
        # if get_user_from_db(emp_id=data.empId, query=["*"]):
        #     raise HTTPException(status_code=400, detail="User already exists")
        hashed_password = password_hash(data.password)
        emp_id = get_incremented_empId()

        # Defensive: Check imageData
        if not data.imageData or "," not in data.imageData:
            return JSONResponse(content={"message": "No image data provided."}, status_code=400)
        try:
            header, encoded = data.imageData.split(",", 1)
            img_bytes = base64.b64decode(encoded)
        except Exception as e:
            return JSONResponse(content={"message": "Invalid image data."}, status_code=400)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Debug: Save the decoded image to disk
        try:
            if frame is not None:
                cv2.imwrite("test_register_image.jpg", frame)
        except Exception as e:
            print(f"Failed to save debug image: {e}")
        if frame is None or len(frame.shape) != 3 or frame.dtype != np.uint8:
            return JSONResponse(content={"message": "Invalid image format. Please try again."}, status_code=400)
        # Convert to RGB for dlib
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Use dlib's HOG face detector
        dets = detector(rgb_frame, 1)
        if not dets:
            return JSONResponse(content={"message": "No face detected. Try again."}, status_code=400)
        
        # Get face landmarks and compute face descriptor
        shape = sp(rgb_frame, dets[0])
        face_descriptor = facerec.compute_face_descriptor(rgb_frame, shape)
        face_en = np.array(face_descriptor, dtype=np.float64).tobytes()
        
        # Insert user first, then save face to DB
        insert_into_userDB(data.reportingTo, data.joiningDate, data.location, data.email, emp_id, data.fullName, data.department, hashed_password, data.role)
        save_face_to_db(
            emp_id,
            frame,  
            face_en
        )

        return {"message": f"{data.fullName} registered successfully With Employee ID: {emp_id}!"}
    except Exception as e:
        # Log the error for debugging
        import traceback
        print("Registration failed:", traceback.format_exc())
        return JSONResponse(content={"message": f"Error: {str(e)}"}, status_code=500)

@app.get("/admindashboard/delete")
def test(current_user: User = Depends(require_admin)):
    with open("delete_faces.html","r",encoding="utf-8") as file:
        return HTMLResponse(content=file.read())

@app.post("/admindashboard/delete/{emp_id}", response_model=DeleteResponse)
async def delete_employee(
    emp_id: str,
    current_user: User = Depends(require_admin)
):
    result = delete_face(emp_id)
    return {"message": result}

@app.get("/shivani")
def a(date: str):
    return compare_is_off_day(date_str=date)


@app.get("/mark_attendance")
def mark_attendance_page():
    with open("mark_attendance.html", "r", encoding="utf-8") as file:
        return HTMLResponse(content=file.read())

def send_email(recipient_email: str, link):
    msg = EmailMessage()
    msg["Subject"] = "Password Reset Request"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = recipient_email
    reset_token = create_reset_token(recipient_email)
    msg.set_content(f"This Email is for resetting your password. Please click the link below to reset your password: {link}")

    with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)

@app.get("/request_password_reset/")
async def trigger_email(email: EmailStr):
    try:
        user = get_user_from_db(email=email, query=["Email"])
        if not user:
            return {"message": "Email Not Found"}
        
        token = create_reset_token(email)
        reset_link = f"http://localhost:8001/reset_Password?token={token}"
        send_email(recipient_email=email, link=reset_link)
        return {"message": f"Email sent to {email}"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/reset_Password/")
def rstPasswordPage():
    with open("reset_password.html", "r", encoding="utf-8") as file:
        return HTMLResponse(content=file.read())

@app.post("/reset_Password/")
async def resetpassword(token: str = Body(...), password: str = Body(...)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type")!="psrt":
            raise HTTPException(status_code=400, detail="Invalid Token type")
        email = payload.get("sub")
        hashed = pwd_context.hash(password)
    except JWTError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # print(email, password, hashed)
    res = update_user_password(email, hashed)
    if res:
        return {"msg": "Password updated successfully"}
    else:
        return {"msg": "A problem occured while trying to reset Password"} 


@app.post("/mark_attendance", response_model=MarkAttendanceResponse)
async def mark_attendance_api(file: UploadFile = File(...)):
    try:
        content = await file.read()
        
        # Check if image is empty
        if len(content) == 0:
            return JSONResponse(content={"message": "Empty image received"}, status_code=400)
        
        try:
            # Try OpenCV first
            nparr = np.frombuffer(content, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            logger.debug(f"Received image: {len(content)} bytes")
        except Exception as cv_err:
            logger.warning(f"OpenCV decoding failed: {str(cv_err)}")
            # Fallback to PIL
            try:
                image = Image.open(io.BytesIO(content))
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            except Exception as pil_err:
                logger.error(f"PIL decoding failed: {str(pil_err)}")
                return JSONResponse(
                    content={"message": "Invalid image format"},
                    status_code=400
                )        
        known_encodings, metadata = get_face_encodings_from_db()
        
        # Convert to RGB for dlib
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Use dlib's HOG face detector
        dets = detector(rgb_frame, 1)
        if not dets:
            return JSONResponse(content={"message": "No face detected"}, status_code=400)
        
        # Get face landmarks and compute face descriptor
        shape = sp(rgb_frame, dets[0])
        face_descriptor = facerec.compute_face_descriptor(rgb_frame, shape)
        face_encoding = np.array(face_descriptor, dtype=np.float64)
        
        # Compare with known faces
        for i, known_encoding in enumerate(known_encodings):
            # Calculate Euclidean distance for face comparison
            distance = np.linalg.norm(face_encoding - known_encoding)
            if distance < 0.4:  # Threshold for face matching (lower = stricter)
                sr_no = metadata[i][0]
                mark_attendance(sr_no)
                return {"message": "Attendance marked"}
        
        return JSONResponse(content={"message": "Face not recognized", "status": "danger"}, status_code=404)
    except Exception as e:
        logger.exception("Attendance marking failed")
        return JSONResponse(
            content={"message": f"Server error: {str(e)}"},
            status_code=500
        )

@app.get("/admindashboard/search_attendence")
def test(current_user: User = Depends(require_admin)):
    with open("search_attendence_by_name.html","r",encoding="utf-8") as file:
        return HTMLResponse(content=file.read())
    
@app.get("/admindasboard/get_search_attendance/{name}", response_model=AttendanceSearchResponse)
async def search_attendance_api(name: str = "", current_user: Depends = Depends(require_admin)):
    records = search_attendance_records(name)
    return {"records": records}

@app.get("/export/excel")
async def export_excel(current_user: Depends = Depends(require_admin)):
    try:
        file_content = export_data_to_excel(purpose="download")
        return StreamingResponse(
            iter([file_content]),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename=attendance_report.xlsx"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_stats(current_user: Depends = Depends(require_admin)):
    return get_attendance_stats()


def convert_image_to_base64(image_blob):
    return f"data:image/jpeg;base64,{b64encode(image_blob).decode('utf-8')}"

@app.get("/admindashboard/faces")
def test(current_user: User = Depends(require_admin)):
    with open("show_faces.html","r",encoding="utf-8") as file:
        return HTMLResponse(content=file.read())
    
@app.get("/admindashboard/get_faces", response_model=FacesResponse)
async def get_faces(current_user: Depends = Depends(require_admin)):
    conn = sqlite3.connect('faces.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT u.Emp_id, u.Name, u.Department, u.Location,f.image
        FROM users u
        JOIN faces f ON u.SrNo = f.SrNo
        WHERE u.Name IS NOT NULL
    """)
    rows = cursor.fetchall()
    conn.close()

    face_list = []
    for emp_id, name, dept, location, img_blob in rows:
        face_list.append({
            "Emp_id": emp_id,
            "Name": name,
            "Department": dept,
            "Location": location,
            "ImageData": convert_image_to_base64(img_blob)
        })
    return {"faces": face_list}

@app.get("/loader.html")
def loader_page():
    with open("loader.html", "r", encoding="utf-8") as file:
        return HTMLResponse(content=file.read())

# @app.get("/view_spreadsheet", response_model=AttendanceSearchResponse)
# async def view_spreadsheet(current_user: Depends = Depends(require_admin)):
#     try:
#         return {"records": export_data_to_excel(purpose="view")}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.get("/view_spreadsheet", response_model=AttendanceSearchResponse)
async def view_spreadsheet(
    date: Optional[str] = Query(None),
    current_user: Depends = Depends(require_admin)
):
    try:
        return {"records": export_data_to_excel(purpose="view", filter_date=date)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admindashboard/view_spreadsheet")
def serve_view_spreadsheet():
    with open("view_spreadsheet.html", "r", encoding="utf-8") as file:
        return HTMLResponse(content=file.read())

@app.get("/api/admin_dashboard_data")
def admin_dashboard_data(date: str = Query(None), department: str = Query(None), current_user: User = Depends(require_admin)):
    conn = sqlite3.connect('faces.db')
    cursor = conn.cursor()

    # Date filter
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    # Department filter for total employees
    emp_query = "SELECT COUNT(*) FROM users where Name is not null and Name !='Admin User'"
    emp_params = []
    if department and department != 'All':
        emp_query += " and Department = ?"
        emp_params.append(department)
    cursor.execute(emp_query, emp_params)
    total_employees = cursor.fetchone()[0]

    # Get all present employees for the date (in_time exists and working hours >= 9)
    att_query = """
        SELECT a.SrNo, a.in_time, a.out_time, u.Department
        FROM attendance a
        JOIN users u ON a.SrNo = u.SrNo
        WHERE a.date = ?
        and u.Name is not null
        and u.Name != 'Admin User'
    """
    att_params = [date]
    if department and department != 'All':
        att_query += " AND u.Department = ?"
        att_params.append(department)
    cursor.execute(att_query, att_params)
    attendance_rows = cursor.fetchall()

    present = 0
    half_day = 0
    for row in attendance_rows:
        in_time, out_time = row[1], row[2]
        if in_time and out_time:
            try:
                in_dt = datetime.strptime(in_time, "%H:%M:%S")
                out_dt = datetime.strptime(out_time, "%H:%M:%S")
                hours = (out_dt - in_dt).total_seconds() / 3600
                if hours >= 9:
                    present += 1
                elif hours >= 4:
                    half_day += 1
            except:
                continue
    total_present = present
    total_absent = total_employees - (present + half_day)
    attendance_rate = (total_present / total_employees) * 100 if total_employees else 0
    absent_rate = (total_absent / total_employees) * 100 if total_employees else 0

    # Department-wise attendance for the date
    dept_query = """
        SELECT u.Department, COUNT(a.SrNo)
        FROM users u
        LEFT JOIN attendance a ON u.SrNo = a.SrNo AND a.date = ?
        WHERE u.name is not null
    """
    dept_params = [date]
    if department and department != 'All':
        dept_query += " and u.Department = ?"
        dept_params.append(department)
    dept_query += " GROUP BY u.Department"
    cursor.execute(dept_query, dept_params)
    dept_counts = cursor.fetchall()
    attendance_by_department = [
        {"department": dept, "count": count} for dept, count in dept_counts
    ]

    conn.close()
    return {
        "total_employees": total_employees,
        "total_present": total_present,
        "total_absent": total_absent,
        "attendance_rate": round(attendance_rate, 2),
        "absent_rate": round(absent_rate, 2),
        "attendance_by_department": attendance_by_department
    }

@app.get("/admindashboard/export_excel_landing")
def export_excel_landing(current_user: User = Depends(require_admin)):
    with open("export_excel_landing.html", "r", encoding="utf-8") as file:
        return HTMLResponse(content=file.read())

@app.get("/attendance-list")
async def attendance_list(date: str = Query(None), department: str = Query(None), current_user: User = Depends(require_admin)):

    conn = sqlite3.connect('faces.db')
    cursor = conn.cursor()

    # Use today's date if not provided
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    # LEFT JOIN to get all employees and their attendance for the date
    query = """
        SELECT u.Emp_id, u.Name, a.in_time
        FROM users u
        LEFT JOIN attendance a ON u.SrNo = a.SrNo AND a.date = ?
        WHERE u.Name is not null
        and u.Name !='Admin User'
    """
    params = [date]
    if department and department != 'All':
        query += " and u.Department = ?"
        params.append(department)
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    onTime = []
    late = []
    absent = []

    for emp_id, name, in_time in rows:
        if not in_time:
            absent.append({"name": name, "time": None})
        elif "10:00:00" <= in_time <= "10:30:00":
            onTime.append({"name": name, "time": in_time})
        elif in_time > "10:30:00":
            late.append({"name": name, "time": in_time})
        else:
            # Before 10:00:00, treat as onTime
            onTime.append({"name": name, "time": in_time})

    return {"onTime": onTime, "late": late, "absent": absent}

@app.get("/")
def root():
    return RedirectResponse(url="/welcome")

@app.post("/admindashboard/check_duplicate_face")
async def check_duplicate_face(data: ImageData, token: str = Depends(oauth2_scheme)):
    try:
        image_data = data.imageData.split(",")[1]
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert to RGB for dlib
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Use dlib's HOG face detector
        dets = detector(rgb_img, 1)
        if not dets:
            raise HTTPException(status_code=400, detail="No face detected.")

        # Get face landmarks and compute face descriptor
        shape = sp(rgb_img, dets[0])
        face_descriptor = facerec.compute_face_descriptor(rgb_img, shape)
        new_encoding = np.array(face_descriptor, dtype=np.float64)

        # Load all stored face encodings from DB
        conn = sqlite3.connect('faces.db')
        cursor = conn.cursor()
        cursor.execute("SELECT image_en FROM faces")
        rows = cursor.fetchall()
        conn.close()

        for row in rows:
            stored_encoding = np.frombuffer(row[0], dtype=np.float64)
            # Calculate Euclidean distance for face comparison
            distance = np.linalg.norm(new_encoding - stored_encoding)
            if distance < 0.4:  # Threshold for face matching (lower = stricter)
                return {"exists": True}

        return {"exists": False}

    except Exception as e:
        print("❌ Error in check_duplicate_face:", str(e))  
        raise HTTPException(status_code=500, detail="Server error during face check.")
