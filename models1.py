from pydantic import BaseModel
from typing import Optional

class MessageResponse(BaseModel):
    message: str

class FilePathResponse(BaseModel):
    file_path: str

class MarkAttendanceResponse(MessageResponse):
    pass

class RegisterResponse(MessageResponse):
    pass

class DeleteResponse(MessageResponse):
    pass

class FaceData(BaseModel):
    Emp_id: Optional[str]=None
    Name: Optional[str]=None
    Department: Optional[str]=None
    Location: Optional[str]=None
    ImageData: str  # base64 encoded image

class FacesResponse(BaseModel):
    faces: list[FaceData]

class ImageData(BaseModel):
    imageData: str

class AttendanceRecord(BaseModel):
    Emp_id: str
    Name: str
    Department: str
    Reporting_to: str
    Location: str
    Joining_date: str
    date: str
    in_time: str
    out_time: Optional[str]=None
    Working_Hours: Optional[str]="nan"
    status: str

class AttendanceSearchResponse(BaseModel):
    records: list[AttendanceRecord]

class StatsResponse(BaseModel):
    total: int
    present: int
    half_day: int
    absent: int

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    emp_id: str
    role: str

class User(BaseModel):
    Email: Optional[str]=None
    Emp_id: Optional[str]=None
    Name: Optional[str]=None
    Role: Optional[str]=None

class UserInDB(User):
    Password: Optional[str]=None


# Model for registration data
class RegisterData(BaseModel):
    email: str
    fullName: str
    reportingTo: str
    department: str
    location: str
    joiningDate: str
    role: str
    password: Optional[str] = None
    imageData: str  # base64 image




from pydantic import BaseModel
from typing import Optional

class MessageResponse(BaseModel):
    message: str

class FilePathResponse(BaseModel):
    file_path: str

class MarkAttendanceResponse(MessageResponse):
    pass

class RegisterResponse(MessageResponse):
    pass

class DeleteResponse(MessageResponse):
    pass

class FaceData(BaseModel):
    Emp_id: Optional[str]=None
    Name: Optional[str]=None
    Department: Optional[str]=None
    Location: Optional[str]=None
    ImageData: str  # base64 encoded image

class FacesResponse(BaseModel):
    faces: list[FaceData]

class AttendanceRecord(BaseModel):
    Emp_id: str
    Name: str
    Department: str
    Reporting_to: str
    Location: str
    Joining_date: str
    date: str
    in_time: str
    out_time: Optional[str]=None
    Working_Hours: Optional[str]="nan"
    status: str

class AttendanceSearchResponse(BaseModel):
    records: list[AttendanceRecord]

class StatsResponse(BaseModel):
    total: int
    present: int
    half_day: int
    absent: int

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    emp_id: str
    role: str

class User(BaseModel):
    Email: Optional[str]=None
    Emp_id: Optional[str]=None
    Name: Optional[str]=None
    Role: Optional[str]=None

class UserInDB(User):
    Password: Optional[str]=None


# Model for registration data
class RegisterData(BaseModel):
    email: str
    fullName: str
    reportingTo: str
    department: str
    location: str
    joiningDate: str
    role: str
    password: Optional[str] = None
    imageData: str  # base64 image



