from fastapi import FastAPI,UploadFile,File,Form
from fastapi.responses import JSONResponse
from app.services.ats_engine import analyze_resume
import os
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/")
async def home():
    return {"message": "Welcome to the ATS Resume Analyzer API!"}

@app.post("/analyze")
async def analyze_resume_endpoint(resume: UploadFile = File(...),job_description: str = Form(...)):
    try:
        # Save the uploaded resume to a temporary file
        resume_path = os.path.join(UPLOAD_FOLDER, resume.filename)
        with open(resume_path, "wb") as f:
            f.write(await resume.read())
        
        # Analyze the resume against the job description
        analysis_result = analyze_resume(resume_path, job_description)
        
        # Clean up the uploaded file
        os.remove(resume_path)
        
        return JSONResponse(content=analysis_result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)