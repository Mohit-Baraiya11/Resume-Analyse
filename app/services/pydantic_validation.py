from pydantic import BaseModel, Field
from typing import List

class ResumeChunk(BaseModel):
    match_score:float = Field(..., description="The similarity score between the query and the resume chunk")
    missing_skills:List[str] = Field(..., description="List of skills mentioned in the job description but missing in the resume chunk")
    strengths: List[str]
    improvement_suggestions: List[str]
    advice: str

class ExtractResumeInfo(BaseModel):
    skills: List[str] = Field(..., description="List of skills mentioned in the resume chunk")   