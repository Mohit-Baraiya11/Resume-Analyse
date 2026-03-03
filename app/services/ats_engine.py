from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.pdf_reader import extract_text_from_pdf
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from app.services.pydantic_validation import ExtractResumeInfo, ResumeChunk
import os
from dotenv import load_dotenv


def analyze_resume(file_path, job_description):
    load_dotenv()
    
    #model
    from langchain_groq import ChatGroq
    model = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    #extracting text from pdf
    text = extract_text_from_pdf(file_path)

    #pydantic parse skills from pdf
    pydantic_pdf_extract_parser = PydanticOutputParser(pydantic_object=ExtractResumeInfo)
    #get skills from llm
    skills_from_pdf_prompt = PromptTemplate(
        template="""
            You are an information extraction system.

            Extract ONLY the technical skills from the resume below.

            Return the result strictly as valid JSON.
            Do not include explanations.
            Do not include schema.
            Only return the JSON object.

            Resume:
            {text}

            {format_instructions}
            """,
                input_variables=["text"],
                partial_variables={
                    "format_instructions": pydantic_pdf_extract_parser.get_format_instructions()
                }
        )
    #skills chain
    skill_chain = skills_from_pdf_prompt|model|pydantic_pdf_extract_parser     
    skills = skill_chain.invoke({"text": text}).skills

    job_description = job_description.strip()


    resume_score_parser = PydanticOutputParser(pydantic_object=ResumeChunk)

    analysis_prompt = PromptTemplate(
    template="""
            You are a professional ATS evaluation system.

            Evaluate the candidate strictly based on provided information.

            JOB SKILLS REQUIRED:
            {job_description}

            CANDIDATE SKILLS:
            {resume_skills}

            Rules:

            1. Compare required skills with candidate skills.
            2. Identify matched skills.
            3. Identify missing skills.
            4. Calculate match_score as:
            matched_required_skills divided by total_required_skills.
            Round to 2 decimals.
            5. If no required skills are missing:
            - Do NOT leave missing_skills empty.
            - Add at least 1-2 advanced or complementary skills related to the job that are not explicitly mentioned.
            6. Always provide improvement_suggestions.
            - Even if skills fully match, suggest advanced improvements or deeper expertise.
            7. Provide professional advice:
            - Should the candidate apply for this job?
            - Justify briefly.
            8. Do NOT generate Python code.
            9. Do NOT write markdown.
            10. Return ONLY valid JSON.
            11. No text before or after JSON.

            {format_instructions}
            """,
                input_variables=["job_description", "resume_skills"],
                partial_variables={
                    "format_instructions": resume_score_parser.get_format_instructions()
                }
            )

    #chain
    resume_analysis_chain = analysis_prompt | model | resume_score_parser

    resume_analysis = resume_analysis_chain.invoke({
        "job_description": job_description,
        "resume_skills": skills
    })
    return resume_analysis.model_dump()

