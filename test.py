import google.generativeai as genai  # Google Gemini API ke liye
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO
import base64
import json
import time

# FastAPI
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
import uvicorn
import nest_asyncio
from dotenv import load_dotenv

load_dotenv()
nest_asyncio.apply()

# FastAPI app create karna
app = FastAPI()

# Google Gemini API Key Setup
def get_gemini_model():
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GEMINI_API")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini model initialization failed: {str(e)}")

# Request model define karna
class BookRequest(BaseModel):
    title: str = Field(..., title="Book Title", min_length=1)
    subtitle: str = Field(..., title="Subtitle", min_length=1)
    author: str = Field(..., title="Author Name", min_length=1)
    style: str = Field(..., title="Writing Style", min_length=1)
    language: str = Field(..., title="Language", min_length=1)
    length: str = Field(..., title="Target Length", min_length=1)
    num_chapters: int = Field(..., title="Number of Chapters", gt=0)
    sub_chapters: int = Field(..., title="Sub-chapters per Chapter", gt=0)
    goal: str = Field(..., title="Goal of the Book", min_length=1)
    audience: str = Field(..., title="Target Audience", min_length=1)
    tone: str = Field(..., title="Tone of the Book", min_length=1)
    genre: str = Field(..., title="Genre", min_length=1)

# Prompt generate karne ka function
def format_prompt(request: BookRequest):
    prompt = f"""
    You are an advanced AI specializing in crafting deeply engaging autobiographies.
    Your task is to generate a high-quality, immersive autobiography based on the given details.

    Book Specifications:
    - Title: {request.title}
    - Subtitle: {request.subtitle}
    - Author: {request.author}
    - Writing Style: {request.style}
    - Language: {request.language}
    - Target Length: {request.length} words
    - Number of Chapters: {request.num_chapters}
    - Sub-chapters per Chapter: {request.sub_chapters}
    - Main Goal: {request.goal}
    - Target Audience: {request.audience}
    - Tone: {request.tone}
    - Genre: {request.genre}

    Step 1: Generate a structured Table of Contents with concise chapter headings.
    Step 2: Write a complete and immersive autobiography based on the table of contents.
    
    First, return only the Table of Contents. After that, generate the book content.
    """
    return prompt

# AI response handle karne ka function
def generate_book_content(prompt, gemini_model):
    try:
        response = gemini_model.invoke(prompt)
        if not response or not response.content:
            raise HTTPException(status_code=500, detail="AI failed to generate book content.")
        return response.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in generating book: {str(e)}")

# Extract chapter titles from the AI-generated table of contents
def extract_chapter_titles(book_content):
    chapters = []
    lines = book_content.split("\n")
    for line in lines:
        if line.strip() and line[0].isdigit():  # Checking numbered chapters
            chapters.append(line.strip())
    return chapters

# Generate AI image using Gemini API
def generate_chapter_image(chapter_title, gemini_model):
    try:
        prompt = f"Create a realistic, high-quality illustration for the chapter titled '{chapter_title}'."
        response = gemini_model.invoke(prompt, response_format="image")
        
        if not response or not response.content:
            raise HTTPException(status_code=500, detail="Failed to generate image.")

        return response.content  # This should be a base64-encoded image
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")

# Save book as PDF with images
def save_book_as_pdf(book_content, chapter_images):
    pdf_filename = "generated_book.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=letter)

    y_position = 750  # Starting position for text

    for chapter, image_data in zip(book_content, chapter_images):
        c.setFont("Helvetica-Bold", 14)
        c.drawString(100, y_position, chapter)
        y_position -= 30

        # Convert base64 image to BytesIO
        image_bytes = BytesIO(base64.b64decode(image_data))
        image = ImageReader(image_bytes)

        c.drawImage(image, 100, y_position - 100, width=400, height=300)
        y_position -= 350

        # Add Chapter Content
        c.setFont("Helvetica", 12)
        c.drawString(100, y_position, "Chapter content goes here...")
        y_position -= 50

        c.showPage()

    c.save()
    return pdf_filename

# Book generate karne ka API endpoint
@app.post("/generate_book_with_images")
async def generate_book_with_images(request: BookRequest, gemini_model=Depends(get_gemini_model)):
    try:
        # Generate Prompt & AI Response
        prompt = format_prompt(request)
        book_content = generate_book_content(prompt, gemini_model)

        # Extract Chapter Titles
        chapter_titles = extract_chapter_titles(book_content)
        
        # Generate Images for Each Chapter
        chapter_images = [generate_chapter_image(title, gemini_model) for title in chapter_titles]

        # Save as PDF
        pdf_file = save_book_as_pdf(chapter_titles, chapter_images)

        return {"status": True, "message": "Book generated successfully with images", "pdf": pdf_file}
    
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# FastAPI server run karne ka code
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
