import openai  # OpenAI ke models (ChatGPT) ke liye
import google.generativeai as genai  # Google Gemini API ke liye
import os  # Environment variables ke liye

# LangChain ke components
from langchain.schema import HumanMessage  # LLM messages ke liye
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, AIMessage
# LangGraph ke components
import langgraph
from langgraph.graph import Graph  # AI workflows ke liye Graph-based approach
from reportlab.lib.pagesizes import letter  # PDF generation ke liye
from reportlab.pdfgen import canvas  # PDF generation ke liye

# Miscellaneous
import json  # JSON handling ke liye
import time  # Execution timing ke liye

# Google Gemini API Key Setup

genai.configure(api_key=os.getenv("GEMINI_API"))

def initialize_gemini_model():
    """
    Google Gemini model ko initialize karta hai.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GEMINI_API")
    )

gemini_model = initialize_gemini_model()

def get_model_parameters():
    """
    AI model ke parameters define karta hai.
    """
    parameters = {
        "temperature": 0.7,  # Creativity control
    }
    return parameters

def format_prompt(title, subtitle, author, style, language, length):
    """
    Generates an optimized prompt for AI to write an immersive, realistic autobiography.
    """

    prompt = f"""
    You are an advanced AI specializing in crafting deeply engaging autobiographies. 
    Your task is to generate a high-quality, immersive, and emotionally resonant autobiography based on the given details.

    Book Specifications:
    - Title: {title}
    - Subtitle: {subtitle}
    - Author: {author}
    - Writing Style: {style} (Engaging, immersive, first-person storytelling)
    - Language: {language}
    - Target Length: {length} words (Ensure a detailed, comprehensive narrative)
    
    Step 1: Book Structure & Table of Contents
    First, generate a structured Table of Contents with only concise chapter headings.
    - Each chapter should have only a title, without any extra descriptions.
    - The structure should cover all key phases of the author's journey.

    Example:
    
    Table of Contents:
    1. Early Days & Childhood
    2. First Struggles
    3. Breakthrough Moments
    4. Reflections & Lessons

    Step 2: Full-Length Immersive Autobiography
    Based on the structured Table of Contents, write a complete and immersive autobiography:
    - First-Person Perspective: Narrate as if the author is telling their own story.
    - Seamless Flow: Ensure the story progresses naturally across different phases.
    - Expanded Key Moments: Dive deep into emotions, thoughts, and experiences.
    - Sensory & Emotional Detailing: Describe places, people, and events vividly.
    - Authenticity: The storytelling should feel like a genuine human-written memoir.
    - No Artificial Padding: Maintain natural storytelling without unnecessary repetition.

    Output Expectations:
    - Emotional Depth: Capture the highs and lows in a gripping, cinematic way.
    - Realistic Feel: Make it feel like an actual autobiography of a real person.
    - Logical Progression: No abrupt stops, unnatural breaks, or repetitive fillers.
    - Balanced Length: Organically maintain the word count through rich storytelling.

    First, generate the Table of Contents with concise chapter headings, and then write the complete immersive autobiography based on it.
    """
    return prompt

def generate_book_content(title, subtitle, author, style, language, length):
    """
    Gemini AI model se book ka content generate karta hai.
    """
    prompt = format_prompt(title, subtitle, author, style, language, length)  # Prompt generate karna

    response = gemini_model.invoke(prompt)  # AI Model ko call karna

    # Adding author's name in the beginning of the generated content
    content = f"By {author}\n\n" + response.content  # Add author name at the top of the content

    return content  # AI-generated book content return karega

def clean_generated_content(content):
    """
    Removes any unwanted symbols or characters like '*' or other special characters.
    """
    content = content.replace("*", "").strip()  # Remove '*' and extra whitespace
    return content

def split_content_into_chunks(book_content, chunk_size=2000):
    """
    AI-generated book content ko chhoti chunks me todta hai.
    """
    # Clean content before splitting
    book_content = clean_generated_content(book_content)
    return [book_content[i:i + chunk_size] for i in range(0, len(book_content), chunk_size)]

def save_book_to_file(book_chunks, filename="AI_Generated_Book.txt"):
    """
    AI-generated book content ko ek text file me save karta hai.
    """
    with open(filename, "w", encoding="utf-8") as file:
        for chunk in book_chunks:
            file.write(chunk + "\n\n")  # Har chunk ke baad newline add karenge
    return f"Book content saved successfully: {filename}"

def generate_and_save_book(title, subtitle, author, style, language, length):
    """
    AI se book generate karta hai aur usko ek file me save karta hai.
    """
    book_content = generate_book_content(title, subtitle, author, style, language, length)  # Content generate karo
    book_chunks = split_content_into_chunks(book_content)  # Content ko chunks me split karo
    result_message = save_book_to_file(book_chunks)  # File me save karo
    return result_message

# Main logic to take inputs from the user
if __name__ == "__main__":
    title = input("Enter Book Title: ")
    subtitle = input("Enter Subtitle: ")
    author = input("Enter Author Name: ")
    style = input("Enter Writing Style (e.g., Storytelling, Informative): ")
    language = input("Enter Language: ")
    length = input("Enter Approximate Length (e.g., 5000 words, 10 pages): ")

    result = generate_and_save_book(title, subtitle, author, style, language, length)  # Generate and save book
    print(result)
