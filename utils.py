import re
import json
import uuid
from datetime import datetime

import streamlit as st

def load_notes():
    try:
        with open("data/notes.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_notes(notes):
    with open("data/notes.json", "w") as f:
        json.dump(notes, f, indent=2)

def add_note(title, text, tag):
    note = {
        "id": str(uuid.uuid4()),
        "title": title,
        "text": text,
        "tag": tag,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.notes.append(note)
    save_notes(st.session_state.notes)

def delete_note(note_id):
    notes = [note for note in st.session_state.notes if note["id"] != note_id]
    if len(notes) != len(st.session_state.notes):
        st.session_state.notes = notes
        save_notes(notes)
        return True
    return False

def generate_ai_content(text, option):
    import google.generativeai as genai
    from api_key import gemini_api_key

    if not gemini_api_key:
        return "Gemini API is not properly configured."

    genai.configure(api_key=gemini_api_key)

    model = genai.GenerativeModel("gemini-pro")
    try:
        if option == "summarize":
            prompt = f"Summarize the following note:\n{text}"
        elif option == "questions":
            prompt = f"Generate study questions based on this note:\n{text}"
        elif option == "flashcards":
            prompt = f"Create flashcards from the note:\n{text}"
        elif option == "mindmap":
            prompt = f"Create a mind map of this note:\n{text}"
        else:
            return "Invalid enhancement option."

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Error: {e}"

def search_notes(query, num_results=5):
    results = []
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    for note in st.session_state.notes:
        matches = pattern.findall(note["text"])
        if matches:
            score = len(matches)
            results.append({"note": note, "score": score})
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:num_results]
