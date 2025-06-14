import os
import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import PyPDF2
import re
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Smart Notes Organizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Direct API key configuration
GEMINI_API_KEY = "key"  # Configure Gemini API
try:
    genai.configure(api_key=GEMINI_API_KEY)
    # Updated to use Gemini 1.5 Pro
    gemini_model = genai.GenerativeModel("gemini-1.5-pro")
    gemini_available = True
except Exception as e:
    st.sidebar.error(f"⚠ Gemini API configuration error: {str(e)}")
    gemini_available = False

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()
dimension = 384  # Dimension of embeddings from 'all-MiniLM-L6-v2'

# Initialize session state
if "notes" not in st.session_state:
    st.session_state.notes = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = np.empty((0, dimension), dtype=np.float32)
if "index" not in st.session_state:
    st.session_state.index = faiss.IndexFlatL2(dimension)
if "tags" not in st.session_state:
    st.session_state.tags = ["School", "Work", "Personal", "Research", "Ideas"]
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Upload"
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "filter_tag" not in st.session_state:
    st.session_state.filter_tag = "All"

# Function to extract text from uploaded files
def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        note_text = ""
        for page in reader.pages:
            note_text += page.extract_text() + "\n"
    else:
        note_text = uploaded_file.read().decode("utf-8")
    return note_text

# Function to add a note
def add_note(note_text, tag, title=None):
    if not title:
        # Generate a title from the first line or first few words
        title = re.split(r'[\.\n]', note_text)[0][:50].strip()
        if not title:
            title = note_text[:50].strip()
        if len(title) == 50:
            title += "..."
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    note = {
        "text": note_text,
        "tag": tag,
        "title": title,
        "timestamp": timestamp,
        "id": len(st.session_state.notes)
    }
    
    st.session_state.notes.append(note)
    
    # Generate embedding for the note
    embedding = embedding_model.encode([note_text])
    st.session_state.embeddings = np.vstack([st.session_state.embeddings, embedding])
    
    # Update the index
    if len(st.session_state.notes) == 1:
        st.session_state.index = faiss.IndexFlatL2(dimension)
        st.session_state.index.add(st.session_state.embeddings)
    else:
        st.session_state.index.add(embedding)
    
    return note

# Function to perform search
def search_notes(query, top_k=5):
    if not st.session_state.notes:
        return []
    
    query_embedding = embedding_model.encode([query])
    distances, indices = st.session_state.index.search(query_embedding, k=min(top_k, len(st.session_state.notes)))
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(st.session_state.notes):  # Ensure valid index
            note = st.session_state.notes[idx]
            score = 1 / (1 + distances[0][i])
            results.append({
                "note": note,
                "score": score
            })
    
    return results

# Function to delete a note
def delete_note(note_id):
    # Find the index of the note with the given id
    idx_to_delete = next((i for i, note in enumerate(st.session_state.notes) if note["id"] == note_id), None)
    
    if idx_to_delete is not None:
        # Remove the note
        st.session_state.notes.pop(idx_to_delete)
        
        # Rebuild embeddings and index
        if st.session_state.notes:
            texts = [note["text"] for note in st.session_state.notes]
            st.session_state.embeddings = embedding_model.encode(texts)
            st.session_state.index = faiss.IndexFlatL2(dimension)
            st.session_state.index.add(st.session_state.embeddings)
        else:
            st.session_state.embeddings = np.empty((0, dimension), dtype=np.float32)
            st.session_state.index = faiss.IndexFlatL2(dimension)
        
        return True
    return False

# Function to generate AI enhancements
def generate_ai_content(note_text, task):
    if not gemini_available:
        return "Gemini API is not properly configured. Please check your API key."
    
    if task == "summarize":
        prompt = f"Summarize the following note in 3-5 key points: {note_text}"
    elif task == "questions":
        prompt = f"Generate 5 study questions based on the following note: {note_text}"
    elif task == "flashcards":
        prompt = f"Create 5 flashcards (question on one side, answer on the other) for the following note: {note_text}"
    elif task == "mindmap":
        prompt = f"Create a simple text-based mind map outline for the following note: {note_text}"
    
    try:
        # Updated to use the Gemini 1.5 Pro model
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating content: {str(e)}"

# Sidebar navigation
st.sidebar.title("📝 Smart Notes Organizer")

# API key input in sidebar
with st.sidebar.expander("🔑 API Settings", expanded=False):
    new_api_key = st.text_input("Enter Gemini API Key:", 
                                value=GEMINI_API_KEY if GEMINI_API_KEY != "YOUR_API_KEY_HERE" else "",
                                type="password",
                                help="Get your API key from https://makersuite.google.com/")
    
    if st.button("Update API Key"):
        try:
            genai.configure(api_key=new_api_key)
            # Updated to use Gemini 1.5 Pro
            gemini_model = genai.GenerativeModel("gemini-1.5-pro")
            gemini_available = True
            st.success("✅ API key updated successfully!")
        except Exception as e:
            st.error(f"⚠ API configuration error: {str(e)}")
            gemini_available = False

# Navigation tabs
tabs = ["Upload", "Browse", "Search", "AI Enhance"]
selected_tab = st.sidebar.radio("Navigation", tabs, index=tabs.index(st.session_state.current_tab))
st.session_state.current_tab = selected_tab

# Stats in sidebar
st.sidebar.divider()
st.sidebar.subheader("📊 Stats")
st.sidebar.metric("Total Notes", len(st.session_state.notes))
if st.session_state.notes:
    tag_counts = {}
    for note in st.session_state.notes:
        tag = note["tag"]
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    for tag, count in tag_counts.items():
        st.sidebar.text(f"{tag}: {count} notes")

# Clear all notes button
st.sidebar.divider()
if st.sidebar.button("🗑 Clear All Notes", type="primary"):
    st.session_state.notes = []
    st.session_state.embeddings = np.empty((0, dimension), dtype=np.float32)
    st.session_state.index = faiss.IndexFlatL2(dimension)
    st.sidebar.success("All notes cleared!")

# Main content
if selected_tab == "Upload":
    st.header("📤 Upload New Note")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload a text file or PDF:", type=["txt", "pdf"])
        
        if uploaded_file is not None:
            note_text = extract_text_from_file(uploaded_file)
            st.text_area("Preview:", value=note_text[:500] + ("..." if len(note_text) > 500 else ""), height=200, disabled=True)
    
    with col2:
        note_title = st.text_input("Note Title (optional):", 
                                   placeholder="Leave blank for auto-generation")
        
        tag = st.selectbox("Tag:", st.session_state.tags)
        
        new_tag = st.text_input("Add New Tag:", placeholder="Enter new tag name")
        if new_tag and st.button("Add Tag"):
            if new_tag not in st.session_state.tags:
                st.session_state.tags.append(new_tag)
                st.success(f"Added new tag: {new_tag}")
                st.rerun()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if uploaded_file is not None and st.button("Save Note", type="primary"):
            note_text = extract_text_from_file(uploaded_file)
            note = add_note(note_text, tag, note_title)
            st.success(f"Note '{note['title']}' saved successfully!")
    
    with col2:
        manual_note = st.checkbox("Or create a note manually")
        
        if manual_note:
            manual_text = st.text_area("Enter your note:", height=200)
            if manual_text and st.button("Save Manual Note", type="primary"):
                note = add_note(manual_text, tag, note_title)
                st.success(f"Note '{note['title']}' saved successfully!")

elif selected_tab == "Browse":
    st.header("📚 Browse Notes")
    
    # Filter options
    col1, col2 = st.columns([1, 2])
    
    with col1:
        filter_tag = st.selectbox("Filter by tag:", ["All"] + st.session_state.tags, 
                                  index=["All"] + st.session_state.tags.index(st.session_state.filter_tag) if st.session_state.filter_tag in st.session_state.tags else 0)
        st.session_state.filter_tag = filter_tag
    
    with col2:
        sort_by = st.selectbox("Sort by:", ["Newest First", "Oldest First", "Title A-Z", "Title Z-A"])
    
    # Display notes
    if not st.session_state.notes:
        st.info("No notes found. Upload a note to get started.")
    else:
        # Filter notes
        filtered_notes = st.session_state.notes
        if filter_tag != "All":
            filtered_notes = [note for note in filtered_notes if note["tag"] == filter_tag]
        
        # Sort notes
        if sort_by == "Newest First":
            filtered_notes = sorted(filtered_notes, key=lambda x: x["timestamp"], reverse=True)
        elif sort_by == "Oldest First":
            filtered_notes = sorted(filtered_notes, key=lambda x: x["timestamp"])
        elif sort_by == "Title A-Z":
            filtered_notes = sorted(filtered_notes, key=lambda x: x["title"])
        elif sort_by == "Title Z-A":
            filtered_notes = sorted(filtered_notes, key=lambda x: x["title"], reverse=True)
        
        # Display notes in a grid
        num_cols = 3
        cols = st.columns(num_cols)
        
        for i, note in enumerate(filtered_notes):
            with cols[i % num_cols]:
                with st.expander(f"{note['title']}", expanded=False):
                    st.write(f"*Tag:* {note['tag']}")
                    st.write(f"*Added:* {note['timestamp']}")
                    st.write("*Content:*")
                    st.text_area("", value=note['text'], height=200, disabled=True, key=f"note-{note['id']}")
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("🗑 Delete", key=f"delete-{note['id']}"):
                            if delete_note(note['id']):
                                st.success("Note deleted!")
                                st.rerun()
                    
                    with col2:
                        if st.button("✨ Enhance", key=f"enhance-{note['id']}"):
                            st.session_state.current_tab = "AI Enhance"
                            st.session_state.selected_note_id = note['id']
                            st.rerun()

elif selected_tab == "Search":
    st.header("🔍 Search Notes")
    
    query = st.text_input("Enter your search query:", placeholder="Search for keywords or concepts...")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        search_button = st.button("Search", type="primary", use_container_width=True)
    
    with col2:
        num_results = st.slider("Number of results:", 1, 10, 5)
    
    if search_button and query:
        st.session_state.search_results = search_notes(query, num_results)
        
    # Display search results
    if "search_results" in st.session_state and st.session_state.search_results:
        st.subheader(f"Search Results for '{query}'")
        
        for i, result in enumerate(st.session_state.search_results):
            note = result["note"]
            score = result["score"]
            
            with st.expander(f"{note['title']}** (Relevance: {score:.2f})", expanded=True):
                st.write(f"*Tag:* {note['tag']} | *Added:* {note['timestamp']}")
                
                # Highlight search terms
                note_text = note['text']
                if query.lower() in note_text.lower():
                    # Simple highlighting for exact matches
                    highlighted_text = re.sub(f'({re.escape(query)})', r'\1**', note_text, flags=re.IGNORECASE)
                    st.markdown(highlighted_text[:500] + ("..." if len(highlighted_text) > 500 else ""))
                else:
                    st.write(note_text[:500] + ("..." if len(note_text) > 500 else ""))
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("View Full Note", key=f"view-{note['id']}"):
                        st.session_state.current_tab = "Browse"
                        st.session_state.filter_tag = note['tag']
                        st.rerun()
                
                with col2:
                    if st.button("✨ Enhance", key=f"enhance-search-{note['id']}"):
                        st.session_state.current_tab = "AI Enhance"
                        st.session_state.selected_note_id = note['id']
                        st.rerun()
    
    elif search_button and not query:
        st.warning("Please enter a search query.")
    
    elif search_button and not st.session_state.search_results:
        st.info("No results found. Try a different search query.")

elif selected_tab == "AI Enhance":
    st.header("✨ Enhance Notes with AI")
    
    if not gemini_available:
        st.warning("⚠ Gemini API is not properly configured. Please enter your API key in the sidebar.")
    
    if not st.session_state.notes:
        st.info("No notes found. Upload a note to get started.")
    else:
        # Select a note to enhance
        if "selected_note_id" in st.session_state:
            default_idx = next((i for i, note in enumerate(st.session_state.notes) 
                              if note["id"] == st.session_state.selected_note_id), 0)
        else:
            default_idx = 0
        
        note_options = [f"{note['title']} ({note['tag']})" for note in st.session_state.notes]
        selected_note_idx = st.selectbox("Select a note to enhance:", 
                                        range(len(note_options)), 
                                        format_func=lambda i: note_options[i],
                                        index=default_idx)
        
        selected_note = st.session_state.notes[selected_note_idx]
        
        # Clear selected note ID if it exists
        if "selected_note_id" in st.session_state:
            del st.session_state.selected_note_id
        
        # Show the selected note
        st.subheader(f"Selected Note: {selected_note['title']}")
        with st.expander("View Note Content", expanded=False):
            st.write(selected_note["text"])
        
        # AI enhancement options
        st.subheader("Choose Enhancement")
        
        enhancement_options = {
            "summarize": "📝 Summarize",
            "questions": "❓ Generate Study Questions",
            "flashcards": "🎴 Create Flashcards",
            "mindmap": "🧠 Create Mind Map"
        }
        
        cols = st.columns(len(enhancement_options))
        selected_option = None
        
        for i, (key, name) in enumerate(enhancement_options.items()):
            with cols[i]:
                if st.button(name, use_container_width=True, disabled=not gemini_available):
                    selected_option = key
        
        if selected_option:
            with st.spinner(f"Generating {enhancement_options[selected_option].split(' ')[1]}..."):
                result = generate_ai_content(selected_note["text"], selected_option)
                
                if not result.startswith("Error") and not result.startswith("Gemini API is not properly"):
                    st.success(f"{enhancement_options[selected_option]} generated successfully!")
                
                if selected_option == "summarize":
                    st.subheader("📝 Summary")
                    st.write(result)
                
                elif selected_option == "questions":
                    st.subheader("❓ Study Questions")
                    st.write(result)
                
                elif selected_option == "flashcards":
                    st.subheader("🎴 Flashcards")
                    
                    # Parse flashcards
                    flashcards = re.split(r'\n\s*\n', result)
                    
                    for i, card in enumerate(flashcards):
                        if ":" in card or "-" in card:
                            # Try to split by common patterns
                            if ":" in card:
                                parts = card.split(":", 1)
                            else:
                                parts = card.split("-", 1)
                            
                            if len(parts) == 2:
                                question, answer = parts
                                with st.expander(f"Question {i+1}: {question.strip()}", expanded=False):
                                    st.write(f"*Answer:* {answer.strip()}")
                            else:
                                st.write(card)
                        else:
                            st.write(card)
                
                elif selected_option == "mindmap":
                    st.subheader("🧠 Mind Map")
                    st.write(result)
