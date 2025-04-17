import os
os.environ['STREAMLIT_SERVER_ENABLE_STATIC_SERVING'] = 'false'

from simple_rag import graph
import streamlit as st
import json
from io import StringIO
import tiktoken
import time
from langchain_community.document_loaders import PyMuPDFLoader
import traceback
import sqlite3  # Import SQLite

# Token limits
GPT_LIMIT = 128000
GEMINI_LIMIT = 1000000

# Token counters
def count_tokens_gpt(text):
    enc = tiktoken.encoding_for_model("gpt-4")
    return len(enc.encode(text))

def count_tokens_gemini(text):
    return len(text.split())  # Approximation

# Calculate tokens for the entire context window
def calculate_context_window_usage(json_data=None):
    # Reconstruct the full conversation context
    full_conversation = ""
    for sender, message in st.session_state.chat_history:
        full_conversation += f"{sender}: {message}\n\n"
    
    # Add JSON context if provided
    if json_data:
        full_conversation += json.dumps(json_data)
        
    gpt_tokens = count_tokens_gpt(full_conversation)
    gemini_tokens = count_tokens_gemini(full_conversation)
    
    return gpt_tokens, gemini_tokens
def load_pdf(path):
    docs = PyMuPDFLoader(path)
    return docs

docs = load_pdf(path = '/home/pavan/Desktop/FOLDERS/RUBIC/RAG_without_profiler/RAG_rubik/r_IndiaInvestments.pdf')
# Page configuration
st.set_page_config(page_title="📊 RAG Chat Assistant", layout="wide")

# --- Session state setup ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        ("assistant", "👋 Hello! I'm your RAG assistant. Please upload your JSON files and ask me a question about your portfolio.")
    ]
if "processing" not in st.session_state:
    st.session_state.processing = False
if "total_gpt_tokens" not in st.session_state:
    st.session_state.total_gpt_tokens = 0  # Total accumulated
if "total_gemini_tokens" not in st.session_state:
    st.session_state.total_gemini_tokens = 0  # Total accumulated
if "window_gpt_tokens" not in st.session_state:
    st.session_state.window_gpt_tokens = 0  # Current context window
if "window_gemini_tokens" not in st.session_state:
    st.session_state.window_gemini_tokens = 0  # Current context window

# --- Database setup ---
DATABASE_PATH = "Data/chat_history.db"

def initialize_database():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender TEXT,
            message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def load_chat_history():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT sender, message FROM chat_history ORDER BY timestamp")
    history = cursor.fetchall()
    conn.close()
    return history

def save_message(sender, message):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_history (sender, message) VALUES (?, ?)", (sender, message))
    conn.commit()
    conn.close()

# Initialize database
initialize_database()

# Load chat history from the database when the app starts
if "chat_history" not in st.session_state or not st.session_state.chat_history:
    st.session_state.chat_history = load_chat_history()
    if not st.session_state.chat_history:
        st.session_state.chat_history = [("assistant", "👋 Hello! I'm your RAG assistant. Please upload your JSON files and ask me a question about your portfolio.")]

# --- Layout: Chat UI Left | Progress Bars Right ---
col_chat, col_progress = st.columns([3, 1])

# --- LEFT COLUMN: Chat UI ---
with col_chat:
    st.title("💬 RAG Assistant")

    with st.expander("📂 Upload Required JSON Files", expanded=True):
        user_data_file = st.file_uploader("Upload user_data.json", type="json", key="user_data")
        allocations_file = st.file_uploader("Upload allocations.json", type="json", key="allocations")
        
        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.chat_history = [
                ("assistant", "👋 Hello! I'm your RAG assistant. Please upload your JSON files and ask me a question about your portfolio.")
            ]
            st.session_state.total_gpt_tokens = 0
            st.session_state.total_gemini_tokens = 0
            st.session_state.window_gpt_tokens = 0
            st.session_state.window_gemini_tokens = 0
            
            # Clear the chat history in the database
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_history")
            conn.commit()
            conn.close()
            
            st.rerun()

    st.markdown("---")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for sender, message in st.session_state.chat_history:
            if sender == "user":
                st.chat_message("user").write(message)
            else:
                st.chat_message("assistant").write(message)
        
        # Show thinking animation if processing
        if st.session_state.processing:
            thinking_placeholder = st.empty()
            with st.chat_message("assistant"):
                for i in range(3):
                    for dots in [".", "..", "..."]:
                        thinking_placeholder.markdown(f"Thinking{dots}")
                        time.sleep(0.3)

    # Input box at the bottom
    user_input = st.chat_input("Type your question...")

    if user_input and not st.session_state.processing:
        # Set processing flag
        st.session_state.processing = True
        
        # Add user message to history immediately
        st.session_state.chat_history.append(("user", user_input))
        save_message("user", user_input)  # Save user message to DB
        
        # Force a rerun to show the message and thinking indicator
        st.rerun()

# This part runs after the rerun if we're processing
if st.session_state.processing:
    if not user_data_file or not allocations_file:
        st.session_state.chat_history.append(("assistant", "⚠️ Please upload both JSON files before asking questions."))
        st.session_state.processing = False
        st.rerun()
    else:
        try:
            # Load JSONs
            user_data = json.load(StringIO(user_data_file.getvalue().decode("utf-8")))
            allocations = json.load(StringIO(allocations_file.getvalue().decode("utf-8")))
            
            # Combined JSON data (for token calculation)
            combined_json_data = {"user_data": user_data, "allocations": allocations}

            # Get the last user message
            last_user_message = next((msg for sender, msg in reversed(st.session_state.chat_history) if sender == "user"), "")
            
            # Count tokens for this user message
            user_msg_gpt_tokens = count_tokens_gpt(last_user_message)
            user_msg_gemini_tokens = count_tokens_gemini(last_user_message)
            
            # Add to accumulated totals
            st.session_state.total_gpt_tokens += user_msg_gpt_tokens
            st.session_state.total_gemini_tokens += user_msg_gemini_tokens
            
            # Calculate context window usage (conversation + JSON data)
            window_gpt, window_gemini = calculate_context_window_usage(combined_json_data)
            st.session_state.window_gpt_tokens = window_gpt
            st.session_state.window_gemini_tokens = window_gemini

            # Check token limits for context window
            if window_gpt > GPT_LIMIT or window_gemini > GEMINI_LIMIT:
                st.session_state.chat_history.append(("assistant", "⚠️ Your conversation has exceeded token limits. Please clear the chat to continue."))
                st.session_state.processing = False
                st.rerun()
            else:
                # --- Call LangGraph ---
                inputs = {
                    "query": last_user_message,
                    "user_data": user_data,
                    "allocations": allocations,
                    "pdf":docs,
                    "memory": st.session_state.chat_history
                }

                

                response = graph.invoke(inputs)['result']
                

                # Count tokens for the response
                response_gpt_tokens = count_tokens_gpt(response)
                response_gemini_tokens = count_tokens_gemini(response)
                
                # Add to accumulated totals
                st.session_state.total_gpt_tokens += response_gpt_tokens
                st.session_state.total_gemini_tokens += response_gemini_tokens

                # Add to chat history
                st.session_state.chat_history.append(("assistant", response))
                save_message("assistant", response)  # Save assistant message to DB
                
                # Update context window calculations after adding response
                window_gpt, window_gemini = calculate_context_window_usage(combined_json_data)
                st.session_state.window_gpt_tokens = window_gpt
                st.session_state.window_gemini_tokens = window_gemini
                
        except Exception as e:
            tb = traceback.extract_stack()
            filename, line_number, function_name, text = tb[-2]
            error_message = f"❌ Error: {str(e)} in {filename} at line {line_number}, function: {function_name}"
            st.session_state.chat_history.append(("assistant", error_message))
        
        # Reset processing flag
        st.session_state.processing = False
        st.rerun()

# --- RIGHT COLUMN: Progress Bars ---
with col_progress:
    st.markdown("### 📊 Context Window Usage")
    
    # Current window usage (conversation + JSON)
    window_gpt_tokens = st.session_state.window_gpt_tokens
    window_gemini_tokens = st.session_state.window_gemini_tokens
    
    window_gpt_percent = min(int((window_gpt_tokens / GPT_LIMIT) * 100), 100)
    window_gemini_percent = min(int((window_gemini_tokens / GEMINI_LIMIT) * 100), 100)

    st.markdown("**🔷 GPT-4o Context Window**")
    st.progress(window_gpt_percent / 100, text=f"{window_gpt_tokens} / {GPT_LIMIT} tokens ({window_gpt_percent}%)")

    st.markdown("**🔶 Gemini Context Window**")
    st.progress(window_gemini_percent / 100, text=f"{window_gemini_tokens} / {GEMINI_LIMIT} tokens ({window_gemini_percent}%)")
    
    # Total accumulated tokens
    st.markdown("### 📈 Total Tokens Processed")
    total_gpt = st.session_state.total_gpt_tokens
    total_gemini = st.session_state.total_gemini_tokens
    
    st.markdown(f"**GPT-4o**: {total_gpt} tokens")
    st.markdown(f"**Gemini**: {total_gemini} tokens")
    
    # Display status
    st.markdown("### Status")
    if st.session_state.processing:
        st.markdown("🔄 **Processing...**")
    else:
        st.markdown("✅ **Ready**")
    

    

