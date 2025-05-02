# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import requests  # Use requests library for HF API
import json  # To handle JSON response
import time  # To add slight delay and for retry logic
from typing import Any, Dict, List, Optional  # For type hinting
import logging  # For better logging

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.DEBUG,  # Set to DEBUG to see more details like prompts/payloads
                    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Student Assessment Dashboard")

# --- Hugging Face Setup ---
API_URL: str = "https://api-inference.huggingface.co/models/google/flan-t5-large"
MODEL_NAME: str = API_URL.split('/')[-1]  # Extract model name for display
MODEL_MAX_NEW_TOKENS_LIMIT = 250  # Safe limit for generated tokens

# --- Constants & Mappings ---
CORRECT_ANSWERS: Dict[str, str] = {
    "What is the output of the following code?\nprint(2 + 3 * 4)": "14",
    "What is the output of the following code?\n\nnum = 8  \nif num % 2 == 0:  \n    print(\"Even\")  \nelse:  \n    print(\"Odd\")": "Even",
    "What is the output of the following code?\n\nx = \"Hello\"\ny = \"World\"\nprint(x + \" \" + y)": "Hello World",
    "What is the output of the following code?\na = 10  \nb = 5  \na, b = b, a  \nprint(a, b)": "5 10",
    "What is the output of the following code?\nfruits = [\"apple\", \"banana\", \"cherry\"]  \nprint(fruits[1])": "banana",
    "What is the output of the following code?\nprint(len(\"Python\"))": "6",
    "What is the output of the following code?\nx = [1, 2, 3]  \nx.append(4)  \nprint(x)": "[1, 2, 3, 4]",
    "What is the output of the following code?\n\ndef greet(name):  \n    return \"Hello \" + name\n\nprint(greet(\"Alex\"))": "Hello Alex",
    "What is the output of the following code?\n\nprint(type(3.14))": "float",
    "What is the output of the following code?\n\nmy_dict = {\"name\": \"Alex\", \"age\": 21}  \nprint(my_dict[\"name\"])": "Alex"
}
CODING_QUESTION_COLS: List[str] = list(CORRECT_ANSWERS.keys())

ANSWER_MAPPING: Dict[str, str] = {
    "Option 1": "Even", "<class 'float'>": "float", "<type 'float'>": "float",
    "HelloWorld": "Hello World", "Helloworld": "Hello World", "1 2 3 4": "[1, 2, 3, 4]",
    "puthin": "Python", "name": "Alex", "21": "Alex", "Alex Hello": "Hello Alex",
    "fruits[1]": "banana", 'my_dict["name"]': "Alex",
}

# --- Helper Functions ---


def normalize_answer(answer: Any) -> str:
    """Normalizes student answers for comparison."""
    if pd.isna(answer):
        return ""
    ans_str = str(answer).strip().lower()
    # Apply specific known mappings first
    for key, value in ANSWER_MAPPING.items():
        if ans_str == key.lower():
            return value.strip().lower().replace('"', '').replace("'", "")
    # General normalizations
    if ans_str.startswith("<class '") and ans_str.endswith("'>"):
        ans_str = ans_str[8:-2]
    elif ans_str.startswith("<type '") and ans_str.endswith("'>"):
        ans_str = ans_str[7:-2]
    ans_str = ans_str.replace('"', '').replace("'", "").replace(
        "[", "").replace("]", "").replace("(", "").replace(")", "")
    ans_str = ' '.join(ans_str.split())
    return ans_str


@st.cache_data  # Cache data loading
def load_data(filepath: str) -> Optional[pd.DataFrame]:
    """Loads student response data from a CSV file."""
    logging.info(f"Attempting to load data from: {filepath}")
    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
        logging.info(f"CSV Columns loaded: {df.columns.tolist()}")
        if 'Email Address' not in df.columns:
            st.error("CSV file must contain an 'Email Address' column.")
            logging.error("CSV missing 'Email Address' column.")
            return None
        df['Email Address'] = df['Email Address'].astype(str).fillna('Unknown')
        logging.info(
            f"Data loaded successfully from '{filepath}', shape: {df.shape}")
        return df
    except FileNotFoundError:
        st.error(
            f"Error: The file '{filepath}' was not found. Please ensure it's in the same directory or provide the correct path.")
        logging.error(f"FileNotFoundError: {filepath}")
        return None
    except pd.errors.EmptyDataError:
        st.error(f"Error: The file '{filepath}' is empty.")
        logging.error(f"EmptyDataError: {filepath}")
        return None
    except Exception as e:
        st.error(
            f"An error occurred while loading or processing the data: {e}")
        logging.exception("Error loading data:")
        return None


def get_hf_api_token() -> Optional[str]:
    """Safely retrieves the Hugging Face API token from Streamlit secrets."""
    try:
        token = st.secrets["HF_API_TOKEN"]
        if not token:
            st.error(
                "Hugging Face API token (HF_API_TOKEN) found in secrets, but it is empty.")
            logging.error("HF_API_TOKEN secret is empty.")
            st.stop()
        # Basic check if token looks like a HF token (starts with hf_)
        if not token.startswith("hf_"):
            st.warning(
                "The HF_API_TOKEN in secrets doesn't seem to start with 'hf_'. Please ensure it's a valid token.")
            logging.warning("HF_API_TOKEN does not start with hf_.")
        return token
    except KeyError:
        st.error("Hugging Face API token (HF_API_TOKEN) not found. Please add it to your Streamlit secrets (e.g., `.streamlit/secrets.toml`).")
        st.info(
            "Example secrets.toml:\n```toml\nHF_API_TOKEN = 'hf_YourActualToken'\n```")
        logging.error("HF_API_TOKEN not found in Streamlit secrets.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while accessing Streamlit secrets: {e}")
        logging.exception("Error accessing secrets:")
        st.stop()

# --- CORRECTED generate_hf_insight function ---
# @st.cache_data(show_spinner=False) # Keep caching disabled for now during testing


def generate_hf_insight(prompt: str, max_new_tokens: int = 150, temperature: float = 0.7) -> str:
    """
    Generates text using Hugging Face Inference API with improved error handling,
    logging, retry logic, prompt length check, and timeout.
    (Corrected version: Removed 'return_full_text' parameter)
    """
    api_token = get_hf_api_token()
    # get_hf_api_token() already calls st.stop() if no token

    # --- Prompt Length Check ---
    estimated_input_tokens = len(prompt) / 4  # Approximation
    # Using 768 as a slightly safer limit for T5 base/large input sequence length + output
    # Note: Precise limit depends on exact tokenizer and API backend config
    total_token_limit = 768
    if estimated_input_tokens + max_new_tokens > total_token_limit:
        st.warning(
            f"Potential Prompt Too Long: Estimated input tokens ({estimated_input_tokens:.0f}) + max new tokens ({max_new_tokens}) might exceed model limit (~{total_token_limit}). Consider shortening the prompt.")
        logging.warning(
            f"Potential prompt length issue. Estimated input: {estimated_input_tokens:.0f} tokens. Prompt (start): {prompt[:100]}...")

    safe_max_tokens = min(max_new_tokens, MODEL_MAX_NEW_TOKENS_LIMIT)
    if safe_max_tokens != max_new_tokens:
        logging.warning(
            f"Requested max_new_tokens {max_new_tokens} reduced to {safe_max_tokens} to respect model limit.")

    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": safe_max_tokens,
            "temperature": max(0.1, temperature),  # Ensure temp is not zero
            # --- Removed 'return_full_text': False ---
        },
        "options": {
            "wait_for_model": True,
            "use_cache": False
        }
    }

    retries = 3
    delay = 5  # seconds
    timeout_seconds = 120

    logging.info(
        f"Calling HF API. Endpoint: {API_URL}, Max New Tokens: {safe_max_tokens}, Temp: {payload['parameters']['temperature']}")
    logging.debug(f"Prompt (start): {prompt[:200]}...")
    # Log the exact payload prettily
    logging.debug(f"Payload being sent: {json.dumps(payload, indent=2)}")

    for i in range(retries):
        try:
            response = requests.post(
                API_URL, headers=headers, json=payload, timeout=timeout_seconds)

            logging.info(
                f"HF API Call Attempt {i+1}/{retries}. Status Code: {response.status_code}")
            try:
                response_text = response.text
                logging.debug(
                    f"Raw Response Text (first 500 chars): {response_text[:500]}")
            except Exception as log_e:
                logging.warning(f"Could not log raw response text: {log_e}")

            # Check for HTTP errors (like the 400 we saw)
            response.raise_for_status()

            result = response.json()
            logging.debug(f"Parsed JSON Response: {result}")

            if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
                generated_text = result[0]['generated_text'].strip()
                logging.info(
                    f"Successfully generated text (length: {len(generated_text)}).")
                return generated_text
            elif isinstance(result, dict) and 'error' in result:
                error_msg = f"Hugging Face API returned an error object: {result['error']}"
                estimated_time = result.get('estimated_time')
                if estimated_time:
                    error_msg += f" (Model may be loading, estimated time: {estimated_time:.1f}s)"
                st.error(error_msg)
                logging.error(f"API Error Dict Received: {result}")
                if "currently loading" in result['error'].lower() and i < retries - 1:
                    st.warning(f"Model loading detected, retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                return f"Error: {result['error']}"  # Return API error directly
            else:
                st.warning(f"Unexpected response format from HF API: {result}")
                logging.warning(f"Unexpected response format: {result}")
                return "Error: Could not parse the generated text from the API response. Unexpected format."

        except requests.exceptions.Timeout as e:
            st.warning(
                f"Request timed out after {timeout_seconds}s (Attempt {i+1}/{retries}): {e}")
            logging.warning(f"Request timed out (Attempt {i+1}/{retries}): {e}")
            if i < retries - 1:
                time.sleep(delay)
                continue
            else:
                st.error(f"Request timed out after {retries} attempts.")
                return f"Error: Request timed out - {e}"

        except requests.exceptions.HTTPError as e:
            error_status = e.response.status_code if e.response is not None else "N/A"
            error_details = f"Status Code: {error_status}"
            response_body_str = "N/A"
            if e.response is not None:
                try:
                    response_body_json = e.response.json()
                    response_body_str = json.dumps(response_body_json)
                    error_details += f", Response: {response_body_str}"
                except json.JSONDecodeError:
                    response_body_str = e.response.text
                    error_details += f", Response Text: {response_body_str[:500]}..."

            logging.error(
                f"HTTP Error (Attempt {i+1}/{retries}): {e}. Details: {error_details}")
            if error_status == 400:
                st.error(
                    f"API Bad Request (Error 400). Likely an issue with the prompt or parameters sent. Details: {response_body_str}")
            elif error_status == 401:
                st.error(
                    f"API Unauthorized (Error 401). Check your Hugging Face API Token (HF_API_TOKEN) in secrets.")
            elif error_status == 429:
                st.error(
                    f"API Rate Limit Hit (Error 429). Please wait before trying again.")
            else:
                st.error(
                    f"HTTP Error calling Hugging Face API (Attempt {i+1}/{retries}): {e}. {error_details}")

            should_retry = (i < retries - 1 and e.response is not None and
                           (500 <= e.response.status_code < 600 or e.response.status_code == 429))  # Retry 5xx and 429
            if should_retry:
                st.warning(
                    f"Server error ({error_status}) or rate limit hit, retrying in {delay}s...")
                time.sleep(delay)
                continue
            else:
                st.error(
                    f"Failed after {i+1} attempts. Non-retryable HTTP error or max retries reached.")
                # Return details
                return f"Error generating insight: {e}. {error_details}"

        except requests.exceptions.RequestException as e:
            st.error(
                f"Network error calling HF API (Attempt {i+1}/{retries}): {e}")
            logging.error(f"Network error (Attempt {i+1}/{retries}): {e}")
            if i < retries - 1:
                st.warning(f"Retrying in {delay}s...")
                time.sleep(delay)
                continue
            else:
                st.error(f"Failed after {i+1} attempts due to network errors.")
                return f"Error generating insight: Network issue - {e}"
        except json.JSONDecodeError as e:
            # This case should be less likely now with raise_for_status, but good to keep
            st.error(
                f"Failed to decode JSON response from API (Attempt {i+1}/{retries}). Status: {response.status_code if 'response' in locals() else 'N/A'}. Error: {e}")
            logging.error(
                f"JSONDecodeError (Attempt {i+1}/{retries}). Status: {response.status_code if 'response' in locals() else 'N/A'}. Response Text (start): {response_text[:500] if 'response_text' in locals() else 'N/A'}. Error: {e}")
            return f"Error: Failed to parse API response - {e}"
        except Exception as e:
            st.error(
                f"An unexpected error occurred during HF API call (Attempt {i+1}/{retries}): {type(e).__name__} - {e}")
            logging.exception(
                f"Unexpected error during HF API call (Attempt {i+1}/{retries}):")
            if i < retries - 1:
                st.warning(f"Retrying in {delay}s...")
                time.sleep(delay)
                continue
            else:
                return f"Error generating insight: An unexpected error occurred - {e}"

    logging.error("Failed to generate insight after all retries.")
    return "Error: Failed to generate insight after multiple retries."


# --- Prompt Shortening Helper ---
def truncate_text(text: Any, max_length: int) -> str:
    """Safely converts to string and truncates text."""
    if pd.isna(text):
        return "N/A"  # Handle NaN before converting to string
    text_str = str(text)
    if len(text_str) > max_length:
        # Truncate respecting word boundaries if possible near the end
        truncated = text_str[:max_length].rsplit(' ', 1)[0]
        # If rsplit didn't work well (e.g., one long word), just hard truncate
        if len(truncated) < max_length * 0.8:
            return text_str[:max_length - 3] + "..."
        return truncated + "..."
    return text_str


# --- Main Application ---
st.title("🎓 Student Learning & Coding Assessment Dashboard")
st.markdown(
    f"Enhanced with ✨ Gen AI Insights (using Hugging Face - `{MODEL_NAME}`)")

# --- Load Data ---
DATA_FILEPATH = "student_responses.csv"
df = load_data(DATA_FILEPATH)

if df is not None:
    # --- Calculate Aggregate Data ---
    valid_coding_cols: List[str] = [
        q for q in CODING_QUESTION_COLS if q in df.columns]
    if not valid_coding_cols:
        st.warning(
            "Could not find any of the expected coding question columns in the CSV. Score calculation will be skipped.")
        logging.warning(
            f"Expected coding cols not found. Expected: {CODING_QUESTION_COLS}. Found: {df.columns.tolist()}")

    avg_score_pct: Optional[float] = None
    all_scores: List[float] = []

    if valid_coding_cols:
        df['CorrectCount'] = 0
        for index, row in df.iterrows():
            correct_count_row: int = 0
            for q_col in valid_coding_cols:
                student_answer_raw = row.get(q_col)
                correct_answer_str = CORRECT_ANSWERS.get(q_col)
                if student_answer_raw is not None and correct_answer_str is not None:
                    student_answer_norm = normalize_answer(student_answer_raw)
                    correct_answer_norm = normalize_answer(correct_answer_str)
                    if student_answer_norm == correct_answer_norm and student_answer_norm != "":
                        correct_count_row += 1
            # Ensure index exists before setting value (safer for potential future filtering)
            if index in df.index:
                df.loc[index, 'CorrectCount'] = correct_count_row

        df['ScorePercentage'] = (
            df['CorrectCount'] / len(valid_coding_cols)) * 100 if valid_coding_cols else 0
        # Ensure ScorePercentage column exists before accessing
        if 'ScorePercentage' in df.columns:
            all_scores = df['ScorePercentage'].tolist()
            avg_score_pct = df['ScorePercentage'].mean(
            ) if not df['ScorePercentage'].empty else 0.0
            logging.info(
                f"Calculated scores. Average Score: {avg_score_pct:.1f}%")
        else:
            logging.error(
                "Failed to create or calculate 'ScorePercentage' column.")

    confidence_col = 'How confident are you in your programming skills?'
    avg_confidence: Optional[float] = None
    if confidence_col in df.columns:
        confidence_numeric = pd.to_numeric(df[confidence_col], errors='coerce')
        avg_confidence = confidence_numeric.mean(
            skipna=True)  # Explicitly skip NaNs
        if avg_confidence is not None:
            logging.info(f"Calculated average confidence: {avg_confidence:.2f}")
        else:
            logging.info(
                "Confidence column present but average could not be calculated (all values might be non-numeric).")
    else:
        logging.warning(f"Confidence column '{confidence_col}' not found.")

    # --- Student Selection ---
    student_list: List[str] = sorted(
        [email for email in df['Email Address'].unique() if email != 'Unknown' and pd.notna(email)])
    selected_student_email: str = st.selectbox(
        "Select Student Email:", options=[""] + student_list,
        index=0,
        format_func=lambda x: "Select a student..." if x == "" else x,
        key="student_selector"
    )

    # --- Display Area ---
    if selected_student_email:
        student_data_df = df.loc[df['Email Address'] == selected_student_email]

        if student_data_df.empty:
            st.error(
                f"Could not find data for selected student: {selected_student_email}")
            logging.error(
                f"Data not found for selected student: {selected_student_email}")
        else:
            student_data: pd.Series = student_data_df.iloc[0]
            st.header(f"📊 Analysis for: {selected_student_email}")
            st.divider()

            # --- Tabs for Organization ---
            tab1, tab2, tab3, tab4 = st.tabs(
                ["👤 Profile", "💻 Coding Performance", "📝 Exam Feedback", "🤖 AI Insights"])

            # --- Tab 1: Profile ---
            with tab1:
                st.subheader("Student Profile & Preferences")
                col1, col2 = st.columns(2)

                def display_profile_metric(container, label, col_name, suffix=""):
                    value = student_data.get(col_name, 'N/A')
                    if pd.isna(value):
                        value = "N/A"
                    try:
                        num_value = pd.to_numeric(value)
                        if not pd.isna(num_value):
                            container.metric(
                                label=label, value=f"{num_value}{suffix}")
                        else:
                            container.metric(label=label, value="N/A")
                    except (ValueError, TypeError):
                        container.metric(label=label, value=f"{value}{suffix}")

                def display_profile_text(container, label, col_name):
                    value = student_data.get(col_name, 'N/A')
                    if pd.isna(value):
                        value = "N/A"
                    container.write(f"**{label}:** {value}")

                with col1:
                    display_profile_metric(
                        st, "Self-Rated Programming Confidence (1-5)", 'How confident are you in your programming skills?', '/5')
                    display_profile_text(
                        st, "Coding Practice Frequency", 'How often do you practice coding?')
                    display_profile_text(
                        st, "Most Comfortable Language(s)", 'Which programming language are you most comfortable with?')
                    display_profile_text(
                        st, "Preferred Learning Style", 'What is your preferred way of learning technical concepts?')
                with col2:
                    display_profile_metric(
                        st, "DSA Understanding (1-5)", 'How would you rate your understanding of Data Structures and Algorithms?', '/5')
                    display_profile_text(
                        st, "Participated in Contests/Hackathons", 'Have you ever participated in coding contests or hackathons?')
                    display_profile_text(
                        st, "Help Seeking Frequency", 'How often do you seek help when stuck on a problem?')
                    display_profile_text(
                        st, "Preferred Assessment Difficulty", 'Do you prefer assessments with fixed difficulty or adaptive difficulty?')
                st.divider()
                st.write(f"**Learning Goals:**")
                st.caption(
                    f"{student_data.get('What are your learning goals for this semester?', 'N/A')}")
                st.write(f"**Common Challenges:**")
                st.caption(
                    f"{student_data.get('What challenges do you usually face while learning coding concepts?', 'N/A')}")

            # --- Tab 2: Coding Performance ---
            with tab2:
                # Tab specific variables need to be recalculated or retrieved if not global
                correct_count: int = 0
                score: float = 0.0
                total_valid_questions: int = len(
                    valid_coding_cols) if valid_coding_cols else 0
                # Need to regen this list for the specific student
                incorrect_answers_details: List[Dict[str, str]] = []

                if not valid_coding_cols:
                    st.info(
                        "No valid coding question columns found in the data. Cannot display performance details.")
                else:
                    # Recalculate details for this student
                    for q_col in valid_coding_cols:
                        student_answer_raw = student_data.get(q_col)
                        correct_answer_str = CORRECT_ANSWERS.get(q_col)
                        if student_answer_raw is not None and correct_answer_str is not None:
                            student_answer_norm = normalize_answer(
                                student_answer_raw)
                            correct_answer_norm = normalize_answer(
                                correct_answer_str)
                            is_correct = (
                                student_answer_norm == correct_answer_norm and student_answer_norm != "")
                            if is_correct:
                                correct_count += 1
                            else:
                                incorrect_answers_details.append({
                                    "question": q_col.splitlines()[0],
                                    "student_answer": str(student_answer_raw),
                                    "correct_answer": correct_answer_str
                                })
                        else:
                            # Handle case where student didn't answer or correct answer missing for some reason
                            incorrect_answers_details.append({
                                "question": q_col.splitlines()[0],
                                "student_answer": str(student_answer_raw) if student_answer_raw is not None else "N/A",
                                "correct_answer": correct_answer_str if correct_answer_str is not None else "N/A"
                            })

                    score = (correct_count / total_valid_questions) * \
                        100 if total_valid_questions > 0 else 0.0

                    st.subheader("Coding Question Performance")
                    delta_score_str: Optional[str] = None
                    if avg_score_pct is not None:
                        delta_val = score - avg_score_pct
                        delta_score_str = f"{delta_val:.1f}% vs Avg"
                    st.metric(label="Coding Questions Score", value=f"{correct_count}/{total_valid_questions} ({score:.1f}%)",
                             delta=delta_score_str, delta_color="normal")
                    st.progress(score / 100)
                    st.divider()

                    # Chart Data Prep
                    perf_data: Dict[str, List] = {
                        'Question': [], 'Status': [], 'Student Answer': [], 'Correct Answer': []}
                    for i, q_col in enumerate(valid_coding_cols):
                        question_text_short = f"Q{i+1}"
                        student_answer_raw = student_data.get(q_col, "N/A")
                        correct_answer_str = CORRECT_ANSWERS.get(q_col, "N/A")
                        student_answer_norm = normalize_answer(
                            student_answer_raw)
                        correct_answer_norm = normalize_answer(
                            correct_answer_str)
                        is_correct = (
                            student_answer_norm == correct_answer_norm and student_answer_norm != "")
                        status_symbol = "✅" if is_correct else "❌"
                        perf_data['Question'].append(question_text_short)
                        perf_data['Status'].append(status_symbol)
                        perf_data['Student Answer'].append(
                            str(student_answer_raw))
                        perf_data['Correct Answer'].append(correct_answer_str)

                    if perf_data['Question']:
                        perf_df = pd.DataFrame(perf_data)
                        perf_df['Count'] = 1
                        fig_perf = px.bar(perf_df, x='Question', y='Count', color='Status', color_discrete_map={'✅': 'green', '❌': 'red'}, title="Correctness per Coding Question",
                                         labels={'Status': 'Result', 'Count': ''}, hover_data=['Student Answer', 'Correct Answer'], custom_data=['Student Answer', 'Correct Answer'])
                        fig_perf.update_layout(
                            yaxis_title=None, yaxis_showticklabels=False)
                        fig_perf.update_traces(
                            hovertemplate="<b>%{x}</b><br>Result: %{color}<br>Your Answer: %{customdata[0]}<br>Correct Answer: %{customdata[1]}<extra></extra>")
                        st.plotly_chart(fig_perf, use_container_width=True)
                    else:
                        st.info("Could not generate performance chart data.")
                    st.divider()

                    with st.expander("Show/Hide Individual Question Details", expanded=False):
                        for i, q_col in enumerate(valid_coding_cols):
                            # Duplicating logic from chart prep slightly for clarity here
                            question_text_short = f"Q{i+1}"
                            student_answer_raw = student_data.get(q_col, "N/A")
                            correct_answer_str = CORRECT_ANSWERS.get(
                                q_col, "N/A")
                            student_answer_norm = normalize_answer(
                                student_answer_raw)
                            correct_answer_norm = normalize_answer(
                                correct_answer_str)
                            is_correct = (
                                student_answer_norm == correct_answer_norm and student_answer_norm != "")
                            status_emoji = "✅ Correct" if is_correct else "❌ Incorrect"
                            st.markdown(
                                f"**{question_text_short}:** `{q_col.splitlines()[0]}...`")
                            st.markdown(
                                f"    *Your Answer:* `{student_answer_raw}`")
                            st.markdown(
                                f"    *Correct Answer:* `{correct_answer_str}`")
                            st.markdown(f"    *Result:* **{status_emoji}**")
                            if i < len(valid_coding_cols) - 1:
                                st.markdown("---")

            # --- Tab 3: Exam Feedback ---
            with tab3:
                st.subheader("Student Feedback on the Assessment")
                feedback_cols: Dict[str, str] = {"Confidence Before": "How confident were you before the exam? (1 = Not Confident at All, 5 = Very Confident)", "Confidence During": "How confident were you during the exam? (1 = Not Confident at All, 5 = Very Confident)", "Clarity": "How well did you understand the questions provided for the exam? (1 = Not Clear at All, 5 = Very Clear)", "Challenge": "How challenging did you find the exam overall? (1 = Very Easy, 5 = Very Challenging)",
                                               "Relevance": "How relevant were the questions to the topics you studied? (1 = Not Relevant at All, 5 = Highly Relevant)", "Suitability": "How well did the exam format suit your learning style? (1 = Not Suitable at All, 5 = Very Suitable)"}
                col3, col4 = st.columns(2)

                def display_feedback_metric(container: st.delta_generator.DeltaGenerator, display_name: str, col_name: str):
                    value = student_data.get(col_name)
                    metric_val = "N/A"
                    label = f"{display_name}"
                    if value is not None and pd.notna(value):
                        try:
                            value_num = pd.to_numeric(value)
                            metric_val = f"{int(value_num)}/5" if value_num.is_integer(
                            ) else f"{value_num:.1f}/5"
                            label = f"{display_name} (1-5)"
                        except (ValueError, TypeError):
                            # Display as text if not clearly numeric
                            metric_val = str(value)
                    container.metric(label=label, value=metric_val)

                feedback_items = list(feedback_cols.items())
                midpoint = (len(feedback_items) + 1) // 2
                for i, (disp_name, col_name) in enumerate(feedback_items):
                    target_col = col3 if i < midpoint else col4
                    if col_name in student_data.index:
                        display_feedback_metric(target_col, disp_name, col_name)
                    else:
                        target_col.metric(label=disp_name,
                                        value="N/A (Column Missing)")
                        logging.warning(
                            f"Feedback column missing for student {selected_student_email}: {col_name}")
                st.divider()

                st.subheader("Perceived Question Easiness")
                easiness_cols: List[str] = [
                    col for col in df.columns if "How easy was the" in col and col in student_data.index]
                easiness_data: Dict[str, List] = {
                    'Question': [], 'Easiness Rating (1-5)': []}
                q_num_easiness: int = 1
                if not easiness_cols:
                    st.info(
                        "No 'How easy was the...' columns found in the data for this student.")
                else:
                    for col in easiness_cols:
                        q_label = f"Q{q_num_easiness}"
                        q_num_easiness += 1
                        rating = student_data.get(col)
                        try:
                            rating_num = pd.to_numeric(rating, errors='coerce')
                            if not pd.isna(rating_num):
                                easiness_data['Question'].append(q_label)
                                easiness_data['Easiness Rating (1-5)'].append(
                                    rating_num)
                        except Exception as e:
                            logging.warning(
                                f"Could not process easiness rating for col '{col}': {e}")
                            continue
                    if easiness_data['Question']:
                        easiness_df = pd.DataFrame(easiness_data)
                        fig_easiness = px.bar(easiness_df, x='Question', y='Easiness Rating (1-5)', title="Student's Perceived Easiness per Question",
                                            labels={'Easiness Rating (1-5)': 'Easiness Rating (1=Difficult, 5=Easy)'}, range_y=[0, 5.5])
                        fig_easiness.update_layout(
                            yaxis_title="Easiness Rating (1=Difficult, 5=Easy)")
                        st.plotly_chart(fig_easiness, use_container_width=True)
                    else:
                        st.info(
                            "No valid question easiness rating data found for this student.")

            # --- Tab 4: AI Insights ---
            with tab4:
                st.subheader(
                    f"Gen AI Insights (via Hugging Face - `{MODEL_NAME}`)")
                st.caption(
                    f"Powered by `{MODEL_NAME}`. Results generated by AI, please review for accuracy. Free tier usage may have limits or delays.")
                st.divider()

                # Regenerate incorrect_answers_details if not already done in Tab 2 scope
                if 'incorrect_answers_details' not in locals():  # Check if variable exists from Tab 2
                    incorrect_answers_details = []
                    if valid_coding_cols:
                        for q_col in valid_coding_cols:
                            student_answer_raw = student_data.get(q_col)
                            correct_answer_str = CORRECT_ANSWERS.get(q_col)
                            if student_answer_raw is not None and correct_answer_str is not None:
                                student_answer_norm = normalize_answer(
                                    student_answer_raw)
                                correct_answer_norm = normalize_answer(
                                    correct_answer_str)
                                if not (student_answer_norm == correct_answer_norm and student_answer_norm != ""):
                                    incorrect_answers_details.append({
                                        "question": q_col.splitlines()[0],
                                        "student_answer": str(student_answer_raw),
                                        "correct_answer": correct_answer_str
                                    })

                # --- Generate Performance Summary ---
                if st.button("Generate Performance Summary", key="ai_summary_btn", help="Generates a brief AI summary based on profile and coding score."):
                    with st.spinner(f"🧠 Calling `{MODEL_NAME}`... Generating summary... Please wait."):
                        profile_confidence = student_data.get(
                            'How confident are you in your programming skills?', 'N/A')
                        profile_dsa = student_data.get(
                            'How would you rate your understanding of Data Structures and Algorithms?', 'N/A')
                        profile_practice = student_data.get(
                            'How often do you practice coding?', 'N/A')
                        profile_lang = student_data.get(
                            'Which programming language are you most comfortable with?', 'N/A')
                        performance_info = f"Coding Score: {correct_count}/{total_valid_questions} ({score:.1f}%)" if valid_coding_cols else "Coding score not available."

                        summary_prompt = (
                            f"Analyze the following student data and provide a very brief (2-3 sentences) summary of their programming profile and recent coding performance.\n\n"
                            f"Student Profile Hints:\n"
                            f"- Self-Rated Programming Confidence (1-5): {profile_confidence}\n"
                            f"- Self-Rated DSA Understanding (1-5): {profile_dsa}\n"
                            f"- Coding Practice Frequency: {truncate_text(profile_practice, 50)}\n"
                            f"- Comfortable Language(s): {truncate_text(profile_lang, 50)}\n\n"
                            f"Performance:\n{performance_info}\n\nBrief Summary:"
                        )
                        logging.info("Generating AI Performance Summary...")
                        summary = generate_hf_insight(
                            summary_prompt, max_new_tokens=150, temperature=0.6)
                    st.markdown("**AI Performance Summary:**")
                    if summary.startswith("Error:"):
                        st.error(summary)
                    else:
                        st.success(f"✅ Summary Generated:\n\n{summary}")
                    st.divider()

                # --- Generate Personalized Feedback on Mistakes ---
                if not valid_coding_cols:
                    st.info(
                        "Cannot generate feedback as coding questions were not found.")
                elif incorrect_answers_details:
                    if st.button("Generate Personalized Feedback on Mistakes", key="ai_feedback_btn", help="Generates AI feedback focusing only on the questions answered incorrectly."):
                        with st.spinner(f"🧠 Calling `{MODEL_NAME}`... Analyzing mistakes... Please wait."):
                            feedback_prompt = (
                                f"Act as a helpful programming tutor... Provide brief, constructive hints/explanations for each mistake... Base feedback *only* on these mistakes:\n\n")
                            mistake_count = 0
                            for item in incorrect_answers_details:
                                if mistake_count < 5:  # Limit prompt length
                                    feedback_prompt += (
                                        f"- Question: {truncate_text(item['question'], 80)}\n  Student Answer: `{truncate_text(item['student_answer'], 50)}`\n  Correct Answer: `{truncate_text(item['correct_answer'], 50)}`\n\n")
                                    mistake_count += 1
                                else:
                                    feedback_prompt += "- ... (additional mistakes omitted)\n\n"
                                    break
                            feedback_prompt += "Provide feedback:"
                            logging.info(
                                f"Generating AI Feedback on {mistake_count} mistakes...")
                            feedback = generate_hf_insight(
                                feedback_prompt, max_new_tokens=MODEL_MAX_NEW_TOKENS_LIMIT, temperature=0.7)
                        st.markdown("**AI Personalized Feedback on Mistakes:**")
                        if feedback.startswith("Error:"):
                            st.error(feedback)
                        else:
                            st.info(f"💡 Feedback Generated:\n\n{feedback}")
                        st.divider()
                else:
                    st.info(
                        "✅ No incorrect answers found to generate specific feedback for.")
                    st.divider()

                # --- Generate Learning Suggestions ---
                if st.button("Suggest Learning Path", key="ai_suggest_btn", help="Generates AI-driven learning suggestions based on profile, goals, challenges, and performance."):
                    with st.spinner(f"🧠 Calling `{MODEL_NAME}`... Generating suggestions... Please wait."):
                        goals = student_data.get(
                            'What are your learning goals for this semester?', 'Not specified')
                        challenges = student_data.get(
                            'What challenges do you usually face while learning coding concepts?', 'Not specified')
                        preferred_style = student_data.get(
                            'What is your preferred way of learning technical concepts?', 'Not specified')
                        confidence = student_data.get(
                            'How confident are you in your programming skills?', 'N/A')
                        dsa_understanding = student_data.get(
                            'How would you rate your understanding of Data Structures and Algorithms?', 'N/A')
                        incorrect_questions_summary = "None"
                        if incorrect_answers_details:
                            q_topics = [f"Concept related to '{item['question']}'" for item in incorrect_answers_details[:3]]
                            incorrect_questions_summary = "; ".join(q_topics)
                            incorrect_questions_summary += "; ..." if len(
                                incorrect_answers_details) > 3 else ""

                        suggestions_prompt = (
                            f"Based *only* on the student data below, suggest 2-3 specific and actionable learning steps or resource types...\n\n"
                            f"Student Data:\n- Learning Goals: {truncate_text(goals, 100)}\n- Stated Challenges: {truncate_text(challenges, 100)}\n- Preferred Learning Style: {truncate_text(preferred_style, 50)}\n- Confidence (1-5): {confidence}\n- DSA Understanding (1-5): {dsa_understanding}\n- Recent Quiz Difficulty Areas: {truncate_text(incorrect_questions_summary, 150)}\n- Overall Score: {score:.1f}%" if valid_coding_cols else "(Score N/A)"
                            f"\n\nActionable Learning Suggestions:")
                        logging.info("Generating AI Learning Suggestions...")
                        suggestions = generate_hf_insight(
                            suggestions_prompt, max_new_tokens=MODEL_MAX_NEW_TOKENS_LIMIT, temperature=0.75)
                    st.markdown("**AI Learning Suggestions:**")
                    if suggestions.startswith("Error:"):
                        st.error(suggestions)
                    else:
                        st.warning(f"💡 Suggestions Generated:\n\n{suggestions}")

    # --- Default View (No Student Selected) ---
    elif selected_student_email == "":
        st.info(
            "👈 Select a student from the dropdown menu above to view their assessment details and AI insights.")
        st.divider()
        st.subheader("Overall Class Summary")
        col_agg1, col_agg2, col_agg3 = st.columns(3)
        with col_agg1:
            st.metric("Total Responses", df.shape[0])
        with col_agg2:
            st.metric("Avg. Coding Score",
                     f"{avg_score_pct:.1f}%" if avg_score_pct is not None else "N/A")
        with col_agg3:
            st.metric("Avg. Programming Confidence",
                     f"{avg_confidence:.2f}/5" if avg_confidence is not None else "N/A")
        st.divider()
        learn_style_col = 'What is your preferred way of learning technical concepts?'
        if learn_style_col in df.columns:
            st.write("**Distribution of Preferred Learning Styles:**")
            try:
                learn_style_counts = df[learn_style_col].dropna(
                ).value_counts().reset_index()
                learn_style_counts.columns = ['Learning Style', 'Count']
                if not learn_style_counts.empty:
                    fig_learn_style = px.bar(
                        learn_style_counts, x='Learning Style', y='Count', title="Class Preferred Learning Styles")
                    st.plotly_chart(fig_learn_style, use_container_width=True)
                else:
                    st.info("No valid data found for preferred learning styles.")
            except Exception as e:
                st.error(f"Could not generate learning style chart: {e}")
                logging.exception("Error generating learning style chart:")
        else:
            st.write(
                f"**Preferred Learning Styles:** N/A ('{learn_style_col}' Column Missing)")

# --- Error Message if Data Loading Failed ---
else:
    st.error(
        "Dashboard cannot be displayed because the student data failed to load. Please check the file path, format, and content ('student_responses.csv'). See console/log for details.")
    logging.error("Data loading failed, dashboard cannot be displayed.")

# Add a final log message
logging.info("Streamlit app execution finished.")