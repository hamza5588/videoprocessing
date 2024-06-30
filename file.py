import streamlit as st
import os
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from google.generativeai import configure, upload_file, get_file, GenerativeModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Configuration
apikey = "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"
google_apikey = "AIzaSyDahZhLwTabewAMPZnBvKh-FMn_uA-yg9k"
configure(api_key=google_apikey)
google_api_key = os.environ.get('GOOGLE_API_KEY')
FAISS_INDEX_PATH = "faiss_index.faiss"
TRANSCRIPTION_PATH = "transcription.txt"
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4'}

# Initialize ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=2)

# Helper functions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

async def upload_and_process_video(video_file):
    CHUNK_SIZE_SECONDS = 60  # Process video in chunks of 5 minutes (adjust as needed)

    def upload_to_gemini(path, mime_type=None):
        file = upload_file(path, mime_type=mime_type)
        return file

    def wait_for_files_active(files):
        for name in (file.name for file in files):
            file = get_file(name)
            while file.state.name == "PROCESSING":
                time.sleep(10)
                file = get_file(name)
            if file.state.name != "ACTIVE":
                raise Exception(f"File {file.name} failed to process")

    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    # Initialize an empty transcription
    transcription = ""

    # Process video in chunks
    with open(video_file, "rb") as f:
        chunk = f.read(CHUNK_SIZE_SECONDS * 1024 * 1024)  # Read 5 minutes of video data (adjust size as needed)
        while chunk:
            temp_filename = f"{video_file}_{time.time()}"
            with open(temp_filename, "wb") as temp_f:
                temp_f.write(chunk)

            # Upload chunk to Gemini for processing
            files = [upload_to_gemini(temp_filename, mime_type="video/mp4")]
            wait_for_files_active(files)

            # Start chat session with the processed chunk
            chat_session = model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": [
                            files[0],
                        ],
                    },
                ]
            )

            # Get transcription from the chat session
            response = chat_session.send_message("please transcribe the video")
            transcription += response.text + " "

            # Delete temporary chunk file
            os.remove(temp_filename)

            # Read next chunk
            chunk = f.read(CHUNK_SIZE_SECONDS * 1024 * 1024)

    return transcription.strip()

def create_and_save_faiss_index(transcription):
    with open(TRANSCRIPTION_PATH, "w") as f:
        f.write(transcription)

    loader = TextLoader(TRANSCRIPTION_PATH)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(pages)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector = FAISS.from_documents(documents, embeddings)
    vector.save_local(FAISS_INDEX_PATH)

def load_faiss_index():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    return vector

# Streamlit app
st.title("Video Transcription and Question Answering")

# Upload Video Section
st.header("Upload Video")
uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])
if uploaded_file is not None:
    if allowed_file(uploaded_file.name):
        video_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Video uploaded successfully!")

        # Process the video asynchronously
        transcription = asyncio.run(upload_and_process_video(video_path))
        create_and_save_faiss_index(transcription)

        st.success("Video processed and FAISS index created.")

# Ask Question Section
st.header("Ask a Question")
question = st.text_input("Enter your question:")
if st.button("Submit Question"):
    if question:
        vector = load_faiss_index()
        prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context. Additionally, remember the question I asked, one timestamp back so that when I ask you what I asked, you can tell me. Make sure the answer is well-crafted:

        <context>
        {context}
        </context>

        Question: {input}""")

        model = ChatGoogleGenerativeAI(model="gemini-pro")
        document_chain = create_stuff_documents_chain(model, prompt)
        retriever = vector.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        message_history = ChatMessageHistory()

        agent_with_chat_history = RunnableWithMessageHistory(
            retrieval_chain,
            lambda session_id: message_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        response = agent_with_chat_history.invoke(
            {"input": question},
            config={"configurable": {"session_id": "<foo>"}},
        )
        answer = response["answer"]
        st.write("Question:", question)
        st.write("Answer:", answer)
