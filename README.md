# AI Chatbot Project

A multi-phase AI chatbot built with Streamlit, LangChain, and Groq API, featuring document-based retrieval augmented generation (RAG).

## Features

- **Phase 1**: Basic chatbot with Groq LLM
- **Phase 2**: Enhanced chatbot with improved prompt templates
- **Phase 3**: RAG-enabled chatbot that can answer questions based on PDF documents

## Technologies Used

- **Streamlit**: Web interface
- **LangChain**: LLM framework
- **Groq API**: Language model provider
- **HuggingFace Embeddings**: Text embeddings for RAG
- **PyPDF**: PDF document processing
- **ChromaDB**: Vector database (via LangChain)

## Setup Instructions

### Prerequisites

- Python 3.12+
- Pipenv (for dependency management)
- Groq API key

### Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd AI_Chatbot
   ```

2. Install dependencies:
   ```bash
   pipenv install
   ```

3. Activate the virtual environment:
   ```bash
   pipenv shell
   ```

4. Create a `.env` file in the root directory and add your Groq API key:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. Make sure you have the `Interview Questions.pdf` file in the project root for Phase 3 to work.

### Running the Application

#### Phase 1 - Basic Chatbot
```bash
streamlit run phase1.py
```

#### Phase 2 - Enhanced Chatbot
```bash
streamlit run phase2.py
```

#### Phase 3 - RAG Chatbot
```bash
streamlit run phase3.py
```

The application will be available at `http://localhost:8501`

## Getting a Groq API Key

1. Visit [Groq Console](https://console.groq.com/)
2. Sign up for an account
3. Navigate to API settings
4. Generate a new API key
5. Add it to your `.env` file

## Project Structure

```
AI_Chatbot/
├── .env                    # Environment variables (not tracked)
├── .gitignore             # Git ignore file
├── Interview Questions.pdf # Sample PDF for RAG
├── phase1.py              # Basic chatbot
├── phase2.py              # Enhanced chatbot
├── phase3.py              # RAG-enabled chatbot
├── Pipfile                # Python dependencies
├── Pipfile.lock           # Locked dependencies
└── README.md              # This file
```

## Usage

1. Choose which phase to run based on your needs
2. Enter your questions in the chat input
3. For Phase 3, the chatbot can answer questions based on the content of the PDF document

## Notes

- Phase 3 includes special handling for PyTorch/Streamlit conflicts
- The application uses CPU-only processing to avoid CUDA issues
- All phases include proper error handling and user-friendly messages

## Contributing

Feel free to fork this project and submit pull requests for improvements.

## License

This project is open source and available under the MIT License.
