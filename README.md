# AI Chatbot for Pros & Cons of Living in Germany


## Table of Contents

- [Introduction]
- [Features]
- [ProjectStructure]
- [Installation]
- [HowItWorks]
- [UsingChatbot]
- [ExamplesQueries]
- [TechnologiesUsed]
- [License]
- [Contributing]

### 🌟 Introduction

This project automates the extraction of YouTube video transcripts, processes them using OpenAI’s embedding model, and stores them in a Pinecone vector database for efficient similarity search. It also includes an interactive chatbot powered by GPT-3.5 and a Gradio web interface, allowing users to query the knowledge base about the pros and cons of living in Germany.

### 🚀 Features

✅ Automatic YouTube Transcript Extraction – Retrieves subtitles from YouTube videos.
✅ AI-Powered Embeddings – Uses OpenAI’s text-embedding-ada-002 to convert transcripts into vectors.
✅ Pinecone Vector Search – Enables efficient document retrieval based on user queries.
✅ Conversational AI Chatbot – Provides relevant answers using OpenAI’s GPT-3.5.
✅ Gradio Web Interface – A simple and user-friendly UI for asking questions.
    ```

### 📂ProjectStructure

📁 project-root/
│── app.py                 # Main script for processing transcripts and chatbot  
│── requirements.txt       # List of dependencies  
│── .env                   # API keys and environment variables  
│── README.md              # Documentation (this file)  

### 🛠Installation

1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-repo/project.git
cd project
2️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Set Up API Keys
Create a .env file in the project root directory and add the following API keys:

makefile
Copy
Edit
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
4️⃣ Run the Application
bash
Copy
Edit
python app.py
Once the script runs, the Gradio web interface will open in your browser.

### 🔧 HowItWorks

1️⃣ Fetches Transcripts – Uses youtube_transcript_api to get subtitles for specified videos.
2️⃣ Embeds the Text – Converts transcript data into vector representations using OpenAI embeddings.
3️⃣ Stores in Pinecone – Uploads the embeddings to a Pinecone index for fast similarity search.
4️⃣ Retrieves Answers – Uses a RetrievalQA model to fetch relevant context for user queries.
5️⃣ Provides Chat Interface – Allows users to interact with the chatbot via Gradio.

### 🖥️ Using the Chatbot

Run the chatbot using python app.py.
A Gradio web UI will open.
Enter a query like:
"What are the advantages of living in Germany?"
"What challenges do immigrants face in Germany?"
The chatbot retrieves relevant answers based on the stored transcripts.

### 🎯 ExampleQueries

"Tell me about life in Germany."
"Can you summarize the key challenges immigrants face?"
"What are the benefits of moving to Germany?"

### 🔗 TechnologiesUsed


Technology	Purpose
Python 🐍	Core programming language
OpenAI GPT-3.5 🤖	Conversational AI for chatbot responses
OpenAI Embeddings 🔢	Converts text into vector representations
Pinecone 🔍	Stores embeddings for fast similarity search
YouTube Transcript API 🎥	Retrieves video subtitles
Gradio 🌐	Provides a user-friendly chatbot interface

### 📜License

This project is open-source and licensed under the MIT License.#

### ✨ Contributing

Contributions are welcome! Fork the repo and submit a pull request.
Report issues and suggest features via GitHub Issues.
