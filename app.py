"""
This script uses Gradio to create an interface for querying a knowledge base about the pros and cons
of living in Germany based on YouTube video transcripts.
"""

import os
import time
from getpass import getpass
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv, find_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
import gradio as gr

# Load environment variables from a .env file
_ = load_dotenv(find_dotenv())

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or getpass("Enter your OpenAI API key: ")
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or getpass("Enter your Pinecone API key: ")
MODEL_NAME = 'text-embedding-ada-002'

# Initialize OpenAI Embeddings
embed = OpenAIEmbeddings(
    model=MODEL_NAME,
    openai_api_key=OPENAI_API_KEY
)

# Configure Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region="us-east-1")

INDEX_NAME = "langchain-retrieval-agent"
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

# Check if index already exists (it shouldn't if this is first time)
if INDEX_NAME not in existing_indexes:
    pc.create_index(
        INDEX_NAME,
        dimension=1536,  # Dimensionality of ada 002
        metric='dotproduct',
        spec=spec
    )
    while not pc.describe_index(INDEX_NAME).status['ready']:
        time.sleep(1)

# Connect to index
index = pc.Index(INDEX_NAME)
time.sleep(1)

def get_video_transcript(video_id):
    """
    Fetch the transcript for a given YouTube video ID.
    """
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([item['text'] for item in transcript if 'text' in item])
        return transcript_text
    except YouTubeTranscriptApi.CouldNotRetrieveTranscript as e:
        return f"Error retrieving transcript for video {video_id}: {e}"

def prepare_data(video_ids):
    """
    Prepare data for Pinecone by fetching transcripts for given video IDs.
    """
    transcripts = {}
    for video_id in video_ids:
        transcript = get_video_transcript(video_id)
        if transcript:
            transcripts[video_id] = transcript
    return [{'title': video_id, 'context': transcript, 'id': video_id} for video_id, transcript in transcripts.items()]

def upload_to_pinecone(data):
    """
    Upload data to Pinecone in batches.
    """
    batch_size = 100
    for i in tqdm(range(0, len(data), batch_size)):
        i_end = min(len(data), i + batch_size)
        batch = data[i:i_end]
        metadatas = [{'title': record['title'], 'text': record['context']} for record in batch]
        documents = [record['context'] for record in batch]
        embeds = embed.embed_documents(documents)
        ids = [record['id'] for record in batch]
        index.upsert(vectors=zip(ids, embeds, metadatas))

# Prepare and upload data
VIDEO_IDS = ['2u4ItZerRac', 'I2zF1I60hPg', '8xqSF-uHCUs', 'LtmS-c1pChY', 'sJNxT-I7L6s']
DATA = prepare_data(VIDEO_IDS)
upload_to_pinecone(DATA)

# Initialize vector store
vectorstore = LangchainPinecone(
    index, embed.embed_query, "text"
)

# Initialize ChatOpenAI and RetrievalQA
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

def query_knowledge_base(query):
    """
    Query the knowledge base using the provided query.
    """
    return qa.run(query)

# Create Gradio interface
iface = gr.Interface(
    fn=query_knowledge_base,
    inputs="text",
    outputs="text",
    title="Knowledge Base Query",
    description="Ask questions about the video transcripts."
)

# Launch Gradio interface
if __name__ == "__main__":
    iface.launch()
