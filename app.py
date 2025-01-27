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
from langchain.agents import Tool, initialize_agent
import gradio as gr


# Load environment variables from a .env file
_ = load_dotenv(find_dotenv())

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or getpass("Enter your OpenAI API key: ")
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or getpass("Enter your Pinecone API key: ")
model_name = 'text-embedding-ada-002'

# Initialize OpenAI Embeddings
embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)

# Configure Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region="us-east-1")

index_name = "langchain-retrieval-agent"
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

# Check if index already exists (it shouldn't if this is first time)
if index_name not in existing_indexes:
    pc.create_index(
        index_name,
        dimension=1536,  # Dimensionality of ada 002
        metric='dotproduct',
        spec=spec
    )
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# Connect to index
index = pc.Index(index_name)
time.sleep(1)

# Function to get video transcript
def get_video_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([item['text'] for item in transcript if 'text' in item])
        return transcript_text
    except Exception as e:
        return f"Error retrieving transcript for video {video_id}: {e}"


# Function to prepare data for Pinecone
def prepare_data(video_ids):
    transcripts = {}
    for video_id in video_ids:
        transcript = get_video_transcript(video_id)
        if transcript:
            transcripts[video_id] = transcript
    return [{'title': video_id, 'context': transcript, 'id': video_id} 
            for video_id, transcript in transcripts.items()]


# Function to upload data to Pinecone
def upload_to_pinecone(data):
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
video_ids = ['2u4ItZerRac', 'I2zF1I60hPg', '8xqSF-uHCUs', 'LtmS-c1pChY', 'sJNxT-I7L6s']
data = prepare_data(video_ids)
upload_to_pinecone(data)

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


# Define Gradio interface functions
def query_knowledge_base(query):
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
    
