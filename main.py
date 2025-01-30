"""
This script fetches YouTube video transcripts, embeds them using OpenAI embeddings,
and uploads them to Pinecone for similarity search and retrieval.
"""

import os
import time
import gradio as gr
from getpass import getpass
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv, find_dotenv
from tqdm.auto import tqdm

from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

# Grouping langchain imports together
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent

def get_video_transcript(vid_id):
    """
    Fetch transcript for a given YouTube video ID.
    """
    try:
        transcript_data = YouTubeTranscriptApi.get_transcript(vid_id)
        transcript_text = " ".join([item['text'] for item in transcript_data if 'text' in item])
        return transcript_text
    except Exception as e:
        print(f"Error retrieving transcript for video {vid_id}: {e}")
        return None

# Example usage with a list of video IDs
video_ids = ['2u4ItZerRac', 'I2zF1I60hPg', '8xqSF-uHCUs', 'LtmS-c1pChY', 'sJNxT-I7L6s']
transcripts = {}

# Fetch the transcript for each video
for vid_id in video_ids:
    trans = get_video_transcript(vid_id)
    if trans:
        transcripts[vid_id] = trans
    else:
        print(f"No transcript available for video {vid_id}")

# Display the fetched transcripts
for video_id, transcript in transcripts.items():
    print(f"Transcript for {video_id}:")
    print(transcript[:500])

_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]="Pros&Cons of living in Germany"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# Get API key for OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or getpass("Enter your OpenAI API key: ")
MODEL_NAME = 'text-embedding-ada-002'

# Initialize OpenAI Embeddings
embed = OpenAIEmbeddings(
    model=MODEL_NAME,
    openai_api_key=OPENAI_API_KEY
)

# Get API key for Pinecone
api_key = os.getenv("PINECONE_API_KEY") or getpass("Enter your Pinecone API key: ")

# Configure Pinecone client
pc = Pinecone(api_key=api_key)

# Define serverless specification
spec = ServerlessSpec(cloud="aws", region="us-east-1")

INDEX_NAME = "langchain-retrieval-agent"
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

# Check if index already exists (it shouldn't if this is first time)
if INDEX_NAME not in existing_indexes:
    # If does not exist, create index
    pc.create_index(
        INDEX_NAME,
        dimension=1536,  # Dimensionality of ada 002
        metric='dotproduct',
        spec=spec
    )
    # Wait for index to be initialized
    while not pc.describe_index(INDEX_NAME).status['ready']:
        time.sleep(1)

# Connect to index
index = pc.Index(INDEX_NAME)
time.sleep(1)
# View index stats
index.describe_index_stats()

BATCH_SIZE = 100

# Prepare data for upload to Pinecone
data = [{'title': video_id, 'context': transcript, 'id': video_id} for video_id, transcript in transcripts.items()]

for i in tqdm(range(0, len(data), BATCH_SIZE)):
    # Get end of batch
    i_end = min(len(data), i + BATCH_SIZE)
    batch = data[i:i_end]
    # First get metadata fields for this batch
    metadatas = [{'title': record['title'], 'text': record['context']} for record in batch]
    # Get the list of contexts / documents
    documents = [record['context'] for record in batch]
    # Create document embeddings
    embeds = embed.embed_documents(documents)
    # Get IDs
    ids = [record['id'] for record in batch]
    # Add everything to Pinecone
    index.upsert(vectors=zip(ids, embeds, metadatas))

index.describe_index_stats()

TEXT_FIELD = "text"  # The metadata field that contains our text

# Initialize the vector store object
vectorstore = PineconeVectorStore(
    index, embed.embed_query, TEXT_FIELD
)

query = "what are the pros and cons of living in Germany?"

# Perform similarity search
vectorstore.similarity_search(
    query,  # Our search query
    k=3  # Return 3 most relevant docs
)

# Chat completion LLM
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

# Conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

# Retrieval QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

qa.run(query)

tools = [
    Tool(
        name='Knowledge Base',
        func=qa.run,
        description=(
            'Use this tool when answering general knowledge queries to get '
            'more information about the topic'
        )
    )
]

agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory
)

agent(query)

agent("can you tell me some facts about living in Germany?")

agent("can you summarize these facts in two short sentences")

agent("can you explain what immigrants faces in germany?")


def chatbot_response(user_query):
    """
    Function to get response from the agent with dynamic retrieval.
    """
    try:
        # Perform a similarity search to retrieve the most relevant context
        retrieved_docs = vectorstore.similarity_search(user_query, k=3)

        # Get response from the agent
        response = agent.run(user_query)

        # Debugging: Print retrieved documents (Optional)
        print("Retrieved Docs:", [doc.page_content for doc in retrieved_docs])

        return response
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
iface = gr.Interface(
    fn=chatbot_response,
    inputs=gr.Textbox(lines=2, placeholder="Ask me about living in Germany..."),
    outputs=gr.Textbox(),
    title="AI Chatbot: Pros & Cons of Living in Germany",
    description="This chatbot retrieves information about living in Germany by analyzing YouTube transcripts. Ask any question!",
    theme="compact"
)

# Launch Gradio app
iface.launch(share=True)
