# yt_transcript_agent.py

from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv, find_dotenv
from getpass import getpass
from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent
import os
import time

def get_video_transcript(video_id):
    """
    Fetch transcript for a given YouTube video ID.
    """
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([item['text'] for item in transcript if 'text' in item])
        return transcript_text
    except Exception as e:
        print(f"Error retrieving transcript for video {video_id}: {e}")
        return None

# Example usage with a list of video IDs
video_ids = ['2u4ItZerRac', 'I2zF1I60hPg', '8xqSF-uHCUs', 'LtmS-c1pChY', 'sJNxT-I7L6s']
transcripts = {}

# Fetch the transcript for each video
for video_id in video_ids:
    transcript = get_video_transcript(video_id)
    if transcript:
        transcripts[video_id] = transcript
    else:
        print(f"No transcript available for video {video_id}")

# Display the fetched transcripts
for video_id, transcript in transcripts.items():
    print(f"Transcript for {video_id}:")
    print(transcript[:500])

_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Get API key for OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or getpass("Enter your OpenAI API key: ")
model_name = 'text-embedding-ada-002'

# Initialize OpenAI Embeddings
embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)

# Get API key for Pinecone
api_key = os.getenv("PINECONE_API_KEY") or getpass("Enter your Pinecone API key: ")

# Configure Pinecone client
pc = Pinecone(api_key=api_key)

# Define serverless specification
spec = ServerlessSpec(cloud="aws", region="us-east-1")

index_name = "langchain-retrieval-agent"
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

# Check if index already exists (it shouldn't if this is first time)
if index_name not in existing_indexes:
    # If does not exist, create index
    pc.create_index(
        index_name,
        dimension=1536,  # Dimensionality of ada 002
        metric='dotproduct',
        spec=spec
    )
    # Wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# Connect to index
index = pc.Index(index_name)
time.sleep(1)
# View index stats
index.describe_index_stats()

batch_size = 100

# Prepare data for upload to Pinecone
data = [{'title': video_id, 'context': transcript, 'id': video_id} for video_id, transcript in transcripts.items()]

for i in tqdm(range(0, len(data), batch_size)):
    # Get end of batch
    i_end = min(len(data), i + batch_size)
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

text_field = "text"  # The metadata field that contains our text

# Initialize the vector store object
vectorstore = Pinecone(
    index, embed.embed_query, text_field
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

agent("what are the pros and cons of living in Germany?")

agent("what is the topic of the videos?")

agent("can you tell me some facts about living in Germany?")

agent("can you summarize these facts in two short sentences")