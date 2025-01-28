# YouTube Video Transcript Retrieval and Similarity Search

This repository contains a Python script that fetches YouTube video transcripts, embeds them using OpenAI embeddings, and uploads them to Pinecone for similarity search and retrieval. The script also demonstrates how to perform similarity searches and interact with the embeddings using a conversational AI model.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Example Queries](#example-queries)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.6 or higher
- An OpenAI API key
- A Pinecone API key
- The following Python packages:
  - `youtube_transcript_api`
  - `python-dotenv`
  - `tqdm`
  - `langchain`
  - `transformers`
  - `evaluate`

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Install the required packages:

    ```bash
    pip install youtube_transcript_api python-dotenv tqdm langchain transformers evaluate
    ```

3. Set up your environment variables by creating a `.env` file in the root directory of the project with the following content:

    ```env
    OPENAI_API_KEY=<your_openai_api_key>
    PINECONE_API_KEY=<your_pinecone_api_key>
    ```

## Usage

1. Run the script to fetch YouTube video transcripts and upload them to Pinecone:

    ```bash
    python your_script_name.py
    ```

2. The script will perform the following steps:
   - Fetch transcripts for a list of YouTube video IDs.
   - Embed the transcripts using OpenAI embeddings.
   - Upload the embeddings to Pinecone.
   - Perform similarity searches using a query.
   - Interact with the embeddings using a conversational AI model.

## Configuration

### Environment Variables

The script uses environment variables to store API keys. Ensure you have the following variables set in your `.env` file:

- `OPENAI_API_KEY`: Your OpenAI API key
- `PINECONE_API_KEY`: Your Pinecone API key

### Video IDs

You can modify the list of video IDs in the script to fetch transcripts for different videos:

```python
video_ids = ['2u4ItZerRac', 'I2zF1I60hPg', '8xqSF-uHCUs', 'LtmS-c1pChY', 'sJNxT-I7L6s']

