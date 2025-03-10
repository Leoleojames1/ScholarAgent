# # import feedparser
# # import urllib.parse
# # import yaml
# # import gradio as gr
# # from smolagents import CodeAgent, HfApiModel, tool
# # from tools.final_answer import FinalAnswerTool

# # @tool
# # def fetch_latest_arxiv_papers(keywords: list, num_results: int = 3) -> list:
# #     """Fetches the latest research papers from arXiv based on provided keywords.

# #     Args:
# #         keywords: A list of keywords to search for relevant papers.
# #         num_results: The number of papers to fetch (default is 3).

# #     Returns:
# #         A list of dictionaries containing:
# #             - "title": The title of the research paper.
# #             - "authors": The authors of the paper.
# #             - "year": The publication year.
# #             - "abstract": A summary of the research paper.
# #             - "link": A direct link to the paper on arXiv.
# #     """
# #     try:
# #         print(f"DEBUG: Searching arXiv papers with keywords: {keywords}")  # Debug input
        
# #         #Properly format query with +AND+ for multiple keywords
# #         query = "+AND+".join([f"all:{kw}" for kw in keywords])  
# #         query_encoded = urllib.parse.quote(query)  # Encode spaces and special characters
        
# #         url = f"http://export.arxiv.org/api/query?search_query={query_encoded}&start=0&max_results={num_results}&sortBy=submittedDate&sortOrder=descending"
        
# #         print(f"DEBUG: Query URL - {url}")  # Debug URL
        
# #         feed = feedparser.parse(url)

# #         papers = []
# #         for entry in feed.entries:
# #             papers.append({
# #                 "title": entry.title,
# #                 "authors": ", ".join(author.name for author in entry.authors),
# #                 "year": entry.published[:4],  # Extract year
# #                 "abstract": entry.summary,
# #                 "link": entry.link
# #             })

# #         return papers

# #     except Exception as e:
# #         print(f"ERROR: {str(e)}")  # Debug errors
# #         return [f"Error fetching research papers: {str(e)}"]


# #"""------Applied BM25 search for paper retrival------"""
# # from rank_bm25 import BM25Okapi
# # import nltk

# # import os
# # import shutil


# # nltk_data_path = os.path.join(nltk.data.path[0], "tokenizers", "punkt")
# # if os.path.exists(nltk_data_path):
# #     shutil.rmtree(nltk_data_path)  # Remove corrupted version

# # print("Removed old NLTK 'punkt' data. Reinstalling...")

# # # Step 2: Download the correct 'punkt' tokenizer
# # nltk.download("punkt_tab")

# # print("Successfully installed 'punkt'!")


# # @tool  # Register the function properly as a SmolAgents tool
# # def fetch_latest_arxiv_papers(keywords: list, num_results: int = 5) -> list:
# #     """Fetches and ranks arXiv papers using BM25 keyword relevance.

# #     Args:
# #         keywords: List of keywords for search.
# #         num_results: Number of results to return.

# #     Returns:
# #         List of the most relevant papers based on BM25 ranking.
# #     """
# #     try:
# #         print(f"DEBUG: Searching arXiv papers with keywords: {keywords}")

# #         # Use a general keyword search (without `ti:` and `abs:`)
# #         query = "+AND+".join([f"all:{kw}" for kw in keywords])  
# #         query_encoded = urllib.parse.quote(query)
# #         url = f"http://export.arxiv.org/api/query?search_query={query_encoded}&start=0&max_results=50&sortBy=submittedDate&sortOrder=descending"

# #         print(f"DEBUG: Query URL - {url}")

# #         feed = feedparser.parse(url)
# #         papers = []

# #         # Extract papers from arXiv
# #         for entry in feed.entries:
# #             papers.append({
# #                 "title": entry.title,
# #                 "authors": ", ".join(author.name for author in entry.authors),
# #                 "year": entry.published[:4],
# #                 "abstract": entry.summary,
# #                 "link": entry.link
# #             })

# #         if not papers:
# #             return [{"error": "No results found. Try different keywords."}]

# #         # Apply BM25 ranking
# #         tokenized_corpus = [nltk.word_tokenize(paper["title"].lower() + " " + paper["abstract"].lower()) for paper in papers]
# #         bm25 = BM25Okapi(tokenized_corpus)

# #         tokenized_query = nltk.word_tokenize(" ".join(keywords).lower())
# #         scores = bm25.get_scores(tokenized_query)

# #         # Sort papers based on BM25 score
# #         ranked_papers = sorted(zip(papers, scores), key=lambda x: x[1], reverse=True)

# #         # Return the most relevant ones
# #         return [paper[0] for paper in ranked_papers[:num_results]]

# #     except Exception as e:
# #         print(f"ERROR: {str(e)}")
# #         return [{"error": f"Error fetching research papers: {str(e)}"}]


"""------Applied TF-IDF for better semantic search------"""
"""
ScholarAgent with Advanced ArXiv Integration

This app extends the ScholarAgent with powerful capabilities for fetching, 
downloading, and analyzing arXiv papers including their LaTeX source code.
"""

import feedparser
import urllib.parse
import yaml
import os
from tools.final_answer import FinalAnswerTool
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, load_tool, tool
import nltk
import datetime
import requests
import pytz
import argparse
from pathlib import Path

# Import UI components
from Gradio_UI import GradioUI

# Import the Advanced ArXiv Tool
from tools.advanced_arxiv_tool import AdvancedArxivTool

# Set up NLTK resources
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

@tool
def fetch_latest_arxiv_papers(keywords: list, num_results: int = 5) -> list:
    """Fetches and ranks arXiv papers using TF-IDF and Cosine Similarity.

    Args:
        keywords: List of keywords for search.
        num_results: Number of results to return.

    Returns:
        List of the most relevant papers based on TF-IDF ranking.
    """
    try:
        print(f"DEBUG: Searching arXiv papers with keywords: {keywords}")

        # Use a general keyword search
        query = "+AND+".join([f"all:{kw}" for kw in keywords])  
        query_encoded = urllib.parse.quote(query)
        url = f"http://export.arxiv.org/api/query?search_query={query_encoded}&start=0&max_results=50&sortBy=submittedDate&sortOrder=descending"

        print(f"DEBUG: Query URL - {url}")

        feed = feedparser.parse(url)
        papers = []

        # Extract papers from arXiv
        for entry in feed.entries:
            papers.append({
                "title": entry.title,
                "authors": ", ".join(author.name for author in entry.authors),
                "year": entry.published[:4],
                "abstract": entry.summary,
                "link": entry.link
            })

        if not papers:
            return [{"error": "No results found. Try different keywords."}]

        # Prepare TF-IDF Vectorization
        corpus = [paper["title"] + " " + paper["abstract"] for paper in papers]
        vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))  # Remove stopwords
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # Transform Query into TF-IDF Vector
        query_str = " ".join(keywords)
        query_vec = vectorizer.transform([query_str])

        # Compute Cosine Similarity
        similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

        # Sort papers based on similarity score
        ranked_papers = sorted(zip(papers, similarity_scores), key=lambda x: x[1], reverse=True)

        # Return the most relevant papers
        return [paper[0] for paper in ranked_papers[:num_results]]

    except Exception as e:
        print(f"ERROR: {str(e)}")
        return [{"error": f"Error fetching research papers: {str(e)}"}]

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ScholarAgent with Advanced ArXiv capabilities')
    parser.add_argument('--download-path', default='./papers', help='Path to download papers')
    parser.add_argument('--db-path', default=None, help='Path to SQLite database (optional)')
    parser.add_argument('--model-id', default='Qwen/Qwen2.5-Coder-32B-Instruct', 
                       help='HuggingFace model ID to use')
    parser.add_argument('--share', action='store_true', help='Share the Gradio interface')
    
    return parser.parse_args()

def run_basic_ui():
    """Run the original basic UI"""
    # Create Gradio UI
    with gr.Blocks() as demo:
        gr.Markdown("# ScholarAgent")
        keyword_input = gr.Textbox(label="Enter keywords (comma-separated)", 
                                  placeholder="e.g., deep learning, reinforcement learning")
        output_display = gr.Markdown()
        search_button = gr.Button("Search")

        search_button.click(search_papers, inputs=[keyword_input], outputs=[output_display])

        print("DEBUG: Basic Gradio UI is running. Waiting for user input...")

    # Launch Gradio App
    demo.launch()

def run_advanced_ui(agent, args):
    """Run the advanced UI with the full agent"""
    # Create an upload folder for PDFs and other documents
    upload_folder = Path("./uploads")
    upload_folder.mkdir(exist_ok=True)
    
    # Initialize the GradioUI with our agent
    ui = GradioUI(agent=agent, file_upload_folder=str(upload_folder))
    
    # Launch the UI
    print("DEBUG: Advanced Gradio UI with agent integration is running...")
    ui.launch(share=args.share)

def search_papers(user_input):
    """Legacy function for basic UI paper search"""
    keywords = [kw.strip() for kw in user_input.split(",") if kw.strip()]  # Ensure valid keywords
    print(f"DEBUG: Received input keywords - {keywords}")  # Debug user input
    
    if not keywords:
        print("DEBUG: No valid keywords provided.")
        return "Error: Please enter at least one valid keyword."
    
    results = fetch_latest_arxiv_papers(keywords, num_results=3)  # Fetch 3 results
    print(f"DEBUG: Results received - {results}")  # Debug function output

    # Check if the API returned an error
    if isinstance(results, list) and len(results) > 0 and "error" in results[0]:
        return results[0]["error"]  # Return the error message directly

    # Format results only if valid papers exist
    if isinstance(results, list) and results and isinstance(results[0], dict):
        formatted_results = "\n\n".join([
            f"---\n\n"
            f"📌 **Title:** {paper['title']}\n\n"
            f"👨‍🔬 **Authors:** {paper['authors']}\n\n"
            f"📅 **Year:** {paper['year']}\n\n"
            f"📖 **Abstract:** {paper['abstract'][:500]}... *(truncated for readability)*\n\n"
            f"[🔗 Read Full Paper]({paper['link']})\n\n"
            for paper in results
        ])
        return formatted_results

    print("DEBUG: No results found.")
    return "No results found. Try different keywords."

def main():
    """Main entry point for the application"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create download directory if it doesn't exist
    os.makedirs(args.download_path, exist_ok=True)
    
    # Initialize tools
    final_answer = FinalAnswerTool()
    web_search = DuckDuckGoSearchTool(max_results=5)
    
    # Initialize the advanced arXiv tool
    advanced_arxiv = AdvancedArxivTool(
        download_path=args.download_path,
        db_path=args.db_path
    )
    
    # Load prompt templates
    with open("prompts.yaml", 'r') as stream:
        prompt_templates = yaml.safe_load(stream)
    
    # Create the AI Model
    model = HfApiModel(
        max_tokens=2096,
        temperature=0.5,
        model_id=args.model_id,
        custom_role_conversions=None,
    )
    
    # Create the AI Agent with all tools
    agent = CodeAgent(
        model=model,
        tools=[final_answer, fetch_latest_arxiv_papers, web_search, advanced_arxiv, get_current_time_in_timezone],
        max_steps=6,
        verbosity_level=1,
        grammar=None,
        planning_interval=None,
        name="ScholarAgent",
        description="An AI agent that fetches, downloads, and analyzes research papers from arXiv.",
        prompt_templates=prompt_templates
    )
    
    # Run the advanced UI with the full agent
    run_advanced_ui(agent, args)

if __name__ == "__main__":
    main()


# Launch Gradio App
demo.launch()

