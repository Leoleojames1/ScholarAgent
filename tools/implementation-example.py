"""
ScholarAgent with Advanced ArXiv Integration

This module demonstrates how to extend the ScholarAgent with advanced arXiv capabilities
for fetching paper metadata, downloading LaTeX source, and analyzing paper content.

Features:
- Full paper metadata retrieval from arXiv
- LaTeX source code download and extraction
- Content analysis and extraction
- SQLite database for persistent storage and searching

Usage:
    python scholar_agent_with_arxiv.py --download-path ./papers [--db-path ./arxiv_papers.db]
"""

import feedparser
import urllib.parse
import yaml
import gradio as gr
from smolagents import CodeAgent, HfApiModel, load_tool, tool
from tools.final_answer import FinalAnswerTool
import os
from pathlib import Path
import argparse
import sqlite3

# Import the advanced arXiv tool
from advanced_arxiv_tool import AdvancedArxivTool

# Import other required tools
from web_search import DuckDuckGoSearchTool

def main():
    parser = argparse.ArgumentParser(description='ScholarAgent with Advanced ArXiv Integration')
    parser.add_argument('--download-path', default='./papers', help='Path to download papers')
    parser.add_argument('--db-path', default=None, help='Path to SQLite database (optional)')
    
    args = parser.parse_args()
    
    # Create download directory if it doesn't exist
    os.makedirs(args.download_path, exist_ok=True)
    
    # Initialize tools
    advanced_arxiv = AdvancedArxivTool(
        download_path=args.download_path,
        db_path=args.db_path
    )
    
    final_answer = FinalAnswerTool()
    web_search = DuckDuckGoSearchTool(max_results=5)
    
    # Load prompt templates
    with open("prompts.yaml", 'r') as stream:
        prompt_templates = yaml.safe_load(stream)
    
    # Create the AI Model
    model = HfApiModel(
        max_tokens=2096,
        temperature=0.5,
        model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
    )
    
    # Create the AI Agent with the advanced arXiv tool
    agent = CodeAgent(
        model=model,
        tools=[final_answer, web_search, advanced_arxiv],
        max_steps=6,
        verbosity_level=1,
        name="ScholarAgent",
        description="An AI agent that fetches, downloads, and analyzes research papers from arXiv",
        prompt_templates=prompt_templates
    )
    
    # Define UI search function
    def search_papers(user_input):
        results = agent.run(user_input)
        return results
    
    # Create a function to explore the database
    def explore_database():
        if not os.path.exists(advanced_arxiv.db_path):
            return "Database not initialized yet. Search for papers first to populate the database."
        
        try:
            conn = sqlite3.connect(advanced_arxiv.db_path)
            cursor = conn.cursor()
            
            # Get paper count
            cursor.execute("SELECT COUNT(*) FROM papers")
            paper_count = cursor.fetchone()[0]
            
            # Get latex content count
            cursor.execute("SELECT COUNT(*) FROM latex_content")
            latex_count = cursor.fetchone()[0]
            
            # Get 5 most recent papers
            cursor.execute("SELECT arxiv_id, title, published FROM papers ORDER BY download_timestamp DESC LIMIT 5")
            recent_papers = cursor.fetchall()
            
            recent_papers_text = "\n".join([f"- {row[0]}: {row[1]} ({row[2][:10]})" for row in recent_papers])
            
            conn.close()
            
            return f"""## Database Statistics
- Total papers: {paper_count}
- Papers with LaTeX content: {latex_count}

## Recent Papers:
{recent_papers_text}
"""
        except sqlite3.Error as e:
            return f"Error exploring database: {e}"
    
    # Create Gradio UI
    with gr.Blocks() as demo:
        gr.Markdown("# ScholarAgent with Advanced ArXiv Integration")
        
        with gr.Tab("Search Papers"):
            keyword_input = gr.Textbox(
                label="Enter your query about arXiv papers", 
                placeholder="e.g., Get information about the recent paper 2403.17271 on quantum computing"
            )
            output_display = gr.Markdown()
            search_button = gr.Button("Search")
            search_button.click(search_papers, inputs=[keyword_input], outputs=[output_display])
        
        with gr.Tab("Download & Analyze"):
            paper_id = gr.Textbox(
                label="Enter arXiv ID or URL", 
                placeholder="e.g., 2403.17271 or https://arxiv.org/abs/2403.17271"
            )
            analysis_query = gr.Textbox(
                label="What would you like to know about this paper?",
                placeholder="e.g., Extract all equations, Find the main contributions, Summarize methodology"
            )
            analysis_output = gr.Markdown()
            analyze_button = gr.Button("Analyze Paper")
            analyze_button.click(
                lambda id, query: agent.run(f"For the paper {id}, {query}"),
                inputs=[paper_id, analysis_query], 
                outputs=[analysis_output]
            )
        
        with gr.Tab("Database Explorer"):
            db_info = gr.Markdown(value="Click 'Refresh' to view database statistics")
            refresh_button = gr.Button("Refresh")
            refresh_button.click(explore_database, inputs=[], outputs=[db_info])
    
    # Launch the demo
    demo.launch()

if __name__ == "__main__":
    main()
