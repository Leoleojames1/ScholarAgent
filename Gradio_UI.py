#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Enhanced Gradio UI for ScholarAgent with Advanced ArXiv Integration

This module provides a custom Gradio UI that extends the standard GradioUI
with specialized tabs for advanced arXiv functionalities.
"""
import os
import sqlite3
import json
from pathlib import Path
import gradio as gr
from smolagents.agents import MultiStepAgent
from Gradio_UI import GradioUI, stream_to_gradio


class ScholarAgentUI(GradioUI):
    """
    Enhanced UI for ScholarAgent with specialized tabs for arXiv research
    
    This UI extends the standard GradioUI with tabs for:
    - Basic paper search
    - Advanced paper analysis
    - Database exploration
    - LaTeX extraction
    """
    
    def __init__(self, agent: MultiStepAgent, download_path: str, db_path: str = None, 
                 file_upload_folder: str = None):
        """
        Initialize the ScholarAgent UI
        
        Args:
            agent: The CodeAgent instance with tools
            download_path: Path where papers are downloaded
            db_path: Path to the SQLite database
            file_upload_folder: Folder for general file uploads
        """
        super().__init__(agent=agent, file_upload_folder=file_upload_folder)
        self.download_path = Path(download_path)
        self.db_path = db_path if db_path else str(self.download_path / "arxiv_papers.db")
    
    def search_papers_simple(self, keywords):
        """Simple keyword search for papers"""
        formatted_input = f"Search for recent papers about {keywords}"
        return self.agent.run(formatted_input)
    
    def analyze_paper(self, paper_id, analysis_request):
        """Analyze a specific paper"""
        if not paper_id or not analysis_request:
            return "Please provide both a paper ID/URL and analysis request."
        
        formatted_input = f"For the paper {paper_id}, {analysis_request}"
        return self.agent.run(formatted_input)
    
    def extract_latex(self, paper_id, extract_type):
        """Extract LaTeX content from a paper"""
        if not paper_id:
            return "Please provide a paper ID or URL."
        
        if extract_type == "All LaTeX":
            query = f"Download and extract all LaTeX content from paper {paper_id}"
        elif extract_type == "Equations Only":
            query = f"Extract and explain all mathematical equations from paper {paper_id}"
        elif extract_type == "Algorithms Only":
            query = f"Extract and explain all algorithms from paper {paper_id}"
        else:
            query = f"Download and extract the most important parts from the LaTeX source of paper {paper_id}"
        
        return self.agent.run(query)
    
    def explore_database(self):
        """Explore the papers in the database"""
        if not os.path.exists(self.db_path):
            return "Database not initialized yet. Search for papers first to populate the database."
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get paper count
            cursor.execute("SELECT COUNT(*) FROM papers")
            paper_count = cursor.fetchone()[0]
            
            # Get latex content count
            cursor.execute("SELECT COUNT(*) FROM latex_content")
            latex_count = cursor.fetchone()[0]
            
            # Get 5 most recent papers
            cursor.execute('''
                SELECT arxiv_id, title, published FROM papers 
                ORDER BY download_timestamp DESC LIMIT 5
            ''')
            recent_papers = cursor.fetchall()
            
            recent_papers_text = "\n".join([
                f"- [{row[0]}]({row[0]}): {row[1]} ({row[2][:10]})" 
                for row in recent_papers
            ])
            
            # Get top categories
            cursor.execute('''
                SELECT papers.categories FROM papers
            ''')
            all_categories = cursor.fetchall()
            
            # Process categories (they're stored as JSON strings)
            category_count = {}
            for cat_json in all_categories:
                if cat_json[0]:
                    categories = json.loads(cat_json[0])
                    for cat in categories:
                        if cat in category_count:
                            category_count[cat] += 1
                        else:
                            category_count[cat] = 1
            
            # Get top 5 categories
            top_categories = sorted(category_count.items(), key=lambda x: x[1], reverse=True)[:5]
            categories_text = "\n".join([f"- {cat}: {count} papers" for cat, count in top_categories])
            
            conn.close()
            
            return f"""## Database Statistics
- Total papers: {paper_count}
- Papers with LaTeX content: {latex_count}

## Top Categories:
{categories_text}

## Recent Papers:
{recent_papers_text}
"""
        except sqlite3.Error as e:
            return f"Error exploring database: {e}"
    
    def search_database(self, query):
        """Search the database for papers matching a query"""
        if not query:
            return "Please enter a search query."
        
        return self.agent.run(f"Search the database for papers about {query}")
    
    def launch(self, share=False, **kwargs):
        """Launch the custom Gradio UI"""
        with gr.Blocks(title="ScholarAgent with Advanced ArXiv Tools") as demo:
            gr.Markdown("# 📚 ScholarAgent with Advanced ArXiv Integration")
            
            with gr.Tabs():
                # Tab 1: Standard Chat Interface
                with gr.Tab("Chat"):
                    stored_messages = gr.State([])
                    file_uploads_log = gr.State([])
                    chatbot = gr.Chatbot(
                        label="ScholarAgent",
                        type="messages",
                        avatar_images=(
                            None,
                            "https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/communication/Alfred.png",
                        ),
                        resizeable=True,
                        scale=1,
                    )
                    
                    # File upload capability
                    if self.file_upload_folder is not None:
                        upload_file = gr.File(label="Upload a file")
                        upload_status = gr.Textbox(label="Upload Status", interactive=False, visible=False)
                        upload_file.change(
                            self.upload_file,
                            [upload_file, file_uploads_log],
                            [upload_status, file_uploads_log],
                        )
                    
                    text_input = gr.Textbox(lines=2, label="Ask about any research topic")
                    text_input.submit(
                        self.log_user_message,
                        [text_input, file_uploads_log],
                        [stored_messages, text_input],
                    ).then(self.interact_with_agent, [stored_messages, chatbot], [chatbot])
                
                # Tab 2: Paper Search
                with gr.Tab("Quick Paper Search"):
                    gr.Markdown("## 🔍 Search for Recent arXiv Papers")
                    keywords_input = gr.Textbox(
                        label="Enter keywords", 
                        placeholder="e.g., quantum computing, transformers, computer vision",
                        lines=1
                    )
                    search_button = gr.Button("Search")
                    search_results = gr.Markdown()
                    search_button.click(
                        self.search_papers_simple, 
                        inputs=[keywords_input], 
                        outputs=[search_results]
                    )
                
                # Tab 3: Paper Analysis
                with gr.Tab("Paper Analysis"):
                    gr.Markdown("## 📝 Analyze Specific arXiv Papers")
                    paper_id = gr.Textbox(
                        label="Enter arXiv ID or URL", 
                        placeholder="e.g., 2403.17271 or https://arxiv.org/abs/2403.17271",
                        lines=1
                    )
                    analysis_request = gr.Textbox(
                        label="What would you like to know about this paper?",
                        placeholder="e.g., Summarize the main contributions, Explain the methodology, Evaluate the limitations",
                        lines=2
                    )
                    analyze_button = gr.Button("Analyze Paper")
                    analysis_results = gr.Markdown()
                    analyze_button.click(
                        self.analyze_paper, 
                        inputs=[paper_id, analysis_request], 
                        outputs=[analysis_results]
                    )
                
                # Tab 4: LaTeX Extraction
                with gr.Tab("LaTeX Extraction"):
                    gr.Markdown("## 📄 Extract LaTeX Content from Papers")
                    latex_paper_id = gr.Textbox(
                        label="Enter arXiv ID or URL", 
                        placeholder="e.g., 2403.17271 or https://arxiv.org/abs/2403.17271",
                        lines=1
                    )
                    extract_type = gr.Radio(
                        ["All LaTeX", "Equations Only", "Algorithms Only", "Important Content"],
                        label="What would you like to extract?",
                        value="Important Content"
                    )
                    extract_button = gr.Button("Extract Content")
                    extract_results = gr.Markdown()
                    extract_button.click(
                        self.extract_latex, 
                        inputs=[latex_paper_id, extract_type], 
                        outputs=[extract_results]
                    )
                
                # Tab 5: Database Explorer
                with gr.Tab("Database Explorer"):
                    gr.Markdown("## 🗃️ Explore Your Local arXiv Database")
                    with gr.Row():
                        refresh_button = gr.Button("Refresh Database Statistics")
                        search_db_button = gr.Button("Search Database")
                    
                    db_query = gr.Textbox(
                        label="Search Query", 
                        placeholder="e.g., quantum computing",
                        lines=1
                    )
                    
                    db_info = gr.Markdown("Click 'Refresh' to view database statistics")
                    
                    refresh_button.click(
                        self.explore_database, 
                        inputs=[], 
                        outputs=[db_info]
                    )
                    
                    search_db_button.click(
                        self.search_database, 
                        inputs=[db_query], 
                        outputs=[db_info]
                    )
        
        # Launch the demo
        demo.launch(share=share, **kwargs)
