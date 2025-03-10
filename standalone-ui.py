"""
Standalone UI for ScholarAgent with Advanced ArXiv Integration

This module provides a simple, standalone Gradio UI for ScholarAgent 
without requiring the GradioUI class from the original codebase.
"""
import os
import sqlite3
import json
from pathlib import Path
import gradio as gr
from smolagents.agents import MultiStepAgent


class ScholarAgentStandaloneUI:
    """
    Standalone UI for ScholarAgent with specialized tabs for arXiv research
    
    This UI creates a Gradio interface with tabs for:
    - Chat with the agent
    - Paper search and analysis
    - Database exploration
    """
    
    def __init__(self, agent: MultiStepAgent, download_path: str, db_path: str = None):
        """
        Initialize the ScholarAgent UI
        
        Args:
            agent: The CodeAgent instance with tools
            download_path: Path where papers are downloaded
            db_path: Path to the SQLite database
        """
        self.agent = agent
        self.download_path = Path(download_path)
        self.db_path = db_path if db_path else str(self.download_path / "arxiv_papers.db")
    
    def run_agent(self, query):
        """Run the agent with a query and return the result"""
        return self.agent.run(query)
    
    def chat_with_agent(self, message, history):
        """Chat interface for the agent"""
        # Append user message
        history.append((message, ""))
        # Run agent
        response = self.agent.run(message)
        # Update last response
        history[-1] = (message, str(response))
        return "", history
    
    def analyze_paper(self, paper_id, option, query):
        """Analyze a specific paper"""
        if not paper_id:
            return "Please provide a paper ID or URL"
        
        if option == "Paper Info":
            prompt = f"Get detailed information about the paper {paper_id}"
        elif option == "Download Source":
            prompt = f"Download the source code for paper {paper_id} and tell me what you found"
        elif option == "Extract LaTeX":
            if query:
                prompt = f"Extract LaTeX content from paper {paper_id} related to: {query}"
            else:
                prompt = f"Extract the most important LaTeX content from paper {paper_id}"
        elif option == "Analyze Content":
            if query:
                prompt = f"For paper {paper_id}, {query}"
            else:
                prompt = f"Analyze the main contributions and methodology of paper {paper_id}"
        
        return self.agent.run(prompt)
    
    def search_papers(self, keywords):
        """Search for papers with keywords"""
        if not keywords:
            return "Please enter keywords to search for."
        
        formatted_input = f"Search for recent papers about {keywords}"
        return self.agent.run(formatted_input)
    
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
            
            if paper_count == 0:
                return "No papers in the database yet. Try searching for some papers first."
            
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
            
            # Get categories if they exist
            category_count = {}
            try:
                cursor.execute('''
                    SELECT papers.categories FROM papers
                ''')
                all_categories = cursor.fetchall()
                
                # Process categories (they're stored as JSON strings)
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
            except:
                categories_text = "No category information available"
            
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
        except Exception as e:
            return f"Unexpected error: {e}"
    
    def search_database(self, query):
        """Search the database for papers matching a query"""
        if not query:
            return "Please enter a search query."
        
        return self.agent.run(f"Search the database for papers about {query}")
    
    def launch(self, share=False, **kwargs):
        """Launch the custom Gradio UI"""
        with gr.Blocks(title="ScholarAgent with Advanced ArXiv") as demo:
            gr.Markdown("# 📚 ScholarAgent with Advanced ArXiv Integration")
            
            with gr.Tabs():
                # Tab 1: Chat interface
                with gr.Tab("Chat with Agent"):
                    chat_history = gr.Chatbot()
                    user_input = gr.Textbox(
                        label="Ask anything about research papers", 
                        placeholder="e.g., Find papers about quantum computing, Analyze paper 2403.12345",
                        lines=2
                    )
                    
                    user_input.submit(
                        self.chat_with_agent, 
                        inputs=[user_input, chat_history], 
                        outputs=[user_input, chat_history]
                    )
                
                # Tab 2: Quick Paper Search
                with gr.Tab("Quick Paper Search"):
                    keywords_input = gr.Textbox(
                        label="Enter keywords", 
                        placeholder="e.g., quantum computing, transformers, reinforcement learning",
                        lines=1
                    )
                    search_button = gr.Button("Search")
                    search_results = gr.Markdown()
                    search_button.click(
                        self.search_papers, 
                        inputs=[keywords_input], 
                        outputs=[search_results]
                    )
                
                # Tab 3: Paper Analysis
                with gr.Tab("Paper Analysis"):
                    paper_id = gr.Textbox(
                        label="Enter arXiv ID or URL", 
                        placeholder="e.g., 2403.17271 or https://arxiv.org/abs/2403.17271",
                        lines=1
                    )
                    analysis_options = gr.Radio(
                        choices=["Paper Info", "Download Source", "Extract LaTeX", "Analyze Content"],
                        label="What would you like to do?",
                        value="Paper Info"
                    )
                    analysis_query = gr.Textbox(
                        label="Specific request (if needed)",
                        placeholder="e.g., Extract equations, Find main contributions",
                        lines=2,
                        visible=False
                    )
                    
                    def update_query_visibility(option):
                        return gr.update(visible=option in ["Extract LaTeX", "Analyze Content"])
                    
                    analysis_options.change(
                        update_query_visibility,
                        inputs=[analysis_options],
                        outputs=[analysis_query]
                    )
                    
                    analyze_button = gr.Button("Process Paper")
                    analysis_results = gr.Markdown()
                    
                    analyze_button.click(
                        self.analyze_paper,
                        inputs=[paper_id, analysis_options, analysis_query],
                        outputs=[analysis_results]
                    )
                
                # Tab 4: Database Explorer
                with gr.Tab("Database Explorer"):
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


# Simple example usage
if __name__ == "__main__":
    from smolagents import CodeAgent, HfApiModel
    from tools.final_answer import FinalAnswerTool
    from tools.advanced_arxiv_tool import AdvancedArxivTool
    import yaml
    
    # Initialize the agent
    final_answer = FinalAnswerTool()
    advanced_arxiv = AdvancedArxivTool(download_path="./papers")
    
    # Load prompt templates
    with open("prompts.yaml", 'r') as stream:
        prompt_templates = yaml.safe_load(stream)
    
    # Create the agent
    model = HfApiModel(
        max_tokens=2096,
        temperature=0.5,
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    )
    
    agent = CodeAgent(
        model=model,
        tools=[final_answer, advanced_arxiv],
        max_steps=6,
        verbosity_level=1,
        prompt_templates=prompt_templates
    )
    
    # Launch the UI
    ui = ScholarAgentStandaloneUI(
        agent=agent,
        download_path="./papers"
    )
    ui.launch(share=True)
