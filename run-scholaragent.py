#!/usr/bin/env python3
"""
Run ScholarAgent with Advanced ArXiv Integration

This script launches the ScholarAgent with the advanced arXiv tool integration,
providing a complete research assistant for finding, downloading, and analyzing
academic papers.

Usage:
    python run_scholaragent.py --download-path ./papers [--db-path ./arxiv.db] [--share]

Options:
    --download-path: Directory to store downloaded papers (default: ./papers)
    --db-path: Path to SQLite database file (default: {download_path}/arxiv_papers.db)
    --share: Generate a public URL for sharing the interface
    --model-id: HuggingFace model ID to use (default: Qwen/Qwen2.5-Coder-32B-Instruct)
"""

import os
import argparse
from pathlib import Path
import yaml
from smolagents import CodeAgent, HfApiModel, DuckDuckGoSearchTool
from tools.final_answer import FinalAnswerTool
from tools.advanced_arxiv_tool import AdvancedArxivTool
from standalone_ui import ScholarAgentStandaloneUI  # Use the standalone UI instead

def parse_args():
    parser = argparse.ArgumentParser(description="Run ScholarAgent with Advanced ArXiv Integration")
    parser.add_argument("--download-path", type=str, default="./papers", 
                        help="Directory to store downloaded papers")
    parser.add_argument("--db-path", type=str, default=None,
                        help="Path to SQLite database file (default: {download_path}/arxiv_papers.db)")
    parser.add_argument("--share", action="store_true", 
                        help="Generate a public URL for sharing the interface")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-Coder-32B-Instruct",
                        help="HuggingFace model ID to use")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create directories
    download_path = Path(args.download_path)
    download_path.mkdir(parents=True, exist_ok=True)
    
    # Set database path if not specified
    db_path = args.db_path if args.db_path else str(download_path / "arxiv_papers.db")
    
    # Initialize tools
    final_answer = FinalAnswerTool()
    try:
        web_search = DuckDuckGoSearchTool(max_results=5)
        have_web_search = True
    except ImportError:
        print("DuckDuckGoSearchTool not available. Web search will be disabled.")
        have_web_search = False
    
    # Initialize the advanced arXiv tool
    advanced_arxiv = AdvancedArxivTool(
        download_path=str(download_path),
        db_path=db_path
    )
    
    print(f"✓ Initialized Advanced ArXiv Tool")
    print(f"  - Download Path: {download_path}")
    print(f"  - Database: {db_path}")
    
    # Load prompt templates
    with open("prompts.yaml", 'r') as stream:
        prompt_templates = yaml.safe_load(stream)
    
    print(f"✓ Loaded prompt templates")
    
    # Create the AI Model
    model = HfApiModel(
        max_tokens=2096,
        temperature=0.5,
        model_id=args.model_id,
    )
    
    print(f"✓ Initialized model: {args.model_id}")
    
    # Create the agent with all tools
    tools = [final_answer, advanced_arxiv]
    if have_web_search:
        tools.append(web_search)
        
    agent = CodeAgent(
        model=model,
        tools=tools,
        max_steps=6,
        verbosity_level=1,
        name="ScholarAgent",
        description="An AI agent that fetches, downloads, and analyzes research papers from arXiv",
        prompt_templates=prompt_templates
    )
    
    print(f"✓ Created ScholarAgent with advanced arXiv capabilities")
    
    # Launch the standalone UI
    print(f"✓ Launching ScholarAgent UI...")
    ui = ScholarAgentStandaloneUI(
        agent=agent,
        download_path=str(download_path),
        db_path=db_path
    )
    
    ui.launch(share=args.share)

if __name__ == "__main__":
    main()
