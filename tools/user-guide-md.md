# ScholarAgent Advanced ArXiv Integration

This guide explains how to use the new advanced arXiv capabilities added to ScholarAgent, which enable you to not just search for papers, but download, analyze and extract content from them.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Feature Overview](#feature-overview)
- [Usage Examples](#usage-examples)
- [Database Features](#database-features)
- [Command-line Options](#command-line-options)
- [Adding to Existing Projects](#adding-to-existing-projects)
- [Troubleshooting](#troubleshooting)

## Installation

Add the advanced arXiv tool to your existing ScholarAgent setup:

1. Download the `advanced_arxiv_tool.py` file to your project's `tools` folder
2. Update your `prompts.yaml` file with the new version
3. Update your `app.py` to import and initialize the tool

For the enhanced UI (optional):
1. Add `custom_gradio_ui.py` to your project
2. Add `run_scholaragent.py` for easier launching

## Quick Start

Run with the standard agent and UI:
```bash
python app.py --download-path ./papers
```

Or use the enhanced interface:
```bash
python run_scholaragent.py --download-path ./papers --share
```

## Feature Overview

The advanced arXiv integration adds these capabilities:

1. **Complete Paper Information**: Get all metadata about a paper including authors, categories, DOI, and more
2. **LaTeX Source Download**: Access the actual LaTeX source files of papers
3. **Content Extraction**: Extract and analyze equations, algorithms, and other content
4. **Local Database**: Build a searchable local database of papers you've analyzed
5. **Enhanced UI**: Custom interface for paper analysis and database exploration

## Usage Examples

### Basic Paper Information

Ask the agent about a paper:
```
Get detailed information about arXiv paper 2403.12345
```

### Analyzing Paper Content

Ask for specific analysis:
```
For paper https://arxiv.org/abs/2311.12462, explain the key mathematical formulations and how they work
```

### Extracting LaTeX Content

Extract specific content:
```
Extract all equations from paper 2403.17764 and explain what they mean
```

### Searching Your Database

Search previously downloaded papers:
```
Search my database for papers about quantum computing
```

## Database Features

The tool creates an SQLite database containing:

- **Paper metadata**: Complete information about each paper
- **LaTeX content**: Extracted source code
- **Analysis history**: Records of papers you've analyzed

You can view database information in the "Database Explorer" tab of the enhanced UI.

## Command-line Options

The launcher supports several options:

```
python run_scholaragent.py --help

Options:
  --download-path PATH   Directory to store downloaded papers (default: ./papers)
  --db-path PATH         Path to SQLite database file (default: {download_path}/arxiv_papers.db)
  --share                Generate a public URL for sharing the interface
  --model-id MODEL       HuggingFace model ID to use (default: Qwen/Qwen2.5-Coder-32B-Instruct)
```

## Adding to Existing Projects

To add just the core functionality to an existing project:

```python
from tools.advanced_arxiv_tool import AdvancedArxivTool

# Initialize the tool
arxiv_tool = AdvancedArxivTool(download_path="./papers")

# Add to your agent's tools
agent = CodeAgent(
    model=model,
    tools=[final_answer, web_search, arxiv_tool],
    # ... other parameters
)
```

## Troubleshooting

### Paper Download Issues

If you encounter issues downloading papers:

1. Check your internet connection
2. Verify the arXiv ID is correct
3. Make sure the download directory is writable

### Database Issues

If database errors occur:

1. Check if the database file path is valid
2. Ensure you have write permissions for the directory
3. Use the `--db-path` option to specify an alternative location

### UI Not Displaying

If the enhanced UI doesn't display properly:

1. Make sure you've installed the latest Gradio version
2. Check for any console errors
3. Try launching with the default UI: `python app.py`
