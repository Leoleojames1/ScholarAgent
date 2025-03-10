# ScholarAgent Advanced ArXiv Integration - Developer Guide

This guide explains how to integrate the Advanced ArXiv Tool into ScholarAgent or similar projects based on the smolagents framework.

## Table of Contents

- [Overview](#overview)
- [Integration Steps](#integration-steps)
- [Advanced Configuration](#advanced-configuration)
- [UI Integration](#ui-integration)
- [Customizing the Tool](#customizing-the-tool)
- [Database Schema](#database-schema)
- [Best Practices](#best-practices)

## Overview

The Advanced ArXiv Tool enhances ScholarAgent with:

1. Full paper metadata retrieval from arXiv
2. LaTeX source code download and extraction
3. Content analysis capabilities
4. SQLite database for persistent storage
5. Enhanced UI components

## Integration Steps

### Step 1: Add the Tool to Your Project

1. Copy `advanced_arxiv_tool.py` to your project's `tools` directory
2. Update your imports in `app.py`:

```python
from tools.advanced_arxiv_tool import AdvancedArxivTool
```

### Step 2: Initialize the Tool

```python
# Create necessary directories
os.makedirs("./papers", exist_ok=True)

# Initialize the tool
advanced_arxiv = AdvancedArxivTool(
    download_path="./papers",
    db_path="./arxiv_papers.db"  # Optional, uses download_path/arxiv_papers.db by default
)
```

### Step 3: Add to Agent's Tools

```python
# Create the agent with the advanced tool
agent = CodeAgent(
    model=model,
    tools=[final_answer, web_search, advanced_arxiv],
    # ... other parameters
)
```

### Step 4: Update Prompts

Replace your `prompts.yaml` file with the updated version that includes instructions for using the advanced arXiv tool.

## Advanced Configuration

### Custom Database Location

```python
# Store database in a different location
advanced_arxiv = AdvancedArxivTool(
    download_path="./papers",
    db_path="/path/to/database/arxiv.db"
)
```

### SQLite Pragmas (Advanced)

For performance optimization, you can customize the SQLite connection:

```python
# In advanced_arxiv_tool.py
def _init_database(self):
    try:
        conn = sqlite3.connect(self.db_path)
        
        # Performance optimizations
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        
        # Rest of initialization...
```

## UI Integration

### Option 1: Use Standard GradioUI

```python
from Gradio_UI import GradioUI

# Initialize the UI with your agent
ui = GradioUI(agent=agent, file_upload_folder="./uploads")
ui.launch(share=True)
```

### Option 2: Use Enhanced UI

```python
from custom_gradio_ui import ScholarAgentUI

# Initialize the enhanced UI
ui = ScholarAgentUI(
    agent=agent,
    download_path="./papers",
    db_path="./arxiv_papers.db",
    file_upload_folder="./uploads"
)
ui.launch(share=True)
```

## Customizing the Tool

### Adding New Actions

To add new functionality to the tool:

1. Add a new method to the `AdvancedArxivTool` class
2. Update the `forward` method to handle the new action
3. Update the tool's `description` and `inputs` attributes

Example for adding a "summarize" action:

```python
def summarize_paper(self, arxiv_id: str) -> Dict[str, Any]:
    """Generate a summary of the paper using its abstract and intro."""
    # Implementation...

def forward(self, action: str, arxiv_id_or_url: str = None, query: str = ""):
    # Existing actions...
    elif action == "summarize":
        arxiv_id = self.extract_arxiv_id(arxiv_id_or_url)
        return self.summarize_paper(arxiv_id)
```

## Database Schema

The tool uses two main tables:

### papers

Stores paper metadata:
- `arxiv_id` (TEXT PRIMARY KEY): Unique identifier
- `title` (TEXT): Paper title
- `authors` (TEXT): JSON array of author names
- `abstract` (TEXT): Paper abstract
- `published` (TEXT): Publication date
- `pdf_link` (TEXT): Link to PDF
- `arxiv_url` (TEXT): Link to arXiv page
- `categories` (TEXT): JSON array of arXiv categories
- `download_timestamp` (TEXT): When the paper was fetched
- `comment` (TEXT): Optional arXiv comment
- `journal_ref` (TEXT): Optional journal reference
- `doi` (TEXT): Optional DOI

### latex_content

Stores extracted LaTeX:
- `arxiv_id` (TEXT PRIMARY KEY): Matches papers table
- `content` (TEXT): Full LaTeX content
- `content_length` (INTEGER): Length of content
- `file_count` (INTEGER): Number of .tex files
- `file_info` (TEXT): JSON array with file details
- `timestamp` (TEXT): When the content was extracted

## Best Practices

1. **Error Handling**: The tool implements robust error handling to deal with API issues, download failures, and parsing errors.

2. **Rate Limiting**: The tool respects arXiv's rate limits with a 3-second delay between API calls.

3. **Caching**: The database serves as a cache to avoid re-downloading papers.

4. **Structured Responses**: All methods return structured dictionaries with consistent keys for better agent interaction.

5. **Status Indicators**: Responses include `success` boolean and descriptive error messages.
