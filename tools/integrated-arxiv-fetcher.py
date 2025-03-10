from typing import Any, Dict, List, Optional
from smolagents.tools import Tool
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import re
import time
import os
import tarfile
import gzip
import shutil
from datetime import datetime
import logging
from pathlib import Path
import json
import sqlite3

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedArxivTool(Tool):
    """
    Advanced ArXiv Tool for ScholarAgent
    
    This tool enhances the ScholarAgent with powerful capabilities to:
    - Fetch detailed paper metadata from arXiv
    - Download and extract LaTeX source files
    - Extract and parse LaTeX content
    - Store data in SQLite for persistent access
    
    Example Usage:
    ```python
    # Basic usage with ScholarAgent
    from smolagents import CodeAgent, HfApiModel
    
    # Import the tool
    from advanced_arxiv_tool import AdvancedArxivTool
    
    # Initialize the tool with a download directory
    arxiv_tool = AdvancedArxivTool(download_path="./papers")
    
    # Create agent with the tool
    agent = CodeAgent(
        model=HfApiModel(...),
        tools=[arxiv_tool, ...],
        ...
    )
    
    # Example tasks the agent can now perform:
    # - "Get detailed information about the paper at https://arxiv.org/abs/2402.17764"
    # - "Download the LaTeX source for arXiv:2311.12462 and extract key equations"
    # - "Analyze the mathematical formulations in the recent paper on quantum computing"
    ```
    
    For SQLite integration (enabled by default):
    ```python
    # With custom database path
    arxiv_tool = AdvancedArxivTool(
        download_path="./papers",
        db_path="./arxiv_papers.db"
    )
    ```
    """
    
    name = "advanced_arxiv"
    description = "An advanced tool to fetch, download and analyze arXiv papers including metadata, LaTeX source, and content extraction."
    inputs = {
        'action': {'type': 'string', 'description': 'Action to perform: "get_info", "download_source", "extract_latex", or "search_db"'},
        'arxiv_id_or_url': {'type': 'string', 'description': 'arXiv ID or URL of the paper'},
        'query': {'type': 'string', 'description': 'Optional query for searching papers in the database', 'default': ''},
    }
    output_type = "dict"
    
    def __init__(self, 
                 download_path: str,
                 db_path: Optional[str] = None):
        """
        Initialize the arXiv tool with SQLite database.
        
        Args:
            download_path: Path where papers will be downloaded
            db_path: Path to SQLite database (default: download_path/arxiv_papers.db)
        """
        super().__init__()
        self.download_path = Path(download_path)
        self.download_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite database
        if db_path is None:
            db_path = str(self.download_path / "arxiv_papers.db")
        
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with required tables if they don't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create papers table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                arxiv_id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                abstract TEXT,
                published TEXT,
                pdf_link TEXT,
                arxiv_url TEXT,
                categories TEXT,
                download_timestamp TEXT,
                comment TEXT,
                journal_ref TEXT,
                doi TEXT
            )
            ''')
            
            # Create latex_content table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS latex_content (
                arxiv_id TEXT PRIMARY KEY,
                content TEXT,
                content_length INTEGER,
                file_count INTEGER,
                file_info TEXT,
                timestamp TEXT,
                FOREIGN KEY (arxiv_id) REFERENCES papers (arxiv_id)
            )
            ''')
            
            conn.commit()
            logger.info(f"Successfully initialized SQLite database at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize SQLite database: {e}")
        finally:
            if conn:
                conn.close()
    
    def extract_arxiv_id(self, url_or_id: str) -> str:
        """Extract arXiv ID from a URL or direct ID string."""
        patterns = [
            r'arxiv.org/abs/([\w.-]+)',
            r'arxiv.org/pdf/([\w.-]+)',
            r'^([\w.-]+)$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url_or_id)
            if match:
                return match.group(1)
        
        raise ValueError("Could not extract arXiv ID from the provided input")

    def fetch_paper_info(self, arxiv_id: str) -> Dict[str, Any]:
        """Fetch paper metadata from arXiv API."""
        # First check if we already have this paper in the database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,))
            result = cursor.fetchone()
            
            if result:
                # Convert row to dictionary
                columns = [col[0] for col in cursor.description]
                paper_info = dict(zip(columns, result))
                
                # Convert string representations back to lists
                paper_info['authors'] = json.loads(paper_info['authors'])
                paper_info['categories'] = json.loads(paper_info['categories'])
                
                logger.info(f"Retrieved paper {arxiv_id} from database")
                conn.close()
                return paper_info
        except sqlite3.Error as e:
            logger.error(f"Database error when retrieving paper: {e}")
        finally:
            if conn:
                conn.close()
        
        # If not in database or error occurred, fetch from API
        base_url = 'http://export.arxiv.org/api/query'
        query_params = {
            'id_list': arxiv_id,
            'max_results': 1
        }
        
        url = f"{base_url}?{urllib.parse.urlencode(query_params)}"
        
        try:
            time.sleep(3)  # Be nice to the API
            with urllib.request.urlopen(url) as response:
                xml_data = response.read().decode('utf-8')
            
            root = ET.fromstring(xml_data)
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            entry = root.find('atom:entry', namespaces)
            if entry is None:
                raise ValueError("No paper found with the provided ID")
            
            paper_info = {
                'arxiv_id': arxiv_id,
                'title': entry.find('atom:title', namespaces).text.strip(),
                'authors': [author.find('atom:name', namespaces).text 
                           for author in entry.findall('atom:author', namespaces)],
                'abstract': entry.find('atom:summary', namespaces).text.strip(),
                'published': entry.find('atom:published', namespaces).text,
                'pdf_link': next(
                    link.get('href') for link in entry.findall('atom:link', namespaces)
                    if link.get('type') == 'application/pdf'
                ),
                'arxiv_url': next(
                    link.get('href') for link in entry.findall('atom:link', namespaces)
                    if link.get('rel') == 'alternate'
                ),
                'categories': [cat.get('term') for cat in entry.findall('atom:category', namespaces)],
                'download_timestamp': datetime.utcnow().isoformat()
            }
            
            # Add optional fields if present
            optional_fields = ['comment', 'journal_ref', 'doi']
            for field in optional_fields:
                elem = entry.find(f'arxiv:{field}', namespaces)
                if elem is not None:
                    paper_info[field] = elem.text
                else:
                    paper_info[field] = None
            
            # Save to database
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                INSERT OR REPLACE INTO papers 
                (arxiv_id, title, authors, abstract, published, pdf_link, arxiv_url, 
                categories, download_timestamp, comment, journal_ref, doi)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    paper_info['arxiv_id'],
                    paper_info['title'],
                    json.dumps(paper_info['authors']),
                    paper_info['abstract'],
                    paper_info['published'],
                    paper_info['pdf_link'],
                    paper_info['arxiv_url'],
                    json.dumps(paper_info['categories']),
                    paper_info['download_timestamp'],
                    paper_info.get('comment'),
                    paper_info.get('journal_ref'),
                    paper_info.get('doi')
                ))
                
                conn.commit()
                logger.info(f"Saved paper {arxiv_id} to database")
            except sqlite3.Error as e:
                logger.error(f"Failed to save paper to database: {e}")
            finally:
                if conn:
                    conn.close()
                    
            return paper_info
            
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to arXiv API: {e}")
        except ET.ParseError as e:
            raise ValueError(f"Failed to parse API response: {e}")

    def download_source(self, arxiv_id: str) -> Dict[str, Any]:
        """
        Download and extract source files for a paper.
        Returns info about extracted files or error details.
        """
        # Construct source URL
        source_url = f"https://arxiv.org/e-print/{arxiv_id}"
        paper_dir = self.download_path / arxiv_id
        paper_dir.mkdir(exist_ok=True)
        
        try:
            # Download source file
            logger.info(f"Downloading source for {arxiv_id}")
            temp_file = paper_dir / "temp_source"
            with urllib.request.urlopen(source_url) as response:
                with open(temp_file, 'wb') as f:
                    f.write(response.read())

            extraction_method = None
            tex_files = []
            
            # Try to extract as tar.gz
            try:
                with tarfile.open(temp_file, 'r:gz') as tar:
                    tar.extractall(path=paper_dir)
                    extraction_method = "tar.gz"
                    logger.info(f"Extracted tar.gz source for {arxiv_id}")
                    tex_files = list(paper_dir.glob('**/*.tex'))
            except tarfile.ReadError:
                # If not tar.gz, try as gzip
                try:
                    with gzip.open(temp_file, 'rb') as gz:
                        with open(paper_dir / 'main.tex', 'wb') as f:
                            f.write(gz.read())
                    extraction_method = "gzip"
                    logger.info(f"Extracted gzip source for {arxiv_id}")
                    tex_files = [paper_dir / 'main.tex']
                except Exception as e:
                    logger.error(f"Failed to extract source as gzip: {e}")
                    return {
                        "success": False,
                        "error": f"Failed to extract source: {str(e)}",
                        "arxiv_id": arxiv_id
                    }

            # Clean up temp file
            temp_file.unlink()
            
            return {
                "success": True,
                "arxiv_id": arxiv_id,
                "extraction_method": extraction_method,
                "paper_dir": str(paper_dir),
                "tex_files": [str(f) for f in tex_files],
                "file_count": len(tex_files)
            }

        except Exception as e:
            logger.error(f"Failed to download source for {arxiv_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to download source: {str(e)}",
                "arxiv_id": arxiv_id
            }

    def extract_latex(self, arxiv_id: str) -> Dict[str, Any]:
        """
        Extract LaTeX content from source files.
        Returns the extracted content or error details.
        """
        paper_dir = self.download_path / arxiv_id
        
        if not paper_dir.exists():
            return {
                "success": False,
                "error": f"Paper directory for {arxiv_id} does not exist. Please download the source first.",
                "arxiv_id": arxiv_id
            }
        
        # Check if we already have this in the database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM latex_content WHERE arxiv_id = ?", (arxiv_id,))
            result = cursor.fetchone()
            
            if result:
                # Convert row to dictionary
                columns = [col[0] for col in cursor.description]
                latex_data = dict(zip(columns, result))
                
                # Convert string representation back to list/dict
                if latex_data['file_info']:
                    latex_data['file_info'] = json.loads(latex_data['file_info'])
                
                logger.info(f"Retrieved LaTeX content for {arxiv_id} from database")
                conn.close()
                return {
                    "success": True,
                    "arxiv_id": arxiv_id,
                    "file_count": latex_data['file_count'],
                    "file_info": latex_data['file_info'],
                    "content": latex_data['content'],
                    "content_length": latex_data['content_length'],
                    "from_database": True
                }
        except sqlite3.Error as e:
            logger.error(f"Database error when retrieving LaTeX content: {e}")
        finally:
            if conn:
                conn.close()
        
        # Find all .tex files
        tex_files = list(paper_dir.glob('**/*.tex'))
        if not tex_files:
            return {
                "success": False,
                "error": f"No .tex files found in {paper_dir}",
                "arxiv_id": arxiv_id
            }
            
        # Read and analyze .tex files
        latex_content = []
        file_info = []
        
        for tex_file in tex_files:
            try:
                with open(tex_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    relative_path = tex_file.relative_to(paper_dir)
                    latex_content.append(f"% From file: {relative_path}\n{content}\n")
                    
                    # Extract basic info about the file
                    size = os.path.getsize(tex_file)
                    is_main = bool(re.search(r'\\begin{document}', content))
                    
                    file_info.append({
                        "path": str(relative_path),
                        "size_bytes": size,
                        "is_main": is_main,
                        "characters": len(content)
                    })
            except Exception as e:
                logger.error(f"Failed to read {tex_file}: {e}")
                continue
        
        combined_content = '\n'.join(latex_content) if latex_content else None
        content_length = sum(len(content) for content in latex_content)
        
        # Save to SQLite database
        if combined_content:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                INSERT OR REPLACE INTO latex_content 
                (arxiv_id, content, content_length, file_count, file_info, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    arxiv_id,
                    combined_content,
                    content_length,
                    len(tex_files),
                    json.dumps(file_info),
                    datetime.utcnow().isoformat()
                ))
                
                conn.commit()
                logger.info(f"Saved LaTeX content for {arxiv_id} to database")
            except sqlite3.Error as e:
                logger.error(f"Failed to save LaTeX content to database: {e}")
            finally:
                if conn:
                    conn.close()
        
        return {
            "success": bool(latex_content),
            "arxiv_id": arxiv_id,
            "file_count": len(tex_files),
            "file_info": file_info,
            "content": combined_content,
            "content_length": content_length,
            "from_database": False
        }
    
    def search_db(self, query: str) -> Dict[str, Any]:
        """
        Search for papers in the database using a simple text query.
        Returns matching papers with their metadata.
        """
        if not query:
            return {
                "success": False,
                "error": "Search query cannot be empty",
                "results": []
            }
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.create_function("REGEXP", 2, lambda expr, item: re.search(expr, item) is not None)
            cursor = conn.cursor()
            
            # Search in title, abstract, and authors
            cursor.execute('''
            SELECT * FROM papers 
            WHERE title LIKE ? OR abstract LIKE ? OR authors LIKE ?
            LIMIT 10
            ''', (f'%{query}%', f'%{query}%', f'%{query}%'))
            
            results = cursor.fetchall()
            columns = [col[0] for col in cursor.description]
            
            papers = []
            for row in results:
                paper = dict(zip(columns, row))
                # Convert string representations back to lists
                paper['authors'] = json.loads(paper['authors'])
                paper['categories'] = json.loads(paper['categories'])
                papers.append(paper)
            
            conn.close()
            
            return {
                "success": True,
                "query": query,
                "result_count": len(papers),
                "results": papers
            }
            
        except sqlite3.Error as e:
            logger.error(f"Database error when searching: {e}")
            return {
                "success": False,
                "error": f"Database error: {str(e)}",
                "query": query,
                "results": []
            }
    
    def forward(self, action: str, arxiv_id_or_url: str = None, query: str = "") -> Dict[str, Any]:
        """Process the action requested by the agent."""
        try:
            # Handle search_db action separately since it doesn't need an arxiv_id
            if action == "search_db":
                return self.search_db(query)
            
            # All other actions require an arxiv_id
            if not arxiv_id_or_url:
                return {
                    "success": False,
                    "error": "arxiv_id_or_url is required for this action"
                }
            
            # Extract arXiv ID from input URL or ID
            arxiv_id = self.extract_arxiv_id(arxiv_id_or_url)
            
            # Process based on requested action
            if action == "get_info":
                return self.fetch_paper_info(arxiv_id)
            elif action == "download_source":
                # First get metadata
                paper_info = self.fetch_paper_info(arxiv_id)
                # Then download source
                download_result = self.download_source(arxiv_id)
                # Combine the results
                return {**paper_info, "source": download_result}
            elif action == "extract_latex":
                # Make sure source exists, download if not
                paper_dir = self.download_path / arxiv_id
                if not paper_dir.exists() or not list(paper_dir.glob('**/*.tex')):
                    self.download_source(arxiv_id)
                # Extract LaTeX content
                return self.extract_latex(arxiv_id)
            else:
                return {
                    "success": False,
                    "error": f"Invalid action: {action}. Valid actions are 'get_info', 'download_source', 'extract_latex', or 'search_db'."
                }
                
        except Exception as e:
            # Return structured error response
            return {
                "success": False,
                "error": str(e),
                "action": action,
                "input": arxiv_id_or_url if arxiv_id_or_url else query
            }
