# ScholarAgent

ScholarAgent is an AI-powered tool that helps researchers find the most relevant and recent research papers from ArXiv. It uses advanced search and ranking techniques to provide high-quality results based on user queries.

## Running ScholarAgent

To start the application, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ScholarAgent.git
   cd ScholarAgent
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add your Hugging Face token as an environment variable:
   ```bash
   export HF_TOKEN=your_huggingface_token
   ```
   (On Windows, use `set HF_TOKEN=your_huggingface_token` instead.)
4. Run the application:
   ```bash
   python app.py
   ```

Once running, you can interact with the AI agent through the Gradio UI.

## Usage

1. Enter keywords or full sentence queries in the search box.
2. The agent fetches the top 3 most relevant and recent ArXiv papers.
3. If no relevant paper is found, it suggests trying different keywords.

### Example Queries
- "Transformer models in NLP"
- "Graph Neural Networks for recommendation systems"
- "Deep learning applications in medicine"

## How It Works
ScholarAgent utilizes:
- **ArXiv API** for fetching research papers.
- **BM25** for ranking papers based on relevance.
- **TF-IDF** for semantic search, allowing sentence-based queries.
- **Optimized keyword matching** to refine search results.

## Future Enhancements
- Support for more research repositories like IEEE, Springer, etc.
- Integration with GPT-powered summarization for research insights.
- Personalized recommendations based on past searches.


## Feedback
Tried ScholarAgent? Share your feedback! Open an issue or reach out.

## License
This project is licensed under the MIT License. See `LICENSE` for details.
