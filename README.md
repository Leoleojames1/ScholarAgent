# ScholarAgent

ScholarAgent is an AI-powered tool that helps researchers find the most relevant and recent research papers from ArXiv. It uses advanced search and ranking techniques to provide high-quality results based on user queries.

## Running ScholarAgent

To start the application, run:

```bash
python app.py
```

Make sure to add your Hugging Face token before running the code.

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

## Contributing
Contributions are welcome!

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-new-search
   ```
3. Commit your changes:
   ```bash
   git commit -m "Added new ranking feature"
   ```
4. Push the branch:
   ```bash
   git push origin feature-new-search
   ```
5. Open a Pull Request.

## Feedback
Tried ScholarAgent? Share your feedback! Open an issue or reach out.

## License
This project is licensed under the MIT License. See `LICENSE` for details.
