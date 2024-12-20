# RAG Evaluation Module

The **RAG Evaluation Module** provides a comprehensive framework for evaluating Retrieval-Augmented Generation (RAG) systems. It supports automatic evaluation of generated answers on key aspects like correctness, faithfulness, and hallucination detection. This module currently works with the **LlamaIndex** framework for query execution and response generation.

## Features

- **Correctness Evaluation:** Automatically score the generated answers based on writing correctness, relevance, goal alignment, and clarity.
- **Faithfulness Evaluation:** Assess how well the generated answers adhere to the retrieved insights.
- **Hallucination Detection:** Evaluate potential hallucinations in the answers by comparing them with responses to rephrased questions.
- **Retrieval Evaluation**: Measures the quality of retrieval using a ground truth dataset.
- **Customizable Evaluation:** Supports both default and custom evaluation modes, allowing the user to supply their own dataset and questions.

## Requirements

- Python 3.8 or higher
- Dependencies listed in `requirements.txt` (automatically installed during setup)

### Key Dependencies:

- `google-generativeai`: For correctness and faithfulness evaluation
- `ranx`: For retrieval evaluation
- `sentence-transformers`: For embedding-based similarity computations
- `sklearn`: For cosine similarity calculations
- `LlamaIndex`: For querying and response generation

## Installation

1. Install the package using pip:

```bash
pip install git+https://github.com/edgar454/rag-evaluator.git
```

Alternatively, clone the repository and install locally:

```bash
git clone https://github.com/edgar454/rag-evaluator.git
cd rag-evaluator
pip install .
```

2. Set up your environment variables for Google Generative AI API:
   ```bash
   export GOOGLE_API_KEY=<your_google_api_key>
   ```

## Usage

### Initializing the Module

```python
from rag_evaluation import RAGEvaluation

# Initialize the query engine (using LlamaIndex)
from llama_index import GPTVectorStoreIndex

data = "your_dataset_here"
index = GPTVectorStoreIndex.from_documents(data)
query_engine = index.as_query_engine()

# Initialize the evaluation module
rageval = RAGEvaluation(query_engine=query_engine, google_api_key='<your_google_api_key>')
```

### Running the Evaluation

Call the `evaluate_rag` method to perform the evaluation:

```python
results = rageval.evaluate_rag()
print(results)
```

### Example Output

```python
{
  "correctness_score": 8.75,
  "faithfulness_score": 9.25,
  "retrieval_score": 8.9,
  "hallucination_score": 7.8,
  "total_score": 8.675
}
```

### Example Explanation of Correctness

The correctness evaluation uses the following criteria:

1. **Writing Correctness:** Clarity, grammar, and spelling of the generated answer.
2. **Relevance:** How well the answer addresses the question.
3. **Goal Alignment:** Whether the answer fulfills the user's intent.
4. **Clarity and Precision:** Conciseness and lack of ambiguity.

The evaluation generates a detailed explanation for each score, ensuring transparency and actionable insights for improvement.

## Notes

- **Retrieval Framework:** This module is currently compatible only with **LlamaIndex** for retrieval and query processing.
- **Google API:** The module uses Google Generative AI models for evaluating correctness and faithfulness. Ensure you have access to these APIs.
- **Custom Evaluation Mode:** Provide custom datasets and questions if needed by specifying `eval_mode="custom"` during initialization.

## Future Improvements

- Support for additional retrieval frameworks.
- Enhanced evaluation metrics for multi-turn dialogues.
- Visualization tools for evaluation metrics and results.
- Enhancement on evaluation feedback
---

For further assistance or feature requests, feel free to create an issue in this repository.

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes with descriptive messages.
4. Push your branch and create a pull request.

