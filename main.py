import torch
from rag.embedding.embedding_model import SentenceTransformersEmbeddingModel
from rag.generation.generation_model import TransformersGenerationModel
from rag.rag_pipeline import RAGPipeline
from textwrap import dedent


import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
GENERATION_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
embedding_model = SentenceTransformersEmbeddingModel(
    model_name=EMBEDDING_MODEL
)
generation_model = TransformersGenerationModel(
    model_name=GENERATION_MODEL
)

rag_pipeline = RAGPipeline(generation_model, embedding_model)
rag_pipeline.create_knowledge_corpus(
    data_dir="data",
    task_pattern=r"(Question\s+\d+\..*?)(?=Question\s+\d+\.|\Z)",
    answer_pattern=r"^[A-D]\)",
    add_start_end_index=True
)

user_query = "Explain how to solve question 12 from test 2."
system_prompt = dedent(
    """
    ## Instructions
    You are a world-class AI assistant designed to answer questions from a user regarding their English proficiency test.
    You excell in explaining tasks in an easy-to-understand way.

    ## Response Format
    1. **Task Explanation**: Briefly describe the task or question.
    2. **Key Points**: Highlight the main concepts or skills being tested.
    3. **Step-by-Step Solution**: Provide a detailed explanation of how to solve the task.
    4. **Final Answer**: Summarize the solution in a clear and concise manner.
    """
)
stream_output = True

# TODO: Build a gradio web UI
generated_text = rag_pipeline.query(
    user_query=user_query,
    system_prompt=system_prompt,
    stream_output=stream_output
)

if not stream_output:
    from rich.console import Console
    from rich.markdown import Markdown
    console = Console()
    console.print(Markdown(generated_text))

print(torch.cuda.memory_summary())
