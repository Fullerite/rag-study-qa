import logging
import gradio as gr
from rag.embedding.embedding_model import SentenceTransformersEmbeddingModel
from rag.generation.generation_model import TransformersGenerationModel
from rag.rag_pipeline import RAGPipeline
from rag_logs.logger import configure_logging
from exceptions.custom_exceptions import ModelLoadingError, RAGInitializationError


def initialize_pipeline(
    embedding_model_name: str,
    generation_model_name: str
):
    global rag_pipeline

    try:
        embedding_model = SentenceTransformersEmbeddingModel(
            model_name=embedding_model_name
        )
        generation_model = TransformersGenerationModel(
            model_name=generation_model_name
        )
    except ModelLoadingError as e:
        return f"An unexpected error occured when loading {repr(e.model_name)}."

    try:
        rag_pipeline = RAGPipeline(embedding_model, generation_model)
    except RAGInitializationError as e:
        return f"An unexpected error occured during RAG pipeline initialization."
    
    rag_pipeline.create_knowledge_corpus(
        data_dir="data",
        task_pattern=r"(Question\s+\d+\..*?)(?=Question\s+\d+\.|\Z)",
        answer_pattern=r"^[A-D]\)",
        add_start_end_index=True
    )

def query(
    system_prompt: str,
    user_query: str
):
    global rag_pipeline

    streamed_output = rag_pipeline.query(
        user_query=user_query,
        system_prompt=system_prompt,
        stream_output=True
    )

    generated_text = ""
    for text_chunk in streamed_output:
        generated_text += text_chunk
        yield generated_text


configure_logging()
logger = logging.getLogger(__name__)

# TODO: create the knowledge corpus on startup, and not on each query call
# TODO: add file upload functionality to dynamically extend knowledge corpus ???
# TODO: add task and answer patterns configuration for better flexibility ???
query_interface = gr.Interface(
    fn=query,
    inputs=[
        gr.Textbox(label="System Prompt", lines=10),
        gr.Textbox(lines=2, placeholder="Enter your question here...", label="User Query"),
    ],
    outputs=gr.Textbox(label="Streaming Output", autoscroll=True),
    title="RAG Pipeline for English Proficiency Test",
    description="Ask questions about your English proficiency test and get detailed, streamed explanations."
)
load_models_interface = gr.Interface(
    fn=initialize_pipeline,
    inputs=[
        gr.Textbox(label="Embedding Model Name"),
        gr.Textbox(label="Generation Model Name")
    ],
    outputs=gr.Textbox(label="State"),
    title="Load The Models for RAG Pipeline"
)
app = gr.TabbedInterface(
    [load_models_interface, query_interface],
    ["Load models", "Query"]
)
app.queue().launch()