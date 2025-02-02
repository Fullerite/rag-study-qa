import logging
import gradio as gr
from logging import Logger
from rag.embedding.embedding_model import SentenceTransformersEmbeddingModel
from rag.generation.generation_model import TransformersGenerationModel
from rag.rag_pipeline import RAGPipeline
from textwrap import dedent
from typing import Optional


def get_logger() -> Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]  # TODO: add persistent logs storage
    )
    logger = logging.getLogger(__name__)

    return logger


def query(
    embedding_model_name: str,
    generation_model_name: str,
    system_prompt: str,
    user_query: str,
):
    embedding_model = SentenceTransformersEmbeddingModel(
        model_name=embedding_model_name
    )
    generation_model = TransformersGenerationModel(
        model_name=generation_model_name
    )

    rag_pipeline = RAGPipeline(generation_model, embedding_model)
    rag_pipeline.create_knowledge_corpus(
        data_dir="data",
        task_pattern=r"(Question\s+\d+\..*?)(?=Question\s+\d+\.|\Z)",
        answer_pattern=r"^[A-D]\)",
        add_start_end_index=True
    )

    streamed_output = rag_pipeline.query(
        user_query=user_query,
        system_prompt=system_prompt,
        stream_output=True
    )

    generated_text = ""
    for text_chunk in streamed_output:
        generated_text += text_chunk
        yield generated_text


logger = get_logger()

# TODO: create the knowledge corpus on startup, and not on each query call
# TODO: add file upload functionality to dynamically extend knowledge corpus ???
# TODO: add task and answer patterns configuration for better flexibility ???
iface = gr.Interface(
    fn=query,
    inputs=[
        gr.Textbox(label="Embedding Model Name"),
        gr.Textbox(label="Generation Model Name"),
        gr.Textbox(label="System Prompt", lines=10),
        gr.Textbox(lines=2, placeholder="Enter your question here...", label="User Query"),
    ],
    outputs=gr.Textbox(label="Streaming Output", autoscroll=True),
    title="RAG Pipeline for English Proficiency Test",
    description="Ask questions about your English proficiency test and get detailed, streamed explanations."
)
iface.queue().launch()