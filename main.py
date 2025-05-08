import logging
import gradio as gr
from rag.embedding.embedding_model import SentenceTransformersEmbeddingModel
from rag.generation.generation_model import TransformersGenerationModel
from rag.rag_pipeline import RAGPipeline
from rag_logs.logger import configure_logging
from exceptions.custom_exceptions import (
    ModelLoadingError,
    RAGInitializationError,
    CorpusCreationError,
    QueryProcessingError
)


configure_logging()
logger = logging.getLogger(__name__)
rag_pipeline = None


# TODO: add the ability to re-initialize the models without re-creating the knowledge corpus
# TODO: load default models on the initial startup
# TODO: make the knowledge corpus persistent ???
def initialize_pipeline(
    embedding_model_name: str,
    generation_model_name: str
):
    global rag_pipeline
    state = []

    try:
        embedding_model = SentenceTransformersEmbeddingModel(
            model_name=embedding_model_name
        )
        state.append("Embedding model has been loaded.")
        yield " ".join(state)
        generation_model = TransformersGenerationModel(
            model_name=generation_model_name
        )
        state.append("Generation model has been loaded.")
        yield " ".join(state)
    except ModelLoadingError as e:
        raise gr.Error(
            message=e.message,
            duration=None
        )
    except Exception as e:
        logger.exception("An unexpected error occured during model loading")
        raise gr.Error(
            message=f"An unexpected error occured during model loading",
            duration=None
        )

    try:
        rag_pipeline = RAGPipeline(embedding_model, generation_model)
        state.append("RAG pipeline initialized.")
        yield " ".join(state)
    except RAGInitializationError as e:
        raise gr.Error(
            message=e.message,
            duration=None
        )
    except Exception as e:
        logger.exception("An unexpected error occured during RAG pipeline initialization")
        raise gr.Error(
            message=f"An unexpected error occured during RAG pipeline initialization",
            duration=None
        )

    try:
        rag_pipeline.create_knowledge_corpus(
            data_dir="data",
            task_pattern=r"(Question\s+\d+\..*?)(?=Question\s+\d+\.|\Z)",
            answer_pattern=r"^[A-D]\)",
        )
        state.append("Knowledge corpus created.")
        yield " ".join(state)
    except CorpusCreationError as e:
        raise gr.Error(
            message=e.message,
            duration=None
        )
    except Exception as e:
        logger.exception("An unexpected error occured during knowledge corpus creation")
        raise gr.Error(
            message=f"An unexpected error occured during knowledge corpus creation",
            duration=None
        )


def query(
    system_prompt: str,
    user_query: str
):
    global rag_pipeline

    if rag_pipeline is not None:
        try:
            streamed_output, context = rag_pipeline.query(
                user_query=user_query,
                system_prompt=system_prompt,
                stream_output=True
            )

            generated_text = ""
            for text_chunk in streamed_output:
                generated_text += text_chunk
                yield generated_text, context
        except QueryProcessingError as e:
            raise gr.Error(
                message=e.message,
                duration=None
            )
        except Exception as e:
            logger.exception("An unexpected error occured during user query processing")
            raise gr.Error(
                message=f"An unexpected error occured during user query processing",
                duration=None
            )
    else:
        raise gr.Error(
            message="You haven't initialized the RAG pipeline yet. Proceed to the 'Load models' page, please.",
            duration=None
        )


# TODO: create the knowledge corpus on startup, and not on each query call
# TODO: add file upload functionality to dynamically extend knowledge corpus ???
# TODO: add task and answer patterns configuration for better flexibility ???
load_models_interface = gr.Interface(
    fn=initialize_pipeline,
    inputs=[
        gr.Textbox(label="Embedding Model Name"),
        gr.Textbox(label="Generation Model Name")
    ],
    outputs=gr.Textbox(label="State"),
    title="Load The Models for RAG Pipeline"
)
query_interface = gr.Interface(
    fn=query,
    inputs=[
        gr.Textbox(label="System Prompt", lines=10),
        gr.Textbox(lines=2, placeholder="Enter your question here...", label="User Query"),
    ],
    outputs=[
        gr.Textbox(label="Source Information Used"),
        gr.Textbox(label="Model Output", autoscroll=True)
    ],
    title="RAG Pipeline for English Proficiency Test",
    description=(
        "Ask questions about your English proficiency test and get detailed explanations.\n\n"
        "**Disclaimer**: This tool provides explanations based only on the content of the loaded PDF documents and model's raw knowledge. The embedding model might not always fetch the correct passage. AI-generated explanations may sometimes be inaccurate or incomplete. 'Source Information Used' shows the retrieved context the model used to generate the output."
    )
)
app = gr.TabbedInterface(
    [load_models_interface, query_interface],
    ["Load models", "Query"]
)
app.queue().launch()
