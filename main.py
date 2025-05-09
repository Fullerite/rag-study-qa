import logging
import gradio as gr

from rag.embedding.embedding_model import SentenceTransformersEmbeddingModel
from rag.generation.generation_model import TransformersGenerationModel
from rag.rag_pipeline import RAGPipeline
from rag_logs.logger import configure_logging

from textwrap import dedent
from exceptions.custom_exceptions import (
    ModelLoadingError,
    RAGInitializationError,
    CorpusCreationError,
    QueryProcessingError
)


configure_logging()
logger = logging.getLogger(__name__)
rag_pipeline = None


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
        state.append("- Embedding model has been loaded.")
        yield "\n".join(state)
        generation_model = TransformersGenerationModel(
            model_name=generation_model_name
        )
        state.append("- Generation model has been loaded.")
        yield "\n".join(state)
    except ModelLoadingError as e:
        raise gr.Error(
            message=(
                f"An error occured during model loading. Could not load '{e.model_name}'. "
                f"Please double check the model name."
            ),
            duration=None,
            print_exception=False
        )
    except Exception as e:
        logger.exception("An unexpected error occured during model loading")
        raise gr.Error(
            message=(
                "An unexpected error occured during model loading. "
                "Please check the application logs for technical details."
            ),
            duration=None,
            print_exception=False
        )

    try:
        rag_pipeline = RAGPipeline(embedding_model, generation_model)
        state.append("- RAG pipeline has been initialized.")
        yield "\n".join(state)
    except RAGInitializationError as e:
        raise gr.Error(
            message=(
                "An error occured during RAG pipeline initializaiton. "
                "Please check the application logs for technical details."
            ),
            duration=None,
            print_exception=False
        )
    except Exception as e:
        logger.exception("An unexpected error occured during RAG pipeline initialization")
        raise gr.Error(
            message=(
                "An unexpected error occured during RAG pipeline initialization. "
                "Please check the application logs for technical details."
            ),
            duration=None,
            print_exception=False
        )

    try:
        rag_pipeline.create_knowledge_corpus(
            data_dir="data",
            task_pattern=r"(Question\s+\d+\..*?)(?=Question\s+\d+\.|\Z)",
            answer_pattern=r"^[A-D]\)",
        )
        state.append("- Knowledge corpus has been created.")
        state.append("\n\n**RAG pipeline initializtion complete**")
        yield "\n".join(state)
    except CorpusCreationError as e:
        raise gr.Error(
            message=(
                f"An error occured during knowledge corpus creation using the '{e.data_dir}' directory. "
                f"Please check the application logs for technical details."
            ),
            duration=None,
            print_exception=False
        )
    except Exception as e:
        logger.exception("An unexpected error occured during knowledge corpus creation")
        raise gr.Error(
            message=(
                "An unexpected error occured during knowledge corpus creation. "
                "Please check the application logs for technical details."
            ),
            duration=None,
            print_exception=False
        )


def query(
    system_prompt: str,
    user_query: str
):
    global rag_pipeline

    if rag_pipeline is not None:
        try:
            system_prompt = system_prompt.strip()
            user_query = user_query.strip()

            if not user_query:
                raise ValueError()

            system_instructions = dedent("""
            ## Behavior instructions
            Explain yourself thoroughly, don't omit any crucial details.
            Be mindful and respectful to your user.
            Don't use any Markdown or LATEX text formatting symbols.
            Your answer language should match your user's query language. The only exceptions are the original file contents.
            """)
            if system_prompt:
                system_prompt = system_instructions + "\n\n" + system_prompt
            else:
                system_prompt = system_instructions
            print(system_prompt)
            streamed_output, context = rag_pipeline.query(
                user_query=user_query,
                system_prompt=system_prompt,
                stream_output=True
            )
            yield context, "Generating..."

            generated_text = ""
            for text_chunk in streamed_output:
                generated_text += text_chunk
                yield context, generated_text
        except QueryProcessingError as e:
            raise gr.Error(
                message=(
                    f"An error occured during query processing: '{e.query}'. "
                    f"Please check the application logs for technical details."
                ),
                duration=None,
                print_exception=False
            )
        except ValueError as e:
            raise gr.Error(
                message="Your query is empty.",
                duration=None,
                print_exception=False
            )
        except Exception as e:
            logger.exception("An unexpected error occured during user query processing")
            raise gr.Error(
                message=(
                    "An unexpected error occured during user query processing. "
                    "Please check the application logs for technical details."
                ),
                duration=None,
                print_exception=False
            )
    else:
        raise gr.Error(
            message=(
                "You haven't initialized the RAG pipeline yet. "
                "Proceed to the 'Load models' page, please."
            ),
            duration=None,
            print_exception=False
        )


# TODO: add file upload functionality to extend knowledge corpus (a separate tab that adds the file to the data folder and maybe calls create_corpus)
# TODO: add task and answer patterns configuration for better flexibility ???
load_models_interface = gr.Interface(
    fn=initialize_pipeline,
    inputs=[
        gr.Textbox(
            label="Embedding Model Name",
            placeholder="e.g., nomic-ai/nomic-embed-text-v1.5",
            value="nomic-ai/nomic-embed-text-v1.5"
        ),
        gr.Textbox(
            label="Generation Model Name",
            placeholder="e.g., Qwen/Qwen2.5-1.5B-Instruct",
            value="Qwen/Qwen2.5-1.5B-Instruct"
        )
    ],
    outputs=gr.Markdown(label="State", value="**[X] Pipeline not itinialized**"),
    title="Load The Models for RAG Pipeline"
)
query_interface = gr.Interface(
    fn=query,
    inputs=[
        gr.Textbox(
            label="System Prompt",
            placeholder="[Optional] Guide the AI's explanation style (e.g., 'Be concise', 'Use simple language', 'Explain like I\'m a novice')",
            lines=10
        ),
        gr.Textbox(
            label="User Query",
            placeholder="Enter your question here...",
            lines=2
        ),
    ],
    outputs=[
        gr.Textbox(label="Source Information Used"),
        gr.Textbox(label="Model Output", autoscroll=True)
    ],
    title="RAG Pipeline for English Proficiency Test",
    description=(
        "Ask questions about your English proficiency test and get detailed explanations.\n\n"
        "**Disclaimer**: This tool provides explanations based only on the content of the loaded PDF documents and model's raw knowledge. "
        "The embedding model might not always fetch the correct passage. AI-generated explanations may sometimes be inaccurate or incomplete. "
        "'Source Information Used' shows the retrieved context the model used to generate the output."
    )
)
app = gr.TabbedInterface(
    [load_models_interface, query_interface],
    ["Load models", "Query"]
)
app.queue().launch()
