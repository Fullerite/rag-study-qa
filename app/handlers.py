import gradio as gr

from rag.embedding.embedding_model import SentenceTransformersEmbeddingModel
from rag.generation.generation_model import TransformersGenerationModel
from rag.rag_pipeline import RAGPipeline

import os
import shutil
from textwrap import dedent

from exceptions.custom_exceptions import (
    ModelLoadingError,
    RAGInitializationError,
    CorpusCreationError,
    QueryProcessingError
)
from typing import List


import logging
logger = logging.getLogger(__name__)
rag_pipeline = None
DATA_DIR = "data"


def get_corpus_files_md():
    files = sorted([filename for filename in os.listdir(DATA_DIR) if filename.endswith(".pdf")])
    if files:
        result_md = f"Current PDF files in '{DATA_DIR}' ({len(files)} files):\n- " + "\n- ".join(files)
    else:
        result_md = f"No files currently in '{DATA_DIR}' ({len(files)} files)"
    return result_md


def upload_file(file_list: List[str]):
    global rag_pipeline

    update_status = ""
    if file_list:
        try:
            for tmp_path in file_list:
                filename = os.path.basename(tmp_path)
                dst_path = os.path.join(DATA_DIR, filename)
                shutil.copy(tmp_path, dst_path)
                logger.info(f"Successfully uploaded new file '{filename}' to '{DATA_DIR}'")
        except Exception as e:
            logger.exception("An unexpected error occured during file upload")
            raise gr.Error(
                message=(
                    "Unexpected error during file upload. "
                    "Please check the application logs for technical details."
                ),
                duration=None,
                print_exception=False
            )

        try:
            if rag_pipeline is not None:
                rag_pipeline.create_knowledge_corpus(
                    data_dir=DATA_DIR,
                    task_pattern=r"(Question\s+\d+\..*?)(?=Question\s+\d+\.|\Z)",
                    answer_pattern=r"^[A-D]\)",
                )
                update_status = "Knowledge corpus has been updated with new files.\n\n"
        except CorpusCreationError as e:
            raise gr.Error(
                message=(
                    f"An error occured during knowledge corpus update using the '{e.data_dir}' directory. "
                    f"Please check the application logs for technical details."
                ),
                duration=None,
                print_exception=False
            )
        except Exception as e:
            logger.exception("An unexpected error occured during knowledge corpus update")
            raise gr.Error(
                message=(
                    "An unexpected error occured during knowledge corpus update. "
                    "Please check the application logs for technical details."
                ),
                duration=None,
                print_exception=False
            )

    return update_status + get_corpus_files_md()


def initialize_pipeline(
    embedding_model_name: str,
    generation_model_name: str
):
    global rag_pipeline

    if rag_pipeline is not None:
        rag_pipeline._cleanup()
        rag_pipeline = None

    state = []
    embedding_model_name = embedding_model_name.strip()
    generation_model_name = generation_model_name.strip()

    if not embedding_model_name:
        raise gr.Error(
            message="Your embedding model name is empty. Please, enter something in the 'Embedding Model Name' text field.",
            duration=None,
            print_exception=False
        )
    elif not generation_model_name:
        raise gr.Error(
            message="Your generation model name is empty. Please, enter something in the 'Generation Model Name' text field.",
            duration=None,
            print_exception=False
        )

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
            data_dir=DATA_DIR,
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

    system_prompt = system_prompt.strip()
    user_query = user_query.strip()

    if rag_pipeline is None:
        raise gr.Error(
            message=(
                "You haven't initialized the RAG pipeline yet. "
                "Proceed to the 'Load Models' page, please."
            ),
            duration=None,
            print_exception=False
        )

    try:
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
            message="Your query is empty. Please, enter something in the 'User Query' text field.",
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
