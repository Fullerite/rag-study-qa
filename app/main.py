import gradio as gr
from app.handlers import get_corpus_files_md, upload_file, initialize_pipeline, query

import logging
from rag_logs.logger import configure_logging
configure_logging()
logger = logging.getLogger(__name__)


with gr.Blocks() as app:
    gr.Markdown("# RAG pipeline for study material Q&A")
    with gr.Tabs():
        with gr.TabItem(label="Load Models"):
            gr.Markdown("### If you have uploaded new files, please reinitialize the RAG pipeline for changes to take effect.")
            with gr.Row():
                with gr.Column(scale=1):
                    embedding_model_tb = gr.Textbox(
                        label="Embedding Model Name",
                        placeholder="e.g., nomic-ai/nomic-embed-text-v1.5",
                        value="nomic-ai/nomic-embed-text-v1.5"
                    )
                    generation_model_tb = gr.Textbox(
                        label="Generation Model Name",
                        placeholder="e.g., Qwen/Qwen2.5-1.5B-Instruct",
                        value="Qwen/Qwen2.5-1.5B-Instruct"
                    )
                    load_models_button = gr.Button(
                        value="Initialize Pipeline",
                        variant="primary",
                    )
                with gr.Column(scale=1):
                    pipeline_state_md = gr.Markdown(
                        value="**[X] Pipeline not initialized**",
                        min_height=100
                    )
                with gr.Column(scale=2):
                    load_docs_file = gr.File(
                        label="Extend knowledge base",
                        file_types=[".pdf"],
                        file_count="multiple"
                    )
                    upload_files_button = gr.Button(
                        value="Upload Files",
                        variant="secondary"
                    )
                    docs_md = gr.Markdown(
                        value=get_corpus_files_md,
                        min_height=100
                    )
            load_models_button.click(
                fn=initialize_pipeline,
                inputs=[embedding_model_tb, generation_model_tb],
                outputs=[pipeline_state_md]
            )
            upload_files_button.click(
                fn=upload_file,
                inputs=[load_docs_file],
                outputs=[docs_md]
            )
        with gr.TabItem(label="Query"):
            with gr.Row():
                gr.Markdown(
                    "### Ask questions about your English proficiency test and get detailed explanations.\n\n"
                    "### *Disclaimer:* This tool provides explanations based only on the content of the loaded PDF documents and model's raw knowledge. "
                    "The embedding model might not always fetch the correct passage. AI-generated explanations may sometimes be inaccurate or incomplete. "
                    "'Source Information Used' shows the retrieved context the model used to generate the output."
                )
            with gr.Row():
                with gr.Column(scale=1):
                    system_prompt_tb = gr.Textbox(
                        label="System Prompt",
                        placeholder="[Optional] Guide the AI's explanation style (e.g., 'Be concise', 'Use simple language', 'Explain like I\'m a novice')",
                        lines=10
                    )
                    user_query_tb = gr.Textbox(
                        label="User Query",
                        placeholder="Enter your question here...",
                        lines=2
                    )
                    submit_button = gr.Button(
                        value="Submit Query",
                        variant="primary"
                    )
                with gr.Column(scale=2):
                    source_info_tb = gr.Textbox(
                        label="Source Information Used",
                        show_copy_button=True,
                        interactive=False
                    )
                    model_output_tb = gr.Textbox(
                        label="Model Output",
                        show_copy_button=True,
                        autoscroll=True,
                        interactive=False
                    )
            submit_button.click(
                fn=query,
                inputs=[system_prompt_tb, user_query_tb],
                outputs=[source_info_tb, model_output_tb]
            )

if __name__ == "__main__":
    logger.info("Starting up Gradio interface")
    app.queue().launch()
