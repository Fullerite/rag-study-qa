import gradio as gr
from app.handlers import initialize_pipeline, query


import logging
from rag_logs.logger import configure_logging
configure_logging()
logger = logging.getLogger(__name__)


# TODO: add file upload functionality to extend knowledge corpus (a separate tab that adds the file to the data folder and maybe calls create_corpus)
# TODO: add task and answer patterns configuration for better flexibility ???
with gr.Blocks() as app:
    gr.Markdown("# RAG pipeline for study material Q&A")
    with gr.Tabs():
        with gr.TabItem(label="Load Models"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        embedding_model_tb = gr.Textbox(
                            label="Embedding Model Name",
                            placeholder="e.g., nomic-ai/nomic-embed-text-v1.5",
                            value="nomic-ai/nomic-embed-text-v1.5"
                        )
                    with gr.Row():
                        generation_model_tb = gr.Textbox(
                            label="Generation Model Name",
                            placeholder="e.g., Qwen/Qwen2.5-1.5B-Instruct",
                            value="Qwen/Qwen2.5-1.5B-Instruct"
                        )
                with gr.Column(scale=1):
                    pipeline_state_md = gr.Markdown(
                        label="State",
                        value="**[X] Pipeline not initialized**"
                    )
                with gr.Column(scale=2):
                    load_docs_file = gr.File(
                        file_types=["pdf"]
                    )
            with gr.Row():
                with gr.Column(scale=1):
                    load_button = gr.Button(
                        value="Initialize Pipeline",
                        variant="primary",
                    )
                gr.Column(scale=1)
            load_button.click(
                fn=initialize_pipeline,
                inputs=[embedding_model_tb, generation_model_tb],
                outputs=[pipeline_state_md]
            )
        with gr.TabItem(label="Query"):
            with gr.Row():
                gr.Markdown(
                    "Ask questions about your English proficiency test and get detailed explanations.\n\n"
                    "**Disclaimer**: This tool provides explanations based only on the content of the loaded PDF documents and model's raw knowledge. "
                    "The embedding model might not always fetch the correct passage. AI-generated explanations may sometimes be inaccurate or incomplete. "
                    "'Source Information Used' shows the retrieved context the model used to generate the output."
                )
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        system_prompt_tb = gr.Textbox(
                            label="System Prompt",
                            placeholder="[Optional] Guide the AI's explanation style (e.g., 'Be concise', 'Use simple language', 'Explain like I\'m a novice')",
                            lines=10
                        )
                    with gr.Row():
                        user_query_tb = gr.Textbox(
                            label="User Query",
                            placeholder="Enter your question here...",
                            lines=2
                        )
                    with gr.Row():
                        submit_button = gr.Button(
                            value="Submit Query",
                            variant="primary"
                        )
                with gr.Column(scale=2):
                    with gr.Row():
                        source_info_tb = gr.Textbox(
                            label="Source Information Used",
                            show_copy_button=True,
                            interactive=False
                        )
                    with gr.Row():
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
