import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from threading import Thread
from textwrap import dedent
from typing import Optional, Union, Generator
from exceptions.custom_exceptions import ModelLoadingError, GenerationError, TemplateTokenizationError


import logging
logger = logging.getLogger(__name__)


class TransformersGenerationModel:
    """
    A class for generating text using a causal language model from Hugging Face's Transformers library.

    This class handles model initialization, tokenization, device management (CPU/GPU),
    and text generation, with support for streaming output and quantization using BitsAndBytesConfig.
    """

    def __init__(self, model_name: str, bnb_config: Optional[BitsAndBytesConfig] = None):
        """
        Initializes the TransformersGenerationModel with the specified model and configuration. Uses half-precision (FP16),
        if bnb_config is not provided.

        Args:
            - model_name (str): The name or path of the pre-trained model to load.
            - bnb_config (Optional[BitsAndBytesConfig]): Configuration for quantization using BitsAndBytes.
                                                         Defaults to None (no quantization).

        Raises:
            - ModelLoadingError: If an error occurs while loading the SentenceTransformers model.
        """

        self.model_name = model_name
        self._bnb_config = bnb_config

        logger.info(f"Initializing generation model: {repr(model_name)}")
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                torch_dtype=torch.float16 if bnb_config is None else None,
                quantization_config=self._bnb_config,
                device_map="auto" if bnb_config is not None else None,
                low_cpu_mem_usage=True
            )
            if self._bnb_config is None:
                self._model = self._model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
                self.device = self._model.device
                logger.info(f"Using device: {self.device}")
            else:
                self.device = self._model.device
                logger.info(f"Using device_map='auto' with primary device: {self.device}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                use_fast=True
            )
        except Exception as e:
            logger.exception(
                f"Failed to load the model\n"
                f"Model: {repr(self.model_name)}\n"
                f"Quantization Config: {self._bnb_config}"
            )
            raise ModelLoadingError(
                f"An error occurred while loading the model {repr(model_name)}:\n{e}",
                model_name=self.model_name,
                quantization_config=self._bnb_config
            )

        if self._tokenizer.chat_template is None:
            chat_template = dedent("""
            {% for message in messages %}
            {{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
            {% endfor %}
            {% if add_generation_prompt %}
            {{ '<|im_start|>assistant\n' }}
            {% endif %}
            """)
            self._tokenizer.chat_template = chat_template

        logger.info("Generation model initialized successfully")


    def __str__(self):
        return (
            f"TransformersGenerationModel(\n"
            f"  model_name={repr(self.model_name)},\n"
            f"  device={self.device}\n"
            f")"
        )


    def __repr__(self):
        return (
            f"TransformersGenerationModel(\n"
            f"  model_name={repr(self.model_name)},\n"
            f"  model={repr(self._model)},\n"
            f"  tokenizer={repr(self._tokenizer)},\n"
            f"  bnb_config={repr(self._bnb_config)},\n"
            f"  device={self.device}\n"
            f")"
        )


    def generate(self, query: str, system_prompt: str = "", stream_output: bool = False) -> Union[Generator[str, None, None], str]:
        """
        Generates text based on the input query and system prompt.

        Args:
            - query (str): The user's input query.
            - system_prompt (str): The system prompt to guide the model's behavior. Defaults to "".
            - stream_output (bool): Whether to stream the output in real-time. Defaults to False.

        Returns:
            - Union[str, Generator[str, None, None]]:
                - stream_output=False: The complete generated text string.
                - stream_output=True: A generator that yields chunks of generated text as they are produced.
        
        Raises:
            - GenerationError: If an error occurs during generation.
        """

        try:
            # Prepare the messages for the chat template
            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": query
                }
            ]

            # Apply the chat template to format the input
            logger.info("Applying chat template")
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Move the inputs ids to a proper device
            if self.device != self._model.device:
                logger.warning(
                    f"Device mismatch: self.device ({self.device}) does not match self._model.device ({self._model.device})."
                    f"Updating self.device."
                )
                self.device = self._model.device
            logger.info("Tokenizing model prompt")
            input_ids = self._tokenizer(
                prompt,
                return_tensors="pt"
            ).to(self.device)
        except Exception as e:
            messages = messages if 'messages' in locals() else 'N/A'
            prompt = prompt if 'prompt' in locals() else 'N/A'
            logger.exception(
                f"Failed during template application or tokenization\n"
                f"Query: '{query}'\n"
                f"System Prompt: '{system_prompt}'\n"
                f"Messages: {messages}\n"
                f"Prompt: {prompt}\n"
                f"Model: {repr(self)}"
            )
            raise TemplateTokenizationError(
                f"An error occurred during template application or tokenization:\n{e}",
                query=query,
                system_prompt=system_prompt,
                model_info=repr(self)
            )

        try:
            # Pre-configure generation kwargs
            generation_kwargs = {
                "max_new_tokens": 512,
                "do_sample": True,
                "temperature": 0.1,
                "top_k": 30,
                "top_p": 0.9
            }

            if stream_output:
                # Configure the streamer for real-time output
                streamer = TextIteratorStreamer(
                    tokenizer=self._tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )

                # Add streamer to generation kwargs
                generation_kwargs["streamer"] = streamer

                # Offload generation to a separate thread
                thread = Thread(
                    target=self._model.generate,
                    kwargs={**input_ids, **generation_kwargs}
                )
                thread.start()

                logger.info("Starting text streaming")

                # Yield the generating text chunk-by-chunk
                for chunk in streamer:
                    yield chunk

                logger.info("Text streaming completed")
            else:
                # Add streamer to generation kwargs
                generation_kwargs["streamer"] = None

                logger.info("Starting text generation")

                # Generate text without streaming
                with torch.no_grad():             
                    output_ids = self._model.generate(
                        **input_ids,
                        **generation_kwargs
                    )

                # Decode the generated text, excluding the input prompt
                generated_text = self._tokenizer.decode(
                    output_ids[0][len(input_ids[0]):],  # filter out the prompt
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )

                logger.info("Text generation completed")
                return generated_text
        except Exception as e:
            input_ids = input_ids if 'input_ids' in locals() else 'N/A'
            streamer = streamer if 'streamer' in locals() else 'N/A'
            generation_kwargs = generation_kwargs if 'generation_kwargs' in locals() else 'N/A'
            logger.exception(
                f"Failed to generate output using\n"
                f"Query: '{query}'\n"
                f"System Prompt: '{system_prompt}'\n"
                f"Input IDs: {input_ids}\n"
                f"Stream Output: {stream_output}\n"
                f"Streamer: {repr(streamer)}\n"
                f"Generation Kwargs: {generation_kwargs}\n"
                f"Model: {repr(self)}"
            )
            raise GenerationError(
                f"An error occurred while generating output:\n{e}",
                query=query,
                system_prompt=system_prompt,
                stream_output=stream_output,
                generation_kwargs=generation_kwargs,
                model_info=repr(self)
            )
