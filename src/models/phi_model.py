# src/models/phi_model.py

"""
Module containing the PhiModel class, which handles operations specific to Phi models.

This includes loading pre-trained or fine-tuned models, training with LoRA adapters,
and evaluating the model's performance on test data.
"""

import os
import logging
import time
import datetime
from typing import Any, List, Dict

import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
)
from peft import LoraConfig, PeftModel
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from .base_model import BaseModel


class PhiModel(BaseModel):
    """
    PhiModel class for handling operations specific to Phi models.

    This class inherits from BaseModel and is designed to load, fine-tune, and evaluate
    transformer-based models using LoRA adapters for efficient parameter fine-tuning.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the PhiModel.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
        """
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _setup_tokenizer_for_finetuning(self):
        """
        Adjust the tokenizer and model embeddings for fine-tuning.

        This ensures the tokenizer has necessary special tokens like PAD and EOS tokens,
        and adjusts token embedding sizes to match the model's configuration.
        """
        self.tokenizer.model_max_length = 1000

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = '[PAD]'
            self.tokenizer.pad_token_id = self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        if self.tokenizer.eos_token == self.tokenizer.pad_token:
            self.tokenizer.eos_token = '[EOS]'
            self.tokenizer.eos_token_id = self.tokenizer.add_special_tokens({'eos_token': '[EOS]'})

        self.tokenizer.padding_side = 'right'


    def load_model(self) -> None:
        """
        Load the model and tokenizer based on the provided model path.

        Depending on the configuration, the function will either load a fine-tuned model
        if available, or the base model from a local or remote repository (e.g., Hugging Face).
        
        Raises:
            FileNotFoundError: If 'finetuned_model_path' is specified but does not exist.
        """
        model_config = self.config['model']
        model_kwargs = {
            'device_map': 'auto',
            'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32,
            'trust_remote_code': True,
            'attn_implementation': "flash_attention_2",
            'use_cache': False,
        }

        model_identifier = model_config['model_identifier']
        model_name = model_config['name']
        finetuned_model_path = model_config.get('finetuned_model_path')

        # Define the base model directory
        base_model_dir = os.path.join('models', model_name, model_identifier)

        if finetuned_model_path: # Try to Load the specified fine-tuned model
            if os.path.exists(finetuned_model_path):
                self.logger.info(f"Loading fine-tuned model from {finetuned_model_path}")
                base_model = AutoModelForCausalLM.from_pretrained(model_identifier, **model_kwargs)
                self.tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path, trust_remote_code=True)
                
                base_model.resize_token_embeddings(len(self.tokenizer))
                base_model.config.pad_token_id = self.tokenizer.pad_token_id
                base_model.config.eos_token_id = self.tokenizer.eos_token_id

                self.model = PeftModel.from_pretrained(base_model, finetuned_model_path)
            else:
                message = f"Specified model_path '{finetuned_model_path}' does not exist."
                self.logger.error(message)
                raise FileNotFoundError(message)
            
        else:  # Load the base model
            if os.path.exists(base_model_dir):  # From local directory
                self.logger.info(f"Loading base model from {base_model_dir}")
                self.model = AutoModelForCausalLM.from_pretrained(base_model_dir, **model_kwargs)
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
                 
            else:   # Download from Huggingface
                self.logger.info(f"Base model not found locally. Downloading from '{model_identifier}'...")
                self.model = AutoModelForCausalLM.from_pretrained(model_identifier, **model_kwargs)
                self.tokenizer = AutoTokenizer.from_pretrained(model_identifier, trust_remote_code=True)
                os.makedirs(base_model_dir, exist_ok=True)
                self.model.save_pretrained(base_model_dir)
                self.tokenizer.save_pretrained(base_model_dir)
                self.logger.info(f"Base model saved to {base_model_dir}")
        
        self.pipeline = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer
        )

    def train(self, train_data: Any, val_data: Any) -> None:
        """
        Fine-tune the Phi model using LoRA (Low-Rank Adaptation) on the training data.

        Args:
            train_data (Any): Training dataset.
            val_data (Any): Validation dataset.
        
        LoRA allows efficient fine-tuning by adapting a few low-rank layers, reducing the 
        need for full fine-tuning of the entire model, which can be costly.
        """
        self.logger.info("Starting fine-tuning...")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_identifier = self.config['model']['model_identifier']
        model_name = self.config['model']['name']
        finetuned_model_dir = os.path.join(
            'models',
            model_name,
            f"{model_identifier}_finetuned",
            timestamp
        )
        os.makedirs(finetuned_model_dir, exist_ok=True)

        self._setup_tokenizer_for_finetuning()
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id

        training_args_dict = self.config['training']
        training_args_dict['output_dir'] = finetuned_model_dir
        training_args_dict['fp16'] = self.config['training'].get('fp16', torch.cuda.is_available())
        training_args = TrainingArguments(**training_args_dict)

        lora_args_dict = self.config['lora']
        lora_config = LoraConfig(**lora_args_dict)

        # Data collator for formatting the input data for the model
        response_template_tokens = [22550, 29901]  # Tokens for "Answer:" 
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template_tokens,
            tokenizer=self.tokenizer
        )

        # Formatting function to format input data
        def formatting_func(examples):
            output_texts = []
            for premise, hypothesis, label in zip(
                examples['premise'], examples['hypothesis'], examples['label']
            ):
                formatted_text = self.config['prompt_template'].format(
                    premise=premise, hypothesis=hypothesis
                )
                output_text = f"{formatted_text}{self.config['output_mapping'][label]}"
                output_texts.append(output_text)
            return output_texts

        # Initialize the trainer
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            peft_config=lora_config,
            data_collator=collator,
            formatting_func=formatting_func,
            max_seq_length=1000,
            tokenizer=self.tokenizer
        )

        # Start training
        trainer.train()

        # Save the fine-tuned model
        self.logger.info(f"Saving the fine-tuned model to '{finetuned_model_dir}'")
        trainer.save_model(finetuned_model_dir)  
        self.tokenizer.save_pretrained(finetuned_model_dir)


    def evaluate(self, test_data: Any) -> Dict[str, Any]:
        """
        Evaluate the model on test data.

        This method computes evaluation metrics such as accuracy and F1-score
        and logs batch-wise performance during inference.

        Args:
            test_data (Any): Test dataset.

        Returns:
            Dict[str, Any]: Dictionary containing accuracy, F1 score, and total evaluation time.
        """
                
        self.logger.info("Starting evaluation...")
        self.model.eval()
        start_time = time.time()

        # Preprocess test data
        test_data = test_data.map(self._preprocess_batch, batched=True, batch_size=self.config['evaluation']['batch_size'])

        total_predictions = []
        total_labels = []

        self.logger.debug("Iterating over test data batches...")

        # Iterate over batches
        for batch in test_data.select_columns(['messages', 'label']).iter(batch_size=self.config['evaluation']['batch_size']):
            batch_start_time = time.time()
            inputs = batch['messages']
            labels = batch['label']

            with torch.no_grad():
                predictions = self.predict(inputs)

            batch_accuracy = accuracy_score(labels, predictions)
            batch_time = time.time() - batch_start_time
            self.logger.info(f"Batch Accuracy: {batch_accuracy * 100:.2f}%, Time Taken: {batch_time:.2f} seconds")

            total_predictions.extend(predictions)
            total_labels.extend(labels)

        # Calculate metrics
        accuracy = accuracy_score(total_labels, total_predictions)
        f1 = f1_score(total_labels, total_predictions, average='weighted')
        total_time = time.time() - start_time

        self.logger.info(f"Evaluation completed in {total_time:.2f} seconds.")

        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "total_time": total_time
        }

    def predict(self, inputs: List[Dict[str, Any]]) -> List[int]:
        """
        Make predictions using the model.

        Args:
            inputs (List[Dict[str, Any]]): List of input messages.

        Returns:
            List[int]: List of predicted labels.
        """
        outputs = self.pipeline(inputs, **self.config.get('generation_args', {}))

        predictions = [
            self._extract_label(output[0]['generated_text'].strip().lower())
            for output in outputs
        ]

        return predictions
        

    def _extract_label(self, generated_text: str) -> int:
        """
        Extract the label from the generated text.
        
        This method looks for specific keywords in the generated text
        and maps them to the appropriate label ID. 

        Args:
            generated_text (str): The text generated by the model.

        Returns:
            int: The predicted label as an integer.
        """
        for word in generated_text.split():
            if word in self.config['inverse_output_mapping']:
                return self.config['inverse_output_mapping'][word]

        # Default to 'neutral' if no label is found
        return self.config['inverse_output_mapping']['neutral']
    

    def _preprocess_batch(self, batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Preprocess a batch of data by formatting the messages according to the model's requirements.

        Args:
            batch (Dict[str, List[Any]]): A batch of data.

        Returns:
            Dict[str, Any]: A dictionary with formatted messages and labels.
        """
        messages = []
        system_message = {"role": "system", "content": self.config['system_message']}

        for premise, hypothesis in zip(batch['premise'], batch['hypothesis']):
            user_prompt = self.config['prompt_template'].format(premise=premise, hypothesis=hypothesis)
            user_message = {"role": "user", "content": user_prompt}
            messages.append([system_message, user_message])  # Include both system and user messages as expected by phidata

        return {"messages": messages, "label": batch["label"]}
