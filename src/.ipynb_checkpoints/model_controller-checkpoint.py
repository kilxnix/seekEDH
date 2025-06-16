# src/model_controller.py
import os
import json
import logging
import datetime
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MTGModelController:
    def __init__(self, model_dir="models", base_model_name="gpt2"):
        self.model_dir = model_dir
        self.base_model_name = base_model_name
        self.current_model_path = os.path.join(model_dir, "current")
        self.versions_dir = os.path.join(model_dir, "versions")
        
        # Create directories
        os.makedirs(self.current_model_path, exist_ok=True)
        os.makedirs(self.versions_dir, exist_ok=True)
        
        # Command log
        self.command_log_path = os.path.join(model_dir, "command_log.jsonl")
        
        # Initialize model
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load the language model"""
        try:
            # Check if we have a fine-tuned model
            if os.path.exists(os.path.join(self.current_model_path, "config.json")):
                logger.info("Loading fine-tuned model")
                self.model = GPT2LMHeadModel.from_pretrained(self.current_model_path)
                self.tokenizer = GPT2Tokenizer.from_pretrained(self.current_model_path)
            else:
                # Load base model
                logger.info(f"Loading base model: {self.base_model_name}")
                self.model = GPT2LMHeadModel.from_pretrained(self.base_model_name)
                self.tokenizer = GPT2Tokenizer.from_pretrained(self.base_model_name)
                
                # Save the base model
                self.model.save_pretrained(self.current_model_path)
                self.tokenizer.save_pretrained(self.current_model_path)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def fine_tune(self, training_data_path, output_dir=None, epochs=3, batch_size=4, learning_rate=5e-5):
        """Fine-tune the model on Magic: The Gathering data"""
        try:
            logger.info(f"Fine-tuning model with {training_data_path}")
            
            # Create output directory if not provided
            if not output_dir:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = os.path.join(self.versions_dir, f"version_{timestamp}")
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Load training data
            with open(training_data_path, 'r', encoding='utf-8') as f:
                training_text = f.read()
            
            # Tokenize training data
            self.tokenizer.pad_token = self.tokenizer.eos_token
            encodings = self.tokenizer(training_text, return_tensors="pt", truncation=True, max_length=512)
            
            # Create dataset
            class MTGDataset(torch.utils.data.Dataset):
                def __init__(self, encodings):
                    self.encodings = encodings
                
                def __len__(self):
                    return self.encodings.input_ids.size(0)
                
                def __getitem__(self, idx):
                    return {key: val[idx] for key, val in self.encodings.items()}
            
            dataset = MTGDataset(encodings)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Prepare optimizer and scheduler
            optimizer = AdamW(self.model.parameters(), lr=learning_rate)
            total_steps = len(dataloader) * epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=100, 
                num_training_steps=total_steps
            )
            
            # Training loop
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            self.model.train()
            
            for epoch in range(epochs):
                logger.info(f"Starting epoch {epoch+1}/{epochs}")
                total_loss = 0
                
                for batch in dataloader:
                    # Move batch to device
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = self.model(input_ids=batch["input_ids"], labels=batch["input_ids"])
                    loss = outputs.loss
                    total_loss += loss.item()
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                avg_loss = total_loss / len(dataloader)
                logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
            
            # Save the fine-tuned model
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            # Update current model
            self.model.save_pretrained(self.current_model_path)
            self.tokenizer.save_pretrained(self.current_model_path)
            
            # Save metadata
            metadata = {
                "version_name": os.path.basename(output_dir),
                "training_data": training_data_path,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model fine-tuned and saved to {output_dir}")
            return output_dir
            
        except Exception as e:
            logger.error(f"Error fine-tuning model: {str(e)}")
            raise
    
    def generate_text(self, prompt, max_length=100, temperature=0.7, top_p=0.9):
        """Generate text with the model"""
        try:
            # Move to evaluation mode
            self.model.eval()
            
            # Encode prompt
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
            
            # Generate text
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            return generated_text
        
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            return f"Error: {str(e)}"
    
    def log_command(self, command, params, user_id):
        """Log a command to the command log"""
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "command": command,
            "params": params,
            "user_id": user_id
        }
        
        with open(self.command_log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def execute_command(self, command, params, user_id):
        """Execute a command"""
        # Log the command
        self.log_command(command, params, user_id)
        
        if command == "generate":
            prompt = params.get("prompt", "")
            max_length = params.get("max_length", 100)
            temperature = params.get("temperature", 0.7)
            top_p = params.get("top_p", 0.9)
            
            result = self.generate_text(prompt, max_length, temperature, top_p)
            
            return {
                "success": True,
                "text": result
            }
        
        elif command == "fine_tune":
            training_data = params.get("training_data")
            epochs = params.get("epochs", 3)
            batch_size = params.get("batch_size", 4)
            learning_rate = params.get("learning_rate", 5e-5)
            
            if not training_data:
                return {
                    "success": False,
                    "error": "No training data provided"
                }
            
            try:
                output_dir = self.fine_tune(training_data, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
                
                return {
                    "success": True,
                    "model_dir": output_dir
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e)
                }
        
        elif command == "list_versions":
            try:
                versions = []
                for version_dir in os.listdir(self.versions_dir):
                    metadata_path = os.path.join(self.versions_dir, version_dir, "metadata.json")
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        versions.append(metadata)
                
                return {
                    "success": True,
                    "versions": versions
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e)
                }
        
        elif command == "activate_version":
            version = params.get("version")
            
            if not version:
                return {
                    "success": False,
                    "error": "No version specified"
                }
            
            try:
                version_path = os.path.join(self.versions_dir, version)
                
                if not os.path.exists(version_path):
                    return {
                        "success": False,
                        "error": f"Version {version} not found"
                    }
                
                # Load the specified version
                model = GPT2LMHeadModel.from_pretrained(version_path)
                tokenizer = GPT2Tokenizer.from_pretrained(version_path)
                
                # Save as current model
                model.save_pretrained(self.current_model_path)
                tokenizer.save_pretrained(self.current_model_path)
                
                # Update instance variables
                self.model = model
                self.tokenizer = tokenizer
                
                return {
                    "success": True,
                    "message": f"Activated version {version}"
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e)
                }
        
        else:
            return {
                "success": False,
                "error": f"Unknown command: {command}"
            }