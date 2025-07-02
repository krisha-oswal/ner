"""
CoNLL-2003 Training Pipeline for Enhanced Named Entity Linking System
Adapts CoNLL-2003 NER dataset for entity linking training
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import pickle
from pathlib import Path
import logging
from tqdm import tqdm
import requests
import time
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EntityMention:
    """Represents an entity mention from CoNLL-2003"""
    text: str
    label: str
    start: int
    end: int
    context: str
    sentence: str

@dataclass
class LinkingExample:
    """Training example for entity linking"""
    entity_text: str
    context: str
    candidate_title: str
    candidate_description: str
    label: int  # 1 for correct link, 0 for incorrect

class CoNLL2003Processor:
    """Processes CoNLL-2003 dataset for entity linking training"""
    
    def __init__(self):
        self.label_map = {
            'O': 'O',
            'B-PER': 'PERSON',
            'I-PER': 'PERSON', 
            'B-LOC': 'LOCATION',
            'I-LOC': 'LOCATION',
            'B-ORG': 'ORGANIZATION',
            'I-ORG': 'ORGANIZATION',
            'B-MISC': 'MISC',
            'I-MISC': 'MISC'
        }
        
    def load_conll2003(self):
        """Load CoNLL-2003 dataset from HuggingFace"""
        logger.info("Loading CoNLL-2003 dataset...")
        
        try:
            dataset = load_dataset("eriktks/conll2003")
            logger.info("✅ CoNLL-2003 dataset loaded successfully")
            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return None
    
    def extract_entities(self, dataset_split) -> List[EntityMention]:
        """Extract entity mentions from CoNLL-2003 format"""
        entities = []
        
        for example in tqdm(dataset_split, desc="Extracting entities"):
            tokens = example['tokens']
            ner_tags = example['ner_tags']
            
            # Convert tag indices to labels
            tag_names = dataset_split.features['ner_tags'].feature.names
            labels = [tag_names[tag] for tag in ner_tags]
            
            # Extract entities using BIO tagging
            current_entity = None
            current_tokens = []
            
            for i, (token, label) in enumerate(zip(tokens, labels)):
                if label.startswith('B-'):
                    # Start of new entity
                    if current_entity:
                        # Save previous entity
                        entities.append(self._create_entity_mention(
                            current_tokens, current_entity, tokens, i-len(current_tokens)
                        ))
                    
                    current_entity = self.label_map[label]
                    current_tokens = [token]
                    
                elif label.startswith('I-') and current_entity:
                    # Continuation of entity
                    current_tokens.append(token)
                    
                else:
                    # End of entity or no entity
                    if current_entity:
                        entities.append(self._create_entity_mention(
                            current_tokens, current_entity, tokens, i-len(current_tokens)
                        ))
                        current_entity = None
                        current_tokens = []
            
            # Handle entity at end of sentence
            if current_entity:
                entities.append(self._create_entity_mention(
                    current_tokens, current_entity, tokens, len(tokens)-len(current_tokens)
                ))
        
        logger.info(f"Extracted {len(entities)} entity mentions")
        return entities
    
    def _create_entity_mention(self, entity_tokens: List[str], entity_type: str, 
                             all_tokens: List[str], start_idx: int) -> EntityMention:
        """Create EntityMention object from tokens"""
        entity_text = ' '.join(entity_tokens)
        
        # Create context (5 tokens before and after)
        context_start = max(0, start_idx - 5)
        context_end = min(len(all_tokens), start_idx + len(entity_tokens) + 5)
        context_tokens = all_tokens[context_start:context_end]
        context = ' '.join(context_tokens)
        
        # Full sentence
        sentence = ' '.join(all_tokens)
        
        return EntityMention(
            text=entity_text,
            label=entity_type,
            start=start_idx,
            end=start_idx + len(entity_tokens),
            context=context,
            sentence=sentence
        )

class WikipediaKnowledgeBase:
    """Creates knowledge base using Wikipedia API"""
    
    def __init__(self, cache_file='wikipedia_cache.json'):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        
    def _load_cache(self):
        """Load cached Wikipedia data"""
        if Path(self.cache_file).exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        """Save cache to file"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def search_wikipedia(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search Wikipedia for entity candidates"""
        if query in self.cache:
            return self.cache[query]
        
        try:
            # Wikipedia search API
            search_url = "https://en.wikipedia.org/api/rest_v1/page/search"
            params = {
                'q': query,
                'limit': max_results
            }
            
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            
            search_results = response.json()
            candidates = []
            
            for result in search_results.get('pages', []):
                # Get page content
                content = self._get_page_content(result['title'])
                
                candidate = {
                    'title': result['title'],
                    'page_id': str(result['pageid']),
                    'description': result.get('description', ''),
                    'url': f"https://en.wikipedia.org/wiki/{result['title'].replace(' ', '_')}",
                    'content': content
                }
                candidates.append(candidate)
                
                # Rate limiting
                time.sleep(0.1)
            
            # Cache results
            self.cache[query] = candidates
            self._save_cache()
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error searching Wikipedia for '{query}': {e}")
            return []
    
    def _get_page_content(self, title: str) -> str:
        """Get Wikipedia page content"""
        try:
            content_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
            response = requests.get(content_url)
            response.raise_for_status()
            
            data = response.json()
            return data.get('extract', '')
            
        except Exception as e:
            logger.error(f"Error getting content for '{title}': {e}")
            return ""

class EntityLinkingDatasetBuilder:
    """Builds training dataset for entity linking from CoNLL-2003"""
    
    def __init__(self):
        self.processor = CoNLL2003Processor()
        self.kb = WikipediaKnowledgeBase()
        
    def build_training_data(self, max_entities_per_type: int = 1000) -> List[LinkingExample]:
        """Build training dataset for entity linking"""
        # Load CoNLL-2003
        dataset = self.processor.load_conll2003()
        if not dataset:
            return []
        
        # Extract entities from training set
        entities = self.processor.extract_entities(dataset['train'])
        
        # Group entities by type
        entities_by_type = {}
        for entity in entities:
            if entity.label not in entities_by_type:
                entities_by_type[entity.label] = []
            entities_by_type[entity.label].append(entity)
        
        # Limit entities per type and remove duplicates
        filtered_entities = []
        for entity_type, entity_list in entities_by_type.items():
            # Remove duplicates based on text
            unique_entities = {}
            for entity in entity_list:
                if entity.text not in unique_entities:
                    unique_entities[entity.text] = entity
            
            # Limit number
            limited_entities = list(unique_entities.values())[:max_entities_per_type]
            filtered_entities.extend(limited_entities)
            
            logger.info(f"Selected {len(limited_entities)} {entity_type} entities")
        
        # Build linking examples
        linking_examples = []
        
        for entity in tqdm(filtered_entities, desc="Building linking examples"):
            examples = self._create_linking_examples(entity)
            linking_examples.extend(examples)
        
        logger.info(f"Created {len(linking_examples)} linking examples")
        return linking_examples
    
    def _create_linking_examples(self, entity: EntityMention) -> List[LinkingExample]:
        """Create positive and negative linking examples for an entity"""
        examples = []
        
        # Search for candidates
        candidates = self.kb.search_wikipedia(entity.text, max_results=10)
        
        if not candidates:
            return examples
        
        # Create positive example (first candidate assumed correct for now)
        # In practice, you'd want manual annotation or better heuristics
        positive_candidate = candidates[0]
        
        positive_example = LinkingExample(
            entity_text=entity.text,
            context=entity.context,
            candidate_title=positive_candidate['title'],
            candidate_description=positive_candidate['description'],
            label=1
        )
        examples.append(positive_example)
        
        # Create negative examples
        negative_candidates = candidates[1:4]  # Take next 3 as negatives
        
        for neg_candidate in negative_candidates:
            negative_example = LinkingExample(
                entity_text=entity.text,
                context=entity.context,
                candidate_title=neg_candidate['title'],
                candidate_description=neg_candidate['description'],
                label=0
            )
            examples.append(negative_example)
        
        # Add random negative examples
        random_entities = random.sample(candidates, min(2, len(candidates)))
        for random_entity in random_entities:
            if random_entity != positive_candidate:
                random_example = LinkingExample(
                    entity_text=entity.text,
                    context=entity.context,
                    candidate_title=random_entity['title'],
                    candidate_description=random_entity['description'],
                    label=0
                )
                examples.append(random_example)
        
        return examples

class EntityLinkingDataset(Dataset):
    """PyTorch dataset for entity linking training"""
    
    def __init__(self, examples: List[LinkingExample], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Create input text: [CLS] context [SEP] entity [SEP] candidate [SEP]
        input_text = f"{example.context} [SEP] {example.entity_text} [SEP] {example.candidate_title} {example.candidate_description}"
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(example.label, dtype=torch.long)
        }

class EntityLinkingTrainer:
    """Trainer for entity linking models"""
    
    def __init__(self, model_name='bert-base-uncased', output_dir='models'):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def train_classifier(self, train_examples: List[LinkingExample], 
                        val_examples: List[LinkingExample],
                        epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5):
        """Train BERT classifier for entity linking"""
        
        # Create datasets
        train_dataset = EntityLinkingDataset(train_examples, self.tokenizer)
        val_dataset = EntityLinkingDataset(val_examples, self.tokenizer)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        from enhanced_nel_system import TrainedNELClassifier  # Import from main file
        model = TrainedNELClassifier(self.model_name)
        model.to(self.device)
        
        # Optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for batch in progress_bar:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            # Validation
            val_accuracy = self._evaluate_classifier(model, val_loader)
            logger.info(f"Epoch {epoch+1}: Train Loss = {total_loss/len(train_loader):.4f}, "
                       f"Val Accuracy = {val_accuracy:.4f}")
        
        # Save model
        model_path = self.output_dir / 'nel_bert_model.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': self.model_name,
            'tokenizer': self.tokenizer
        }, model_path)
        
        logger.info(f"Model saved to {model_path}")
        return model
    
    def _evaluate_classifier(self, model, data_loader):
        """Evaluate classifier accuracy"""
        model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(input_ids, attention_mask)
                preds = torch.argmax(outputs, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        model.train()
        return accuracy_score(true_labels, predictions)
    
    def train_embeddings(self, train_examples: List[LinkingExample], 
                        val_examples: List[LinkingExample],
                        epochs: int = 4, batch_size: int = 16):
        """Train sentence transformer for entity linking"""
        
        # Create sentence transformer examples
        train_samples = []
        for example in train_examples:
            text1 = f"{example.context} {example.entity_text}"
            text2 = f"{example.candidate_title} {example.candidate_description}"
            score = float(example.label)
            
            train_samples.append(InputExample(texts=[text1, text2], label=score))
        
        val_samples = []
        for example in val_examples:
            text1 = f"{example.context} {example.entity_text}"
            text2 = f"{example.candidate_title} {example.candidate_description}"
            score = float(example.label)
            
            val_samples.append(InputExample(texts=[text1, text2], label=score))
        
        # Initialize model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create data loader
        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)
        
        # Loss function
        train_loss = losses.CosineSimilarityLoss(model)
        
        # Evaluator
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
            val_samples, name='nel_eval'
        )
        
        # Training
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=epochs,
            output_path=str(self.output_dir / 'nel_embeddings'),
            evaluation_steps=500,
            warmup_steps=100
        )
        
        logger.info(f"Embedding model saved to {self.output_dir / 'nel_embeddings'}")
        return model

def create_knowledge_base(linking_examples: List[LinkingExample], output_file: str):
    """Create knowledge base from linking examples"""
    kb_entities = {}
    
    for example in linking_examples:
        if example.candidate_title not in kb_entities:
            kb_entities[example.candidate_title] = {
                'title': example.candidate_title,
                'page_id': example.candidate_title.lower().replace(' ', '_'),
                'description': example.candidate_description,
                'url': f"https://en.wikipedia.org/wiki/{example.candidate_title.replace(' ', '_')}",
                'aliases': []
            }
        
        # Add entity text as alias if different from title
        if example.entity_text.lower() != example.candidate_title.lower():
            aliases = kb_entities[example.candidate_title]['aliases']
            if example.entity_text not in aliases:
                aliases.append(example.entity_text)
    
    # Convert to list
    kb_list = list(kb_entities.values())
    
    # Save knowledge base
    with open(output_file, 'w') as f:
        json.dump(kb_list, f, indent=2)
    
    logger.info(f"Knowledge base with {len(kb_list)} entities saved to {output_file}")
    return kb_list

def main():
    """Main training pipeline"""
    logger.info("Starting CoNLL-2003 Entity Linking Training Pipeline")
    
    # Step 1: Build training dataset
    logger.info("Step 1: Building training dataset...")
    builder = EntityLinkingDatasetBuilder()
    linking_examples = builder.build_training_data(max_entities_per_type=100)  # Reduced for demo
    
    if not linking_examples:
        logger.error("No training examples created!")
        return
    
    # Step 2: Split data
    train_examples, val_examples = train_test_split(
        linking_examples, test_size=0.2, random_state=42
    )
    
    logger.info(f"Training examples: {len(train_examples)}")
    logger.info(f"Validation examples: {len(val_examples)}")
    
    # Step 3: Create knowledge base
    logger.info("Step 3: Creating knowledge base...")
    kb_file = 'models/entity_kb.json'
    Path('models').mkdir(exist_ok=True)
    create_knowledge_base(linking_examples, kb_file)
    
    # Step 4: Train models
    logger.info("Step 4: Training models...")
    trainer = EntityLinkingTrainer()
    
    # Train classifier
    logger.info("Training BERT classifier...")
    classifier = trainer.train_classifier(train_examples, val_examples, epochs=2)
    
    # Train embeddings
    logger.info("Training sentence embeddings...")
    embedding_model = trainer.train_embeddings(train_examples, val_examples, epochs=2)
    
    logger.info("🎉 Training pipeline completed!")
    logger.info("Models saved in 'models/' directory:")
    logger.info("- nel_bert_model.pth (BERT classifier)")
    logger.info("- nel_embeddings/ (Sentence transformer)")
    logger.info("- entity_kb.json (Knowledge base)")

if __name__ == "__main__":
    main()