"""
Named Entity Linking (NEL) System
DevifyX NLP Job Assignment

A comprehensive system for linking detected entities in text to Wikipedia knowledge base.
Handles entity detection, candidate generation, disambiguation, and confidence scoring.
"""

import spacy
import requests
import json
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging
from functools import lru_cache
import time
from difflib import SequenceMatcher
import numpy as np
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """Represents a detected named entity"""
    text: str
    label: str
    start: int
    end: int
    context: str

@dataclass
class Candidate:
    """Represents a candidate entity from knowledge base"""
    title: str
    page_id: str
    description: str
    url: str
    score: float = 0.0

@dataclass
class LinkedEntity:
    """Represents a successfully linked entity"""
    entity: Entity
    candidate: Candidate
    confidence: float

class WikipediaAPI:
    """Wikipedia API integration for entity lookup"""
    
    BASE_URL = "https://en.wikipedia.org/api/rest_v1"
    SEARCH_URL = "https://en.wikipedia.org/w/api.php"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'NEL-System/1.0 (https://example.com/contact)'
        })
    
    @lru_cache(maxsize=1000)
    def search_entities(self, query: str, limit: int = 10) -> List[Dict]:
        """Search Wikipedia for potential entity matches"""
        try:
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': query,
                'srlimit': limit,
                'srprop': 'snippet'
            }
            
            response = self.session.get(self.SEARCH_URL, params=params)
            response.raise_for_status()
            
            data = response.json()
            if 'query' in data and 'search' in data['query']:
                return data['query']['search']
            return []
            
        except Exception as e:
            logger.error(f"Error searching Wikipedia for '{query}': {e}")
            return []
    
    @lru_cache(maxsize=500)
    def get_page_summary(self, title: str) -> Optional[Dict]:
        """Get page summary from Wikipedia"""
        try:
            # Clean title for URL
            clean_title = title.replace(' ', '_')
            url = f"{self.BASE_URL}/page/summary/{clean_title}"
            
            response = self.session.get(url)
            if response.status_code == 200:
                return response.json()
            return None
            
        except Exception as e:
            logger.error(f"Error getting summary for '{title}': {e}")
            return None

class EntityDetector:
    """Named Entity Recognition using spaCy"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.error(f"Model {model_name} not found. Please install it using:")
            logger.error(f"python -m spacy download {model_name}")
            raise
    
    def detect_entities(self, text: str) -> List[Entity]:
        """Detect named entities in text"""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            # Filter for relevant entity types
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'EVENT', 'PRODUCT']:
                # Get context (surrounding words)
                context_start = max(0, ent.start - 10)
                context_end = min(len(doc), ent.end + 10)
                context = doc[context_start:context_end].text
                
                entity = Entity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    context=context
                )
                entities.append(entity)
        
        return entities

class CandidateGenerator:
    """Generate candidate entities from knowledge base"""
    
    def __init__(self, wikipedia_api: WikipediaAPI):
        self.wiki_api = wikipedia_api
    
    def generate_candidates(self, entity: Entity) -> List[Candidate]:
        """Generate candidate entities for a given entity mention"""
        candidates = []
        
        # Search variations of the entity text
        search_terms = self._generate_search_terms(entity.text)
        
        for term in search_terms:
            search_results = self.wiki_api.search_entities(term, limit=5)
            
            for result in search_results:
                # Get detailed information
                summary = self.wiki_api.get_page_summary(result['title'])
                
                if summary:
                    candidate = Candidate(
                        title=result['title'],
                        page_id=str(result['pageid']),
                        description=summary.get('extract', ''),
                        url=summary.get('content_urls', {}).get('desktop', {}).get('page', '')
                    )
                    candidates.append(candidate)
        
        # Remove duplicates based on page_id
        seen_ids = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate.page_id not in seen_ids:
                seen_ids.add(candidate.page_id)
                unique_candidates.append(candidate)
        
        return unique_candidates[:10]  # Limit to top 10 candidates
    
    def _generate_search_terms(self, entity_text: str) -> List[str]:
        """Generate variations of entity text for searching"""
        terms = [entity_text]
        
        # Add without punctuation
        clean_text = re.sub(r'[^\w\s]', '', entity_text)
        if clean_text != entity_text:
            terms.append(clean_text)
        
        # Add title case
        title_case = entity_text.title()
        if title_case != entity_text:
            terms.append(title_case)
        
        return list(set(terms))

class EntityDisambiguator:
    """Disambiguate entities using context and similarity"""
    
    def __init__(self):
        # Load sentence transformer for semantic similarity
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
            self.sentence_model = None
    
    def disambiguate(self, entity: Entity, candidates: List[Candidate]) -> Optional[LinkedEntity]:
        """Select the best candidate for an entity"""
        if not candidates:
            return None
        
        # Score each candidate
        for candidate in candidates:
            candidate.score = self._calculate_score(entity, candidate)
        
        # Sort by score (descending)
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        best_candidate = candidates[0]
        
        # Calculate confidence based on score difference
        confidence = self._calculate_confidence(candidates)
        
        # Only link if confidence is above threshold
        if confidence > 0.3:
            return LinkedEntity(
                entity=entity,
                candidate=best_candidate,
                confidence=confidence
            )
        
        return None
    
    def _calculate_score(self, entity: Entity, candidate: Candidate) -> float:
        """Calculate similarity score between entity and candidate"""
        score = 0.0
        
        # 1. Exact string match
        if entity.text.lower() == candidate.title.lower():
            score += 0.4
        
        # 2. String similarity
        string_sim = SequenceMatcher(None, entity.text.lower(), candidate.title.lower()).ratio()
        score += string_sim * 0.3
        
        # 3. Semantic similarity (if available)
        if self.sentence_model and candidate.description:
            try:
                entity_embedding = self.sentence_model.encode([entity.context])
                candidate_embedding = self.sentence_model.encode([candidate.description])
                
                # Cosine similarity
                semantic_sim = np.dot(entity_embedding[0], candidate_embedding[0]) / (
                    np.linalg.norm(entity_embedding[0]) * np.linalg.norm(candidate_embedding[0])
                )
                score += semantic_sim * 0.3
            except Exception:
                pass
        
        return score
    
    def _calculate_confidence(self, candidates: List[Candidate]) -> float:
        """Calculate confidence based on score distribution"""
        if len(candidates) < 2:
            return candidates[0].score if candidates else 0.0
        
        best_score = candidates[0].score
        second_best_score = candidates[1].score
        
        # Confidence is higher when there's a clear winner
        score_diff = best_score - second_best_score
        confidence = min(best_score + score_diff * 0.5, 1.0)
        
        return confidence

class NamedEntityLinker:
    """Main NEL system orchestrating all components"""
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.entity_detector = EntityDetector(spacy_model)
        self.wikipedia_api = WikipediaAPI()
        self.candidate_generator = CandidateGenerator(self.wikipedia_api)
        self.disambiguator = EntityDisambiguator()
        
        logger.info("Named Entity Linker initialized successfully")
    
    def link_entities(self, text: str) -> List[LinkedEntity]:
        """Main method to link entities in text"""
        logger.info(f"Processing text of length {len(text)}")
        
        # Step 1: Detect entities
        entities = self.entity_detector.detect_entities(text)
        logger.info(f"Detected {len(entities)} entities")
        
        linked_entities = []
        
        # Step 2: Process each entity
        for entity in entities:
            logger.info(f"Processing entity: {entity.text} ({entity.label})")
            
            # Step 3: Generate candidates
            candidates = self.candidate_generator.generate_candidates(entity)
            logger.info(f"Generated {len(candidates)} candidates")
            
            # Step 4: Disambiguate
            linked_entity = self.disambiguator.disambiguate(entity, candidates)
            
            if linked_entity:
                linked_entities.append(linked_entity)
                logger.info(f"Linked '{entity.text}' to '{linked_entity.candidate.title}' "
                          f"(confidence: {linked_entity.confidence:.3f})")
            else:
                logger.info(f"Could not link entity '{entity.text}'")
        
        return linked_entities
    
    def process_batch(self, texts: List[str]) -> List[List[LinkedEntity]]:
        """Process multiple texts"""
        results = []
        for i, text in enumerate(texts):
            logger.info(f"Processing text {i+1}/{len(texts)}")
            result = self.link_entities(text)
            results.append(result)
        return results

# Utility functions for output formatting
def format_results(linked_entities: List[LinkedEntity]) -> Dict:
    """Format results for JSON output"""
    results = {
        "entities": [],
        "statistics": {
            "total_entities": len(linked_entities),
            "linked_entities": len(linked_entities),
            "average_confidence": np.mean([le.confidence for le in linked_entities]) if linked_entities else 0
        }
    }
    
    for le in linked_entities:
        entity_info = {
            "text": le.entity.text,
            "type": le.entity.label,
            "position": {
                "start": le.entity.start,
                "end": le.entity.end
            },
            "linked_to": {
                "title": le.candidate.title,
                "page_id": le.candidate.page_id,
                "description": le.candidate.description[:200] + "..." if len(le.candidate.description) > 200 else le.candidate.description,
                "url": le.candidate.url
            },
            "confidence": round(le.confidence, 3)
        }
        results["entities"].append(entity_info)
    
    return results

def save_results(results: Dict, filename: str):
    """Save results to JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

# Example usage and testing
if __name__ == "__main__":
    # Initialize the NEL system
    nel = NamedEntityLinker()
    
    # Sample text for testing
    sample_text = """
    Barack Obama was born in Honolulu, Hawaii. He served as the 44th President of the United States.
    Obama graduated from Harvard Law School and worked as a community organizer in Chicago.
    Microsoft Corporation was founded by Bill Gates and Paul Allen in 1975.
    The company is headquartered in Redmond, Washington.
    """
    
    # Process the text
    results = nel.link_entities(sample_text)
    
    # Format and display results
    formatted_results = format_results(results)
    print(json.dumps(formatted_results, indent=2))
    
    # Save results
    save_results(formatted_results, "sample_output.json")