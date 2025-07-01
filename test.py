"""
Comprehensive Test Suite for Named Entity Linking System
Tests all core functionality and edge cases
"""

import unittest
import json
import time
from unittest.mock import patch, MagicMock
import sys
import os

# Add the current directory to path to import nel_system
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nel_system import (
    NamedEntityLinker, Entity, Candidate, LinkedEntity,
    EntityDetector, WikipediaAPI, CandidateGenerator, EntityDisambiguator,
    format_results, save_results
)

class TestEntityDetector(unittest.TestCase):
    """Test cases for EntityDetector"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        try:
            cls.detector = EntityDetector()
        except OSError:
            cls.skipTest(cls, "spaCy model not available")
    
    def test_detect_basic_entities(self):
        """Test detection of basic named entities"""
        text = "Barack Obama was born in Hawaii."
        entities = self.detector.detect_entities(text)
        
        self.assertGreater(len(entities), 0)
        
        # Check if Barack Obama is detected as PERSON
        person_entities = [e for e in entities if e.label == 'PERSON']
        self.assertGreater(len(person_entities), 0)
        
        # Check if Hawaii is detected as location
        location_entities = [e for e in entities if e.label in ['GPE', 'LOC']]
        self.assertGreater(len(location_entities), 0)
    
    def test_detect_organizations(self):
        """Test detection of organizations"""
        text = "Microsoft Corporation is a technology company."
        entities = self.detector.detect_entities(text)
        
        org_entities = [e for e in entities if e.label == 'ORG']
        self.assertGreater(len(org_entities), 0)
        
        # Check if Microsoft is detected
        microsoft_found = any('Microsoft' in e.text for e in org_entities)
        self.assertTrue(microsoft_found)
    
    def test_empty_text(self):
        """Test handling of empty text"""
        entities = self.detector.detect_entities("")
        self.assertEqual(len(entities), 0)
    
    def test_special_characters(self):
        """Test handling of special characters"""
        text = "Dr. Martin Luther King Jr. lived in the U.S.A."
        entities = self.detector.detect_entities(text)
        
        # Should still detect entities despite punctuation
        self.assertGreater(len(entities), 0)

class TestWikipediaAPI(unittest.TestCase):
    """Test cases for WikipediaAPI"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.api = WikipediaAPI()
    
    def test_search_entities(self):
        """Test Wikipedia search functionality"""
        results = self.api.search_entities("Barack Obama")
        
        self.assertIsInstance(results, list)
        if results:  # If we get results (internet dependent)
            self.assertGreater(len(results), 0)
            self.assertIn('title', results[0])
            self.assertIn('pageid', results[0])
    
    def test_get_page_summary(self):
        """Test getting page summary"""
        summary = self.api.get_page_summary("Barack Obama")
        
        if summary:  # Internet dependent
            self.assertIn('extract', summary)
            self.assertIn('title', summary)
    
    def test_invalid_search(self):
        """Test search with invalid query"""
        results = self.api.search_entities("xyzinvalidquery123")
        self.assertIsInstance(results, list)
        # Should return empty list or minimal results

class TestCandidateGenerator(unittest.TestCase):
    """Test cases for CandidateGenerator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.api = WikipediaAPI()
        self.generator = CandidateGenerator(self.api)
    
    def test_generate_candidates(self):
        """Test candidate generation"""
        entity = Entity(
            text="Obama",
            label="PERSON",
            start=0,
            end=5,
            context="Obama was president"
        )
        
        candidates = self.generator.generate_candidates(entity)
        
        self.assertIsInstance(candidates, list)
        self.assertLessEqual(len(candidates), 10)  # Should limit to 10
        
        if candidates:
            self.assertIsInstance(candidates[0], Candidate)
            self.assertTrue(hasattr(candidates[0], 'title'))
            self.assertTrue(hasattr(candidates[0], 'page_id'))
    
    def test_search_terms_generation(self):
        """Test generation of search terms"""
        terms = self.generator._generate_search_terms("Dr. Obama")
        
        self.assertIn("Dr. Obama", terms)
        self.assertIn("Dr Obama", terms)  # Without punctuation
        self.assertGreater(len(terms), 1)

class TestEntityDisambiguator(unittest.TestCase):
    """Test cases for EntityDisambiguator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.disambiguator = EntityDisambiguator()
    
    def test_calculate_score(self):
        """Test scoring calculation"""
        entity = Entity(
            text="Obama",
            label="PERSON",
            start=0,
            end=5,
            context="Barack Obama was president"
        )
        
        candidate = Candidate(
            title="Barack Obama",
            page_id="123",
            description="44th President of the United States",
            url="http://example.com"
        )
        
        score = self.disambiguator._calculate_score(entity, candidate)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_exact_match_scoring(self):
        """Test exact match gets high score"""
        entity = Entity(
            text="Barack Obama",
            label="PERSON",
            start=0,
            end=12,
            context="Barack Obama was president"
        )
        
        exact_candidate = Candidate(
            title="Barack Obama",
            page_id="123",
            description="President",
            url="http://example.com"
        )
        
        different_candidate = Candidate(
            title="Michelle Obama",
            page_id="456",
            description="First Lady",
            url="http://example.com"
        )
        
        exact_score = self.disambiguator._calculate_score(entity, exact_candidate)
        different_score = self.disambiguator._calculate_score(entity, different_candidate)
        
        self.assertGreater(exact_score, different_score)
    
    def test_confidence_calculation(self):
        """Test confidence calculation"""
        candidates = [
            Candidate("Title1", "1", "Desc1", "url1"),
            Candidate("Title2", "2", "Desc2", "url2")
        ]
        
        candidates[0].score = 0.8
        candidates[1].score = 0.3
        
        confidence = self.disambiguator._calculate_confidence(candidates)
        
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_no_candidates(self):
        """Test disambiguation with no candidates"""
        entity = Entity("Test", "PERSON", 0, 4, "Test context")
        result = self.disambiguator.disambiguate(entity, [])
        
        self.assertIsNone(result)

class TestNamedEntityLinker(unittest.TestCase):
    """Test cases for main NEL system"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        try:
            cls.nel = NamedEntityLinker()
        except OSError:
            cls.skipTest(cls, "spaCy model not available")
    
    def test_link_entities_basic(self):
        """Test basic entity linking"""
        text = "Barack Obama was born in Hawaii."
        results = self.nel.link_entities(text)
        
        self.assertIsInstance(results, list)
        
        if results:  # Internet dependent
            self.assertIsInstance(results[0], LinkedEntity)
            self.assertTrue(hasattr(results[0], 'entity'))
            self.assertTrue(hasattr(results[0], 'candidate'))
            self.assertTrue(hasattr(results[0], 'confidence'))
    
    def test_batch_processing(self):
        """Test batch processing of multiple texts"""
        texts = [
            "Barack Obama was president.",
            "Microsoft is a company."
        ]
        
        results = self.nel.process_batch(texts)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        
        for batch_result in results:
            self.assertIsInstance(batch_result, list)
    
    def test_empty_text_handling(self):
        """Test handling of empty text"""
        results = self.nel.link_entities("")
        self.assertEqual(len(results), 0)
    
    def test_special_characters_handling(self):
        """Test handling of text with special characters"""
        text = "Dr. John F. Kennedy was U.S. President!!!"
        results = self.nel.link_entities(text)
        
        # Should not crash with special characters
        self.assertIsInstance(results, list)

class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions"""
    
    def test_format_results_empty(self):
        """Test formatting empty results"""
        results = format_results([])
        
        self.assertIn('entities', results)
        self.assertIn('statistics', results)
        self.assertEqual(len(results['entities']), 0)
        self.assertEqual(results['statistics']['total_entities'], 0)
    
    def test_format_results_with_data(self):
        """Test formatting results with data"""
        # Create mock linked entity
        entity = Entity("Obama", "PERSON", 0, 5, "Obama was president")
        candidate = Candidate("Barack Obama", "123", "President", "http://example.com")
        linked_entity = LinkedEntity(entity, candidate, 0.95)
        
        results = format_results([linked_entity])
        
        self.assertIn('entities', results)
        self.assertEqual(len(results['entities']), 1)
        
        entity_data = results['entities'][0]
        self.assertEqual(entity_data['text'], 'Obama')
        self.assertEqual(entity_data['type'], 'PERSON')
        self.assertEqual(entity_data['confidence'], 0.95)
        self.assertIn('linked_to', entity_data)
    
    def test_save_results(self):
        """Test saving results to file"""
        test_data = {'test': 'data'}
        filename = 'test_output.json'
        
        try:
            save_results(test_data, filename)
            
            # Verify file was created and contains correct data
            with open(filename, 'r') as f:
                loaded_data = json.load(f)
            
            self.assertEqual(loaded_data, test_data)
            
        finally:
            # Clean up
            if os.path.exists(filename):
                os.remove(filename)

class TestPerformance(unittest.TestCase):
    """Performance and stress tests"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        try:
            cls.nel = NamedEntityLinker()
        except OSError:
            cls.skipTest(cls, "spaCy model not available")
    
    def test_processing_speed(self):
        """Test processing speed"""
        text = """
        Barack Obama served as the 44th President of the United States.
        He was born in Honolulu, Hawaii, and later moved to Chicago.
        Microsoft Corporation was founded by Bill Gates and Paul Allen.
        The company is headquartered in Redmond, Washington.
        """
        
        start_time = time.time()
        results = self.nel.link_entities(text)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should process reasonably quickly (adjust threshold as needed)
        self.assertLess(processing_time, 30.0)  # 30 seconds max for this test
        
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Entities found: {len(results)}")
    
    def test_large_text_handling(self):
        """Test handling of large text"""
        # Create a large text by repeating smaller text
        base_text = "Barack Obama was president. Microsoft is a company. "
        large_text = base_text * 20  # Repeat 20 times
        
        # Should not crash or take too long
        try:
            results = self.nel.link_entities(large_text)
            self.assertIsInstance(results, list)
        except Exception as e:
            self.fail(f"Large text processing failed: {e}")

class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        try:
            cls.nel = NamedEntityLinker()
        except OSError:
            cls.skipTest(cls, "spaCy model not available")
    
    def test_ambiguous_entities(self):
        """Test handling of ambiguous entities"""
        # "Washington" could be person, state, or city
        text = "Washington was a great leader and Washington is a beautiful state."
        results = self.nel.link_entities(text)
        
        # Should handle without crashing
        self.assertIsInstance(results, list)
    
    def test_mixed_case_entities(self):
        """Test entities with mixed case"""
        text = "barack obama and MICROSOFT corporation"
        results = self.nel.link_entities(text)
        
        # Should handle case variations
        self.assertIsInstance(results, list)
    
    def test_non_english_characters(self):
        """Test handling of non-English characters"""
        text = "François Mitterrand was a French president."
        results = self.nel.link_entities(text)
        
        # Should handle accented characters
        self.assertIsInstance(results, list)
    
    def test_numbers_and_dates(self):
        """Test handling of numbers and dates"""
        text = "Obama was born on August 4, 1961 in the 50th state."
        results = self.nel.link_entities(text)
        
        # Should focus on named entities, not numbers
        self.assertIsInstance(results, list)

def run_performance_benchmark():
    """Run a comprehensive performance benchmark"""
    print("\n" + "="*50)
    print("PERFORMANCE BENCHMARK")
    print("="*50)
    
    try:
        nel = NamedEntityLinker()
        
        test_texts = [
            "Barack Obama was the 44th President of the United States.",
            "Microsoft Corporation was founded by Bill Gates and Paul Allen in Albuquerque, New Mexico.",
            "The Eiffel Tower is located in Paris, France and was designed by Gustave Eiffel.",
            "Apple Inc. is headquartered in Cupertino, California. Steve Jobs co-founded the company.",
            "Harvard University is located in Cambridge, Massachusetts near Boston."
        ]
        
        total_entities = 0
        total_time = 0
        
        for i, text in enumerate(test_texts, 1):
            print(f"\nTest {i}: Processing text of length {len(text)}")
            
            start_time = time.time()
            results = nel.link_entities(text)
            end_time = time.time()
            
            processing_time = end_time - start_time
            total_time += processing_time
            total_entities += len(results)
            
            print(f"  Entities found: {len(results)}")
            print(f"  Processing time: {processing_time:.2f}s")
            
            if results:
                avg_confidence = sum(r.confidence for r in results) / len(results)
                print(f"  Average confidence: {avg_confidence:.3f}")
        
        print(f"\n" + "-"*30)
        print(f"TOTAL STATISTICS:")
        print(f"  Total entities processed: {total_entities}")
        print(f"  Total processing time: {total_time:.2f}s")
        print(f"  Average time per entity: {total_time/max(total_entities, 1):.2f}s")
        print(f"  Entities per second: {total_entities/total_time:.2f}")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")

if __name__ == '__main__':
    print("Named Entity Linking - Test Suite")
    print("="*40)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance benchmark
    run_performance_benchmark()
    
    print("\n" + "="*40)
    print("All tests completed!")
    print("If you see failures, check:")
    print("1. Internet connection (for Wikipedia API)")
    print("2. spaCy model installation: python -m spacy download en_core_web_sm")
    print("3. All required packages installed: pip install -r requirements.txt")