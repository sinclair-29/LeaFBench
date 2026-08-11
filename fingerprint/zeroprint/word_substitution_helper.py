import os
import random
import pandas as pd
import numpy as np
import logging
import re
import nltk
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import multiprocessing as mp
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import gensim.downloader as api


def ensure_nltk_resources(nltk_data_path=None):
    """Validate NLTK resources without attempting a network download."""
    if nltk_data_path:
        nltk_data_path = os.path.expanduser(nltk_data_path)
        if nltk_data_path not in nltk.data.path:
            nltk.data.path.insert(0, nltk_data_path)

    resources = {
        'punkt': 'tokenizers/punkt',
        'punkt_tab': 'tokenizers/punkt_tab',
        'stopwords': 'corpora/stopwords',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng': 'taggers/averaged_perceptron_tagger_eng',
    }
    missing = []
    for name, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            missing.append(name)

    if missing:
        location = nltk_data_path or 'one of NLTK\'s configured data paths'
        raise FileNotFoundError(
            f"Missing offline NLTK resources in {location}: {', '.join(missing)}. "
            "Download them on a connected machine and transfer the NLTK data directory."
        )


class WordSubstitutionHelper:
    """
    Helper class for word substitution-based data augmentation.
    Uses word vectors to find similar words and replace them in the original text.
    """
    
    def __init__(self, config):
        self.config = config
        self.word_sub_config = config.get('word_substitution_config', {})
        ensure_nltk_resources(config.get('nltk_data_path'))
        
        # Configuration parameters
        self.word_vector_model = self.word_sub_config.get('word_vector_model', 'glove')
        self.word_vector_cache_dir = self.word_sub_config.get('word_vector_cache_dir')
        self.k_words_to_replace = self.word_sub_config.get('k_words_to_replace', 3)
        self.top_m_neighbors = self.word_sub_config.get('top_m_neighbors', 10)
        self.n_augmented_samples = self.word_sub_config.get('n_augmented_samples', 5)
        self.substitution_queries_path = self.word_sub_config.get('substitution_queries_path', './zeroprint_substitution_queries.csv')
        self.resample = config.get('resample', False)
        
        # Initialize components
        self.word_vectors = None
        self.stop_words = set(stopwords.words('english'))
        
        # Target POS tags for substitution (nouns, verbs, adjectives)
        self.target_pos_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS'}
        
    @property
    def logger(self):
        """Get logger instance for this class."""
        return logging.getLogger(__name__)
    
    def load_word_vectors(self):
        """Load word vectors based on the specified model."""
        if self.word_vectors is not None:
            return
            
        print(f"Loading word vector model: {self.word_vector_model}")
        self.logger.info(f"Loading word vector model: {self.word_vector_model}")

        if self.word_vector_cache_dir:
            cache_dir = os.path.abspath(
                os.path.expanduser(self.word_vector_cache_dir)
            )
            model_names = {
                'glove': 'glove-wiki-gigaword-100',
                'word2vec': 'word2vec-google-news-300',
                'fasttext': 'fasttext-wiki-news-subwords-300',
            }
            model_dir = os.path.join(cache_dir, model_names.get(self.word_vector_model, ''))
            if not os.path.isdir(model_dir):
                raise FileNotFoundError(
                    f"Missing offline word-vector resource: {model_dir}. "
                    "Transfer the complete Gensim downloader cache before running ZeroPrint."
                )
            # Gensim's downloaded dataset loader consults GENSIM_DATA_DIR
            # independently of downloader.BASE_DIR. Keep both in sync with the
            # explicit LeaFBench configuration so callers never need to export
            # an environment variable before running ZeroPrint.
            os.environ['GENSIM_DATA_DIR'] = cache_dir
            api.BASE_DIR = cache_dir
        
        try:
            if self.word_vector_model == 'glove':
                self._load_glove_vectors()
            elif self.word_vector_model == 'word2vec':
                self._load_word2vec_vectors()
            elif self.word_vector_model == 'fasttext':
                self._load_fasttext_vectors()
            else:
                raise ValueError(f"Unsupported word vector model: {self.word_vector_model}")
                
            print(f"Word vector model loaded successfully: {len(self.word_vectors)} words")
            self.logger.info(f"Word vector model loaded: {len(self.word_vectors)} words")
            
            # Build vector matrix for efficient similarity computation
            print("Building vector matrix for efficient similarity computation...")
            self._build_vector_matrix()
            print("Vector matrix built successfully")
            
        except Exception as e:
            print(f"Error loading word vectors: {e}")
            self.logger.error(f"Error loading word vectors: {e}")
            raise
    
    def _load_glove_vectors(self):
        """Load GloVe word vectors using gensim."""
        # Load pre-trained GloVe model via gensim
        glove_model = api.load('glove-wiki-gigaword-100')
        self.word_vectors = {}
        
        for word in glove_model.key_to_index:
            self.word_vectors[word] = glove_model[word]
                        
    
    def _load_word2vec_vectors(self):
        """Load Word2Vec vectors."""
        # Load pre-trained Word2Vec model
        w2v_model = api.load('word2vec-google-news-300')
        self.word_vectors = {}
        
        for word in w2v_model.key_to_index:
            self.word_vectors[word] = w2v_model[word]
    
    def _load_fasttext_vectors(self):
        """Load FastText vectors."""
        # Load pre-trained FastText model
        ft_model = api.load('fasttext-wiki-news-subwords-300')
        self.word_vectors = {}
        
        for word in ft_model.key_to_index:
            self.word_vectors[word] = ft_model[word]
                

    
    def find_similar_words(self, word, top_k=None):
        """
        Find the most similar words to the given word using vectorized operations.
        
        Args:
            word (str): The input word
            top_k (int): Number of similar words to return
            
        Returns:
            list: List of similar words
        """
        if top_k is None:
            top_k = self.top_m_neighbors
            
        word_lower = word.lower()
        
        if word_lower not in self.word_vectors:
            return []
        
        try:
            # Get target vector
            target_vector = self.word_vectors[word_lower]
            
            # Create arrays for vectorized computation
            if not hasattr(self, '_vector_matrix'):
                self._build_vector_matrix()
            
            # Find the index of the target word
            if word_lower not in self._word_to_index:
                return []
            
            target_idx = self._word_to_index[word_lower]
            
            # Calculate cosine similarities using vectorized operations
            # Exclude the target word itself by masking
            mask = np.ones(len(self._words), dtype=bool)
            mask[target_idx] = False
            
            # Compute cosine similarities
            target_norm = np.linalg.norm(target_vector)
            if target_norm == 0:
                return []
            
            # Normalize target vector
            target_normalized = target_vector / target_norm
            
            # Compute similarities with all other vectors
            similarities = np.dot(self._vector_matrix, target_normalized)
            
            # Apply mask to exclude the target word
            similarities = similarities[mask]
            masked_words = [self._words[i] for i in range(len(self._words)) if mask[i]]
            
            # Get top_k most similar words
            top_indices = np.argsort(similarities)[::-1][:top_k]
            return [masked_words[i] for i in top_indices]
            
        except Exception as e:
            print(f"Error finding similar words for '{word}': {e}")
            return []
    
    def _build_vector_matrix(self):
        """Build matrix representation of word vectors for vectorized operations."""
        if not self.word_vectors:
            return
            
        self._words = list(self.word_vectors.keys())
        self._word_to_index = {word: i for i, word in enumerate(self._words)}
        
        # Stack all vectors into a matrix
        vectors = [self.word_vectors[word] for word in self._words]
        self._vector_matrix = np.stack(vectors, axis=0)
        
        # Normalize all vectors for cosine similarity computation
        norms = np.linalg.norm(self._vector_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        self._vector_matrix = self._vector_matrix / norms
    
    def get_replaceable_words(self, text):
        """
        Get words that can be replaced (nouns, verbs, adjectives, not stopwords).
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of tuples (word, position) for replaceable words
        """
        # Tokenize and get POS tags
        tokens = word_tokenize(text.lower())
        pos_tags = pos_tag(tokens)
        
        replaceable_words = []
        
        for i, (word, pos) in enumerate(pos_tags):
            # Check if word is suitable for replacement
            if (pos in self.target_pos_tags and 
                word not in self.stop_words and 
                len(word) > 2 and 
                word.isalpha()):
                replaceable_words.append((word, i))
        
        return replaceable_words
    
    def substitute_words_in_text(self, text, num_substitutions=None):
        """
        Substitute words in the text with similar words.
        
        Args:
            text (str): Original text
            num_substitutions (int): Number of words to substitute
            
        Returns:
            str: Text with substituted words
        """
        if num_substitutions is None:
            num_substitutions = self.k_words_to_replace
        
        # Get replaceable words
        replaceable_words = self.get_replaceable_words(text)
        
        if len(replaceable_words) == 0:
            return text
        
        # Randomly select words to replace
        num_to_replace = min(num_substitutions, len(replaceable_words))
        words_to_replace = random.sample(replaceable_words, num_to_replace)
        
        # Tokenize original text to preserve case and punctuation
        tokens = word_tokenize(text)
        
        # Create a mapping from lowercase to original case
        token_case_map = {}
        for i, token in enumerate(tokens):
            token_case_map[i] = token
        
        # Perform substitutions
        for word, pos in words_to_replace:
            similar_words = self.find_similar_words(word, self.top_m_neighbors)
            
            if similar_words:
                # Randomly choose one similar word
                replacement = random.choice(similar_words)
                
                # Try to match the original case
                if pos < len(tokens):
                    original_token = tokens[pos]
                    if original_token.isupper():
                        replacement = replacement.upper()
                    elif original_token.istitle():
                        replacement = replacement.capitalize()
                    
                    tokens[pos] = replacement
        
        # Reconstruct text
        return ' '.join(tokens)
    
    def _process_single_query(self, args):
        """
        Process a single query and generate substituted versions.
        
        Args:
            args: Tuple of (query_idx, original_query, n_augmented_samples)
            
        Returns:
            list: List of substituted query dictionaries
        """
        query_idx, original_query, n_augmented_samples = args
        results = []
        
        for sub_idx in range(n_augmented_samples):
            try:
                substituted_query = self.substitute_words_in_text(original_query, self.k_words_to_replace)
                
                results.append({
                    'original_query': original_query,
                    'perturbed_query': substituted_query,  # Use same column name for compatibility
                    'original_index': query_idx,
                    'perturbed_version': sub_idx
                })
                
            except Exception as e:
                print(f"Error substituting words in query '{original_query}' (version {sub_idx}): {e}")
                raise
                
        return results
    
    def generate_substituted_queries(self, original_queries):
        """
        Generate substituted versions of queries using word vector similarity with parallel processing.
        
        Args:
            original_queries (list): List of original query strings
            
        Returns:
            list: List of dictionaries containing original and substituted queries
        """
        # Check if substituted queries already exist and don't need resampling
        if os.path.exists(self.substitution_queries_path) and not self.resample:
            print("Loading existing substituted queries...")
            self.logger.info(f"Loading existing substituted queries from {self.substitution_queries_path}")
            df = pd.read_csv(self.substitution_queries_path)
            return df.to_dict('records')
        
        # Load word vectors
        self.load_word_vectors()
        
        print(f"Generating {self.n_augmented_samples} substituted versions for each of {len(original_queries)} queries...")
        self.logger.info(f"Starting word substitution: {len(original_queries)} queries × {self.n_augmented_samples} substitutions")
        self.logger.info(f"Word vector model: {self.word_vector_model}")
        self.logger.info(f"Words to replace per query: {self.k_words_to_replace}")
        self.logger.info(f"Top neighbors to consider: {self.top_m_neighbors}")
        
        # Prepare arguments for parallel processing
        args_list = [(query_idx, original_query, self.n_augmented_samples) 
                     for query_idx, original_query in enumerate(original_queries)]
        
        # Use parallel processing with ThreadPoolExecutor (better for I/O bound tasks)
        # We use threads instead of processes to avoid pickling issues with the word vectors
        max_workers_config = self.word_sub_config.get('max_workers', 8)
        max_workers = min(mp.cpu_count(), len(original_queries), max_workers_config)  # Limit to reasonable number
        print(f"Using {max_workers} workers for parallel processing...")
        
        all_substituted_data = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Process queries in batches to show progress
            batch_size = max(1, len(original_queries) // 10)  # 10 progress updates
            
            for i in range(0, len(args_list), batch_size):
                batch_args = args_list[i:i + batch_size]
                
                # Submit batch to executor
                future_results = executor.map(self._process_single_query, batch_args)
                
                # Collect results from this batch
                for results in future_results:
                    all_substituted_data.extend(results)
                
                print(f"Processed {min(i + batch_size, len(original_queries))}/{len(original_queries)} queries")
        
        # Save substituted queries
        os.makedirs(os.path.dirname(self.substitution_queries_path), exist_ok=True)
        df = pd.DataFrame(all_substituted_data)
        df.to_csv(self.substitution_queries_path, index=False)
        
        self.logger.info(f"Word substitution completed: generated {len(all_substituted_data)} total substituted queries")
        self.logger.info(f"Saved substituted queries to {self.substitution_queries_path}")
        
        print(f"Generated {len(all_substituted_data)} total substituted queries")
        return all_substituted_data
