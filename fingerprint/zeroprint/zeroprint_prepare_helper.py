import os
import random
import pandas as pd
import torch
import numpy as np
import logging
import re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from fingerprint.zeroprint.word_substitution_helper import WordSubstitutionHelper

DATASET_ROOT = '~/.cache/huggingface/hub/'

class ZeroPrintPrepareHelper:
    def __init__(self, config):
        self.config = config
        self.query_csv_path = config.get('query_csv_path', './zeroprint_query_samples.csv')
        self.n_samples = config.get('n_samples', 200)
        self.resample = config.get('resample', False)
        self.query_datasets = config.get('query_datasets', ['truthful_qa', 'squad', 'wikipedia'])
        self.continuation_word_limit = config.get('continuation_word_limit', 20)
        self.minimum_qa_length = config.get('minimum_qa_length', 20)  # Minimum length of queries (in words)
        
        # Embedding configuration
        self.embedding_model_name = config.get('embedding_model_name', 'Qwen/Qwen3-Embedding-4B')
        self.embedding_batch_size = config.get('embedding_batch_size', 32)
        self.embeddings_path = config.get('embeddings_path', './zeroprint_embeddings.pth')
        self.perturbed_queries_path = config.get('perturbed_queries_path', './cache/zeroprint/zeroprint_perturbed_queries.csv')
        
        # Initialize model components
        self.embedding_model = None
        
        # Initialize word substitution helper if needed
        self.word_substitution_helper = WordSubstitutionHelper(config)
        # Use word substitution specific paths
        word_sub_config = config.get('word_substitution_config', {})
        self.num_rewrites = word_sub_config.get('n_augmented_samples', 5)

    @property
    def logger(self):
        """Get logger instance for this class."""
        return logging.getLogger(__name__)

    def _is_query_length_valid(self, query):
        """
        Check if the query meets the minimum length requirement.
        
        Args:
            query (str): Query text to check
            
        Returns:
            bool: True if query length is valid, False otherwise
        """
        if not query or not query.strip():
            return False
        
        word_count = len(query.strip().split())
        return word_count >= self.minimum_qa_length

    def _extract_text_snippet(self, text, target_length=100):
        """
        Extract a meaningful text snippet from Wikipedia text.
        
        Args:
            text (str): Raw Wikipedia text
            target_length (int): Desired length of text snippet
            
        Returns:
            str: Text snippet or None if extraction fails
        """
        # Clean text by removing unwanted characters and formatting
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
        text = text.strip()
        
        # Need some buffer for selection
        min_required_length = target_length + 20
        if len(text) < min_required_length:
            return None
        
        # Try to find a good starting point that doesn't start mid-word
        max_attempts = 5
        for _ in range(max_attempts):
            start_pos = random.randint(0, len(text) - target_length)
            
            # Try to start at word boundary if possible
            if start_pos > 0 and text[start_pos - 1] != ' ':
                next_space = text.find(' ', start_pos)
                if next_space != -1 and next_space < len(text) - target_length:
                    start_pos = next_space + 1
            
            substring = text[start_pos:start_pos + target_length]
            
            # Check if substring is valid (contains meaningful content)
            min_valid_length = int(target_length * 0.9)
            min_words = max(10, target_length // 15)
            if len(substring.strip()) >= min_valid_length and len(substring.split()) >= min_words:
                return substring
        
        # Fallback: just take first target_length characters if available
        if len(text) >= target_length:
            return text[:target_length]
        
        return None

    def get_query_samples(self):
        """Get query samples for ZeroPrint fingerprinting."""
        if os.path.exists(self.query_csv_path) and not self.resample:
            df = pd.read_csv(self.query_csv_path)
            return df['query'].tolist()
        
        # Calculate samples per dataset
        samples_per_dataset = self.n_samples // len(self.query_datasets)
        remaining_samples = self.n_samples % len(self.query_datasets)
        
        all_queries = []
        
        for i, dataset_name in enumerate(self.query_datasets):
            # Add one extra sample for first few datasets if there's remainder
            current_samples = samples_per_dataset + (1 if i < remaining_samples else 0)
            queries = self._load_dataset_queries(dataset_name, current_samples)
            all_queries.extend(queries)
        
        # Shuffle all queries
        random.shuffle(all_queries)
        os.makedirs(os.path.dirname(self.query_csv_path), exist_ok=True)
        pd.DataFrame({'query': all_queries}).to_csv(self.query_csv_path, index=False)
        return all_queries
    
    def _load_dataset_queries(self, dataset_name, num_samples):
        """Load queries from a specific dataset using reservoir sampling for random selection."""
        reservoir = []  # Store selected samples
        total_seen = 0  # Count of total valid samples seen
        max_attempts = num_samples * 5000  # Maximum attempts to prevent infinite loops
        
        try:
            if dataset_name == 'truthful_qa':
                ds = load_dataset('truthful_qa', 'generation', split='validation', streaming=True, trust_remote_code=True)
                for i, x in enumerate(ds):
                    if i >= max_attempts:
                        break
                    query = x.get('question', '')
                    if query and self._is_query_length_valid(query):
                        total_seen += 1
                        # Reservoir sampling algorithm
                        if len(reservoir) < num_samples:
                            reservoir.append(query)
                        else:
                            # Replace with probability num_samples/total_seen
                            j = random.randint(0, total_seen - 1)
                            if j < num_samples:
                                reservoir[j] = query
                        
            elif dataset_name == 'squad':
                ds = load_dataset('squad', split='validation', streaming=True, trust_remote_code=True)
                for i, x in enumerate(ds):
                    if i >= max_attempts:
                        break
                    query = x.get('question', '')
                    if query and self._is_query_length_valid(query):
                        total_seen += 1
                        if len(reservoir) < num_samples:
                            reservoir.append(query)
                        else:
                            j = random.randint(0, total_seen - 1)
                            if j < num_samples:
                                reservoir[j] = query
                        
            elif dataset_name == 'metaeval/reclor':
                ds = load_dataset('metaeval/reclor', split='validation', streaming=True, trust_remote_code=True)
                for i, x in enumerate(ds):
                    if i >= max_attempts:
                        break
                    query = x.get('question', '')
                    if query and self._is_query_length_valid(query):
                        total_seen += 1
                        if len(reservoir) < num_samples:
                            reservoir.append(query)
                        else:
                            j = random.randint(0, total_seen - 1)
                            if j < num_samples:
                                reservoir[j] = query

            elif dataset_name == 'ucinlp/drop':
                ds = load_dataset('ucinlp/drop', split='validation', streaming=True, trust_remote_code=True)
                for i, x in enumerate(ds):
                    if i >= max_attempts:
                        break
                    query = x.get('question', '')
                    if query and self._is_query_length_valid(query):
                        total_seen += 1
                        if len(reservoir) < num_samples:
                            reservoir.append(query)
                        else:
                            j = random.randint(0, total_seen - 1)
                            if j < num_samples:
                                reservoir[j] = query
                        
            elif dataset_name == 'hendrycks/ethics':
                # Ethics dataset has multiple subsets, use 'commonsense' as default
                ds = load_dataset('hendrycks/ethics', 'commonsense', split='test', streaming=True, trust_remote_code=True)
                for i, x in enumerate(ds):
                    if i >= max_attempts:
                        break
                    input_text = x.get('input', '')
                    if input_text:
                        query = f"Is this statement ethical: {input_text}"
                        if self._is_query_length_valid(query):
                            total_seen += 1
                            if len(reservoir) < num_samples:
                                reservoir.append(query)
                            else:
                                j = random.randint(0, total_seen - 1)
                                if j < num_samples:
                                    reservoir[j] = query
                            
            elif dataset_name == 'allenai/social_i_qa':
                ds = load_dataset('allenai/social_i_qa', split='validation', streaming=True, trust_remote_code=True)
                for i, x in enumerate(ds):
                    if i >= max_attempts:
                        break
                    query = x.get('question', '')
                    if query and self._is_query_length_valid(query):
                        total_seen += 1
                        if len(reservoir) < num_samples:
                            reservoir.append(query)
                        else:
                            j = random.randint(0, total_seen - 1)
                            if j < num_samples:
                                reservoir[j] = query
                        
            elif dataset_name == 'qiaojin/PubMedQA':
                ds = load_dataset('qiaojin/PubMedQA', 'pqa_labeled', split='train', streaming=True, trust_remote_code=True)
                for i, x in enumerate(ds):
                    if i >= max_attempts:
                        break
                    query = x.get('question', '')
                    if query and self._is_query_length_valid(query):
                        total_seen += 1
                        if len(reservoir) < num_samples:
                            reservoir.append(query)
                        else:
                            j = random.randint(0, total_seen - 1)
                            if j < num_samples:
                                reservoir[j] = query
                        
            elif dataset_name == 'sciq':
                ds = load_dataset('sciq', split='validation', streaming=True, trust_remote_code=True)
                for i, x in enumerate(ds):
                    if i >= max_attempts:
                        break
                    query = x.get('question', '')
                    if query and self._is_query_length_valid(query):
                        total_seen += 1
                        if len(reservoir) < num_samples:
                            reservoir.append(query)
                        else:
                            j = random.randint(0, total_seen - 1)
                            if j < num_samples:
                                reservoir[j] = query
                        
            elif dataset_name == 'openai_humaneval':
                ds = load_dataset('openai_humaneval', split='test', streaming=True, trust_remote_code=True)
                for i, x in enumerate(ds):
                    if i >= max_attempts:
                        break
                    prompt_text = x.get('prompt', '')
                    if prompt_text:
                        words = prompt_text.split()[:self.continuation_word_limit]
                        truncated_text = ' '.join(words)
                        query = f"Complete the following code: {truncated_text}"
                        if self._is_query_length_valid(query):
                            total_seen += 1
                            if len(reservoir) < num_samples:
                                reservoir.append(query)
                            else:
                                j = random.randint(0, total_seen - 1)
                                if j < num_samples:
                                    reservoir[j] = query
                            
            elif dataset_name == 'pkoerner/cam_stories':
                ds = load_dataset('pkoerner/cam_stories', split='val', streaming=True, trust_remote_code=True)
                for i, x in enumerate(ds):
                    if i >= max_attempts:
                        break
                    story_text = x.get('story', '')
                    if story_text:
                        words = story_text.split()[:self.continuation_word_limit]
                        truncated_text = ' '.join(words)
                        query = f"Continue the following story: {truncated_text}"
                        if self._is_query_length_valid(query):
                            total_seen += 1
                            if len(reservoir) < num_samples:
                                reservoir.append(query)
                            else:
                                j = random.randint(0, total_seen - 1)
                                if j < num_samples:
                                    reservoir[j] = query
                            
            elif dataset_name == 'wikipedia':
                try:
                    # Set cache directory explicitly
                    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
                    
                    ds = None
                    print(f"Attempting to load Wikipedia dataset from cache: {cache_dir}")
                    
                    # Try different approaches to load Wikipedia dataset
                    try:
                        # First try loading from cache with explicit cache_dir
                        ds = load_dataset(
                            "wikipedia", 
                            "20220301.en", 
                            split="train", 
                            streaming=True, 
                            trust_remote_code=True,
                            cache_dir=cache_dir
                        )
                        print("Successfully loaded Wikipedia dataset from cache")
                    except Exception as e1:
                        print(f"Failed to load wikipedia 20220301.en from cache: {e1}")
                        
                        # Try loading non-streaming first to check if data exists
                        try:
                            print("Trying non-streaming mode to verify dataset...")
                            ds_check = load_dataset(
                                "wikipedia", 
                                "20220301.en", 
                                split="train", 
                                streaming=False, 
                                trust_remote_code=True,
                                cache_dir=cache_dir
                            )
                            print(f"Non-streaming dataset loaded successfully: {len(ds_check)} samples")
                            
                            # Now try streaming again
                            ds = load_dataset(
                                "wikipedia", 
                                "20220301.en", 
                                split="train", 
                                streaming=True, 
                                trust_remote_code=True,
                                cache_dir=cache_dir
                            )
                            print("Streaming mode now working")
                            
                        except Exception as e2:
                            print(f"Non-streaming also failed: {e2}")
                            ds = None
                    
                    if ds is not None:
                        print(f"Processing Wikipedia dataset with max_attempts={max_attempts}")
                        for i, x in enumerate(ds):
                            if i >= max_attempts:
                                print(f"Reached max attempts ({max_attempts}), collected {len(reservoir)} samples")
                                break
                            text = x.get('text', '')
                            if text and len(text.split()) > self.continuation_word_limit:
                                # Extract first n words from the text like other datasets
                                words = text.split()[:self.continuation_word_limit]
                                truncated_text = ' '.join(words)
                                query = f"Continue the following text: {truncated_text}"
                                if self._is_query_length_valid(query):
                                    total_seen += 1
                                    if len(reservoir) < num_samples:
                                        reservoir.append(query)
                                    else:
                                        j = random.randint(0, total_seen - 1)
                                        if j < num_samples:
                                            reservoir[j] = query
                                        
                                # Print progress occasionally
                                if total_seen % 100 == 0:
                                    print(f"Processed {total_seen} valid samples, collected {len(reservoir)}")
                                    
                                # Early exit if we have enough samples
                                if len(reservoir) >= num_samples and total_seen >= num_samples * 2:
                                    print(f"Collected enough samples: {len(reservoir)}")
                                    break
                    else:
                        print("Wikipedia dataset could not be loaded, using fallback samples")
                        
                except Exception as e:
                    print(f"Error loading Wikipedia dataset: {e}")
                    ds = None
                
                # Fallback to dummy samples if dataset loading failed or not enough samples
                if ds is None or len(reservoir) < num_samples:
                    print(f"Using fallback samples (current: {len(reservoir)}, needed: {num_samples})")
                    dummy_texts = [
                        "The solar system consists of the Sun and all celestial objects bound to it by gravity including planets moons asteroids comets and meteoroids The formation of the solar system began about 4.6 billion years ago with the gravitational collapse",
                        "Machine learning is a method of data analysis that automates analytical model building It is a branch of artificial intelligence based on the idea that systems can learn from data identify patterns and make decisions",
                        "Climate change refers to long term shifts in temperatures and weather patterns These shifts may be natural such as through variations in the solar cycle But since the 1800s human activities have been the main driver",
                        "Photosynthesis is the process by which plants use sunlight water and carbon dioxide to create oxygen and energy in the form of sugar This process is fundamental to life on Earth as it provides the oxygen",
                        "The Internet is a global system of interconnected computer networks that uses the Internet protocol suite to communicate between networks and devices It is a network of networks that consists of private public academic",
                        "Artificial intelligence AI is intelligence demonstrated by machines in contrast to the natural intelligence displayed by humans and animals Computer science defines AI research as the study of intelligent agents",
                        "Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles It is the foundation of all quantum physics",
                        "DNA or deoxyribonucleic acid is a molecule composed of two polynucleotide chains that coil around each other to form a double helix carrying genetic instructions for the development operation and maintenance",
                        "The human brain is the central organ of the human nervous system and with the spinal cord makes up the central nervous system The brain consists of the cerebrum brainstem and cerebellum",
                        "Global warming is the long term warming of the planet overall temperature This warming trend has been observed over the past century or more and is attributed primarily to human activities"
                    ]
                    
                    # Fill remaining slots with dummy samples
                    while len(reservoir) < num_samples and len(dummy_texts) > 0:
                        for text in dummy_texts:
                            if len(reservoir) >= num_samples:
                                break
                            words = text.split()[:self.continuation_word_limit]
                            truncated_text = ' '.join(words)
                            query = f"Continue the following text: {truncated_text}"
                            if self._is_query_length_valid(query):
                                total_seen += 1
                                if len(reservoir) < num_samples:
                                    reservoir.append(query)
                                else:
                                    j = random.randint(0, total_seen - 1)
                                    if j < num_samples:
                                        reservoir[j] = query
                                
            else:
                print(f"Warning: Unknown dataset {dataset_name}, skipping...")
                return []
            
            # Remove duplicates while preserving randomness
            unique_queries = []
            seen = set()
            for query in reservoir:
                if query not in seen:
                    unique_queries.append(query)
                    seen.add(query)
            
            if len(unique_queries) < num_samples:
                print(f"Warning: Dataset {dataset_name} yielded only {len(unique_queries)} unique samples after deduplication, less than requested {num_samples}")
                
            print(f"Dataset {dataset_name}: sampled {len(unique_queries)} samples from {total_seen} total valid samples")
            return unique_queries
            
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return []
    
    def generate_perturbed_queries(self, original_queries):
        """Generate augmented versions of queries using word substitution."""
        print(f"Using word substitution method for data augmentation...")
        self.logger.info(f"Starting word substitution data augmentation")
        
        # Use the word substitution helper
        substituted_data = self.word_substitution_helper.generate_substituted_queries(original_queries)
        
        print(f"Generated {len(substituted_data)} total substituted queries using word substitution")
        return substituted_data
    
    def load_embedding_model(self):
        """Load the sentence embedding model."""
        if self.embedding_model is None:
            print(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            print("Embedding model loaded successfully")
    
    def compute_query_embeddings(self, perturbed_queries_data, embedding_model=None):
        """
        Compute sentence embeddings for all perturbed queries.
        
        Args:
            perturbed_queries_data (list): List of dictionaries containing query data
            
        Returns:
            dict: Dictionary containing queries and their embeddings
        """
        # Check if embeddings already exist and don't need resampling
        if os.path.exists(self.embeddings_path) and not self.resample:
            print("Loading existing embeddings...")
            self.logger.info(f"Loading existing embeddings from {self.embeddings_path}")
            data = torch.load(self.embeddings_path)
            return data
        
        # Load embedding model
        if embedding_model is not None and self.embedding_model is None:
            self.embedding_model = embedding_model
        else:
            self.load_embedding_model()
        
        # Extract all queries: original + perturbed
        # First, get all original queries
        original_queries = list(set(item['original_query'] for item in perturbed_queries_data))
        perturbed_queries = [item['perturbed_query'] for item in perturbed_queries_data]
        
        # Combine: original queries first, then perturbed queries
        all_queries = original_queries + perturbed_queries
        
        print(f"Computing embeddings for {len(all_queries)} queries ({len(original_queries)} original + {len(perturbed_queries)} perturbed)...")
        print(f"Using batch size: {self.embedding_batch_size}")
        
        self.logger.info(f"Starting embedding computation for {len(all_queries)} queries")
        self.logger.info(f"  - Original queries: {len(original_queries)}")
        self.logger.info(f"  - perturbed queries: {len(perturbed_queries)}")
        self.logger.info(f"Embedding model: {self.embedding_model_name}")
        self.logger.info(f"Batch size: {self.embedding_batch_size}")
        
        # Compute embeddings in batches
        all_embeddings = []
        
        for i in range(0, len(all_queries), self.embedding_batch_size):
            batch_queries = all_queries[i:i + self.embedding_batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch_queries,
                batch_size=len(batch_queries),
                show_progress_bar=True if i == 0 else False,
                convert_to_tensor=True
            )
            all_embeddings.append(batch_embeddings)
            
            if (i // self.embedding_batch_size + 1) % 10 == 0:
                print(f"Processed {min(i + self.embedding_batch_size, len(all_queries))}/{len(all_queries)} queries")
        
        # Concatenate all embeddings
        embeddings_tensor = torch.cat(all_embeddings, dim=0)
        
        # Prepare data to save
        embedding_data = {
            'original_queries': original_queries,
            'perturbed_queries': perturbed_queries,
            'all_queries': all_queries,
            'embeddings': embeddings_tensor,
            'original_embeddings': embeddings_tensor[:len(original_queries)],  # First part: original queries
            'perturbed_embeddings': embeddings_tensor[len(original_queries):],  # Second part: perturbed queries
            'perturbed_queries_data': perturbed_queries_data,
            'metadata': {
                'embedding_model': self.embedding_model_name,
                'num_total_queries': len(all_queries),
                'num_original_queries': len(original_queries),
                'num_perturbed_queries': len(perturbed_queries),
                'embedding_dim': embeddings_tensor.shape[1],
                'num_rewrites': self.num_rewrites
            }
        }
        
        # Save embeddings
        os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
        torch.save(embedding_data, self.embeddings_path)
        
        # Log results
        self.logger.info(f"Embedding computation completed:")
        self.logger.info(f"  - Total embeddings: {len(all_queries)}")
        self.logger.info(f"  - Original embeddings: {len(original_queries)}")
        self.logger.info(f"  - perturbed embeddings: {len(perturbed_queries)}")
        self.logger.info(f"  - Embedding dimension: {embeddings_tensor.shape[1]}")
        self.logger.info(f"  - Embeddings tensor shape: {embeddings_tensor.shape}")
        self.logger.info(f"  - Saved to: {self.embeddings_path}")
        
        print(f"Successfully computed and saved {len(all_queries)} embeddings")
        print(f"Embedding dimension: {embeddings_tensor.shape[1]}")
        
        return embedding_data
    
    def unload_embedding_model(self):
        """Unload the embedding model to free memory."""
        if hasattr(self, 'embedding_model') and self.embedding_model is not None:
            del self.embedding_model
            self.embedding_model = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print("Embedding model unloaded")
    
    def load_cached_data(self):
        """
        Load cached queries and embeddings if they exist.
        
        Returns:
            tuple: (perturbed_queries_data, embedding_data) or (None, None) if not found
        """
        # Check if both files exist
        if (os.path.exists(self.perturbed_queries_path) and 
            os.path.exists(self.embeddings_path) and 
            not self.resample):
            
            print("Loading cached queries and embeddings...")
            self.logger.info("Loading cached data:")
            self.logger.info(f"  - Queries from: {self.perturbed_queries_path}")
            self.logger.info(f"  - Embeddings from: {self.embeddings_path}")
            
            # Load perturbed queries
            df = pd.read_csv(self.perturbed_queries_path)
            perturbed_queries_data = df.to_dict('records')
            
            # Load embeddings
            embedding_data = torch.load(self.embeddings_path)
            
            self.logger.info(f"Cached data loaded successfully:")
            self.logger.info(f"  - Queries: {len(perturbed_queries_data)}")
            self.logger.info(f"  - Embeddings: {len(embedding_data['all_queries'])}")
            self.logger.info(f"  - Embedding shape: {embedding_data['embeddings'].shape}")
            
            print(f"Loaded {len(perturbed_queries_data)} queries and {len(embedding_data['all_queries'])} embeddings from cache")
            return perturbed_queries_data, embedding_data
        
        return None, None
