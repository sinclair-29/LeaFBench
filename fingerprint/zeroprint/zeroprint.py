from fingerprint.fingerprint_interface import LLMFingerprintInterface
from fingerprint.zeroprint.zeroprint_prepare_helper import ZeroPrintPrepareHelper
from fingerprint.zeroprint.zeroprint_fingerprint_helper import ZeroPrintFingerprintHelper
import logging
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap


class ZeroPrintFingerprint(LLMFingerprintInterface):
    """
    ZeroPrint Fingerprinting implementation based on zero-order gradient estimation.
    """

    def __init__(self, config=None, accelerator=None):
        super().__init__(config, accelerator)
        self.query_samples = None
        self.perturbed_queries = None
        self.embedding_data = None
        self.fingerprint_helper = ZeroPrintFingerprintHelper(config, accelerator)
        self.embedding_model = None

    @property
    def logger(self):
        """Get logger instance for this class."""
        return logging.getLogger(__name__)
    
    def _load_embedding_model(self):
        """Load the sentence embedding model."""
        if self.embedding_model is None:
            # load config
            embedding_model_name = self.config.get('embedding_model_name', 'Qwen/Qwen3-Embedding-4B')

            self.logger.info(f"Loading embedding model: {embedding_model_name}")

            # load embedding model
            self.embedding_model = SentenceTransformer(embedding_model_name)

        
            self.logger.info("Embedding model loaded successfully")

    def prepare(self, train_models=None):
        """
        Prepare the fingerprinting method: sample query inputs, generate perturbed versions, and compute embeddings.

        ``train_models`` is accepted for compatibility with the LeafBench
        fingerprint interface. ZeroPrint is training-free and does not use it.
        """
        print("Preparing ZeroPrint fingerprinting method...")
        
        # Log hyperparameters
        self.logger.info("ZeroPrint hyperparameters:")
        self.logger.info(f"  - n_samples: {self.config.get('n_samples', 200)}")
        self.logger.info(f"  - gradient_estimation_method: {self.config.get('gradient_estimation_method', 'jacobian')}")
        self.logger.info(f"  - num_rewrites: {self.config.get('num_rewrites', 5)}")
        self.logger.info(f"  - query_datasets: {self.config.get('query_datasets', [])}")
        
        word_sub_config = self.config.get('word_substitution_config', {})
        self.logger.info(f"  - word_vector_model: {word_sub_config.get('word_vector_model', 'glove')}")
        self.logger.info(f"  - k_words_to_replace: {word_sub_config.get('k_words_to_replace', 3)}")
        self.logger.info(f"  - top_m_neighbors: {word_sub_config.get('top_m_neighbors', 10)}")
        self.logger.info(f"  - n_augmented_samples: {word_sub_config.get('n_augmented_samples', 5)}")
        
        self.logger.info(f"  - embedding_model: {self.config.get('embedding_model_name', 'N/A')}")
        self.logger.info(f"  - embedding_batch_size: {self.config.get('embedding_batch_size', 32)}")
        self.logger.info(f"  - resample: {self.config.get('resample', False)}")
        
        # Initialize the prepare helper
        prepare_helper = ZeroPrintPrepareHelper(self.config)
        
        # Check if cached data exists
        cached_queries, cached_embeddings = prepare_helper.load_cached_data()
        
        if cached_queries is not None and cached_embeddings is not None:
            print("Using cached data...")
            self.logger.info("Loading cached queries and embeddings")
            
            self.perturbed_queries = cached_queries
            self.embedding_data = cached_embeddings
            
            # Extract original queries from cached data
            original_queries = list(set(item['original_query'] for item in self.perturbed_queries))
            self.query_samples = original_queries
            
            # Log results
            self.logger.info(f"Loaded {len(self.query_samples)} original queries from cache")
            self.logger.info(f"Loaded {len(self.perturbed_queries)} perturbed queries from cache")
            self.logger.info(f"Loaded embeddings with shape {self.embedding_data['embeddings'].shape}")
            
            print(f"Loaded {len(self.query_samples)} original queries")
            print(f"Loaded {len(self.perturbed_queries)} perturbed queries")
            print(f"Loaded {len(self.embedding_data['embeddings'])} embeddings")
            
        else:
            print("Generating new data...")
            self.logger.info("Generating new queries, rewrites, and embeddings")
            
            # Step 1: Get query samples from datasets
            print("Step 1: Sampling query inputs from datasets...")
            self.query_samples = prepare_helper.get_query_samples()
            
            # Log step 1 results
            self.logger.info(f"Step 1 completed: sampled {len(self.query_samples)} queries from datasets")
            print(f"Successfully prepared {len(self.query_samples)} original query samples")
            
            # Step 2: Generate perturbed versions of queries
            print(f"Step 2: Generating augmented versions of queries using word substitution method...")
            self.perturbed_queries = prepare_helper.generate_perturbed_queries(self.query_samples)
            
            word_sub_config = self.config.get('word_substitution_config', {})
            num_augmentations = word_sub_config.get('n_augmented_samples', 5)
                
            expected_total = len(self.query_samples) * num_augmentations
            
            # Log step 2 results
            self.logger.info(f"Step 2 completed: generated {len(self.perturbed_queries)} augmented queries")
            self.logger.info(f"Expected augmentations: {expected_total}, Actual: {len(self.perturbed_queries)}")
            
        # Step 3: Compute sentence embeddings
        print("Step 3: Computing sentence embeddings...")
        self._load_embedding_model()
        self.embedding_data = prepare_helper.compute_query_embeddings(self.perturbed_queries, self.embedding_model)
        
        # Log step 3 results
        embedding_shape = self.embedding_data['embeddings'].shape
        self.logger.info(f"Step 3 completed: computed embeddings with shape {embedding_shape}")
        self.logger.info(f"Embedding model used: {self.embedding_data['metadata']['embedding_model']}")
        
        print(f"Successfully computed {len(self.embedding_data['embeddings'])} embeddings")
        print(f"Embedding dimension: {self.embedding_data['embeddings'].shape[1]}")
        
        # Log final statistics
        stats = self.get_query_statistics()
        self.logger.info("ZeroPrint preparation completed successfully:")
        self.logger.info(f"  - Total original queries: {stats['num_original_queries']}")
        self.logger.info(f"  - Total perturbed queries: {stats['num_perturbed_queries']}")
        self.logger.info(f"  - Total queries for fingerprinting: {stats['total_queries']}")
        self.logger.info(f"  - Embedding dimension: {stats.get('embedding_dimension', 'N/A')}")
        self.logger.info(f"  - Has embeddings: {stats['has_embeddings']}")
        
        print("ZeroPrint fingerprinting method preparation completed.")
    
    def get_all_queries(self):
        """
        Get all queries (original + perturbed versions) for fingerprinting.
        
        Returns:
            list: List of all query strings.
        """
        if self.embedding_data is not None:
            return self.embedding_data['all_queries']
        elif self.perturbed_queries is not None:
            return [item['perturbed_query'] for item in self.perturbed_queries]
        else:
            return self.query_samples if self.query_samples else []
    
    def get_all_embeddings(self):
        """
        Get all query embeddings.
        
        Returns:
            torch.Tensor: Tensor of all query embeddings.
        """
        if self.embedding_data is not None:
            return self.embedding_data['embeddings']
        else:
            return None
    
    def get_query_statistics(self):
        """
        Get statistics about the prepared queries and embeddings.
        
        Returns:
            dict: Statistics about queries and embeddings.
        """
        stats = {
            'num_original_queries': len(self.query_samples) if self.query_samples else 0,
            'num_perturbed_queries': len(self.perturbed_queries) if self.perturbed_queries else 0,
            'num_rewrites_per_query': self.config.get('num_rewrites', 0),
            'total_queries': len(self.get_all_queries()),
            'embedding_dimension': None,
            'has_embeddings': self.embedding_data is not None
        }
        
        if self.embedding_data is not None:
            metadata = self.embedding_data['metadata']
            stats['embedding_dimension'] = metadata['embedding_dim']
            stats['embedding_model'] = metadata['embedding_model']
            stats['num_total_embeddings'] = metadata['num_total_queries']
            stats['num_original_embeddings'] = metadata['num_original_queries']
            stats['num_perturbed_embeddings'] = metadata['num_perturbed_queries']
        
        return stats
    
    def get_fingerprint(self, model):
        """
        Generate a fingerprint for the given model using zero-order gradient estimation.

        Args:
            model: The model to extract the fingerprint.

        Returns:
            torch.Tensor: The fingerprint tensor (estimated gradients).
        """
        if self.embedding_data is None:
            raise ValueError("Must call prepare() first to compute query embeddings")
        
        self.logger.info(f"Generating fingerprint for model using zero-order gradient estimation")
        
        # Get data
        original_queries = self.embedding_data['original_queries']
        perturbed_queries_data = self.embedding_data['perturbed_queries_data']
        original_embeddings = self.embedding_data['original_embeddings']  # I_0 embeddings
        perturbed_embeddings = self.embedding_data['perturbed_embeddings']  # I_1...I_m embeddings
        
        n_original = len(original_queries)
        m_rewrites = self.config.get('num_rewrites', 5)
        
        self.logger.info(f"Processing {n_original} original queries with {m_rewrites} rewrites each")
        
        # Step 1: Generate outputs for all queries (original + perturbed)
        print("Step 1: Generating model outputs...")
        all_queries = self.embedding_data['all_queries']
        truncate_length = self.config.get('output_truncate_length', 0)
        model_outputs = self._generate_model_outputs(model, all_queries, truncate_length)
        
        # Step 2: Compute sentence embeddings for outputs
        print("Step 2: Computing output embeddings...")
        output_embeddings = self._compute_output_embeddings(model_outputs)
        
        # Split output embeddings: original outputs (O_0) and perturbed outputs (O_1...O_m)
        original_output_embeddings = output_embeddings[:n_original]  # O_0
        perturbed_output_embeddings = output_embeddings[n_original:]  # O_1...O_m
        
        # Step 3: Estimate gradients for each sample
        gradient_method = self.config.get('gradient_estimation_method', 'jacobian')
        print(f"Step 3: Estimating gradients using {gradient_method} method...")
        estimated_gradients = self._estimate_gradients(
            original_queries, 
            perturbed_queries_data,
            original_embeddings,  # I_0
            perturbed_embeddings,  # I_1...I_m
            original_output_embeddings,  # O_0
            perturbed_output_embeddings  # O_1...O_m
        ).cpu()
        
        return estimated_gradients
    
    def _generate_model_outputs(self, model, queries, truncate_length=None):
        """
        Generate model outputs for all queries.
        
        Args:
            model: The model to generate outputs with
            queries: List of query strings
            truncate_length (int, optional): Maximum number of tokens/words to keep in output. 
                                           If 0 or None, no truncation is applied.
            
        Returns:
            list: List of generated output strings (potentially truncated), with multiple outputs per query if num_repeats > 1
        """
        batch_size = self.config.get('generation_batch_size', 10)
        num_repeats = self.config.get('num_repeats', 1)
        
        # Get truncate_length from config if not provided
        if truncate_length is None:
            truncate_length = self.config.get('output_truncate_length', 0)
        
        # Create expanded query list with repeats
        expanded_queries = []
        for query in queries:
            for _ in range(num_repeats):
                expanded_queries.append(query)
        
        self.logger.info(f"Generating outputs for {len(expanded_queries)} total queries ({len(queries)} queries × {num_repeats} repeats) with batch size {batch_size}")
        if truncate_length and truncate_length > 0:
            self.logger.info(f"Output truncation enabled: keeping first {truncate_length} tokens/words")
        
        outputs = []
        
        # Process in batches
        for i in range(0, len(expanded_queries), batch_size):
            batch_queries = expanded_queries[i:i + batch_size]
            
            with torch.no_grad():
                batch_outputs = model.generate(batch_queries)
            
            # Apply truncation if specified
            if truncate_length and truncate_length > 0:
                truncated_outputs = []
                for output in batch_outputs:
                    # Split by whitespace to get tokens/words
                    tokens = output.split()
                    if len(tokens) > truncate_length:
                        # Keep only the first truncate_length tokens and rejoin
                        truncated_output = ' '.join(tokens[:truncate_length])
                        truncated_outputs.append(truncated_output)
                    else:
                        truncated_outputs.append(output)
                batch_outputs = truncated_outputs
            
            outputs.extend(batch_outputs)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Generated outputs for {min(i + batch_size, len(expanded_queries))}/{len(expanded_queries)} total queries")
        
        return outputs
    
    def _compute_output_embeddings(self, outputs):
        """
        Compute sentence embeddings for model outputs.
        
        Args:
            outputs: List of output strings (length = num_queries * num_repeats)
            
        Returns:
            torch.Tensor: Average embeddings for outputs (length = num_queries)
        """
        
        # Load the same embedding model used for input queries
        self._load_embedding_model()
        
        batch_size = self.config.get('embedding_batch_size', 32)
        num_repeats = self.config.get('num_repeats', 1)
        
        # Calculate number of unique queries
        num_queries = len(outputs) // num_repeats
        
        self.logger.info(f"Computing embeddings for {len(outputs)} outputs ({num_queries} queries × {num_repeats} repeats) using batch size {batch_size}")
        
        # Compute embeddings in batches
        all_embeddings = []
        
        for i in range(0, len(outputs), batch_size):
            batch_outputs = outputs[i:i + batch_size]
            with torch.no_grad():
                batch_embeddings = self.embedding_model.encode(
                    batch_outputs,
                    show_progress_bar=False,
                    convert_to_tensor=True
                )
            all_embeddings.append(batch_embeddings)
            
            # Log progress
            if (i // batch_size + 1) % 10 == 0:
                print(f"Computed embeddings for {min(i + batch_size, len(outputs))}/{len(outputs)} outputs")
        
        # Concatenate all embeddings
        all_output_embeddings = torch.cat(all_embeddings, dim=0)  # Shape: (num_queries * num_repeats, embedding_dim)
        
        # Reshape and average embeddings if num_repeats > 1
        if num_repeats > 1:
            embedding_dim = all_output_embeddings.shape[1]
            # Reshape to (num_queries, num_repeats, embedding_dim)
            reshaped_embeddings = all_output_embeddings.view(num_queries, num_repeats, embedding_dim)
            # Average across repeats dimension
            output_embeddings = torch.mean(reshaped_embeddings, dim=1)  # Shape: (num_queries, embedding_dim)
            self.logger.info(f"Averaged {num_repeats} embeddings per query, final shape: {output_embeddings.shape}")
        else:
            output_embeddings = all_output_embeddings
        
        # Clean up
        # del embedding_model
        # torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return output_embeddings
    
    def _estimate_gradients(self, original_queries, perturbed_queries_data,
                           original_input_embeddings, perturbed_input_embeddings,
                           original_output_embeddings, perturbed_output_embeddings):
        """
        Estimate gradients using the specified method.
        
        Args:
            original_queries: List of original queries
            perturbed_queries_data: List of rewrite dicts (containing original_query)
            original_input_embeddings: Original input embeddings I_0, tensor [n_original, D_in]
            perturbed_input_embeddings: perturbed input embeddings I_k, tensor [N_rewrites, D_in]
            original_output_embeddings: Original output embeddings O_0, tensor [n_original, D_out]
            perturbed_output_embeddings: perturbed output embeddings O_k, tensor [N_rewrites, D_out]

        Returns:
            torch.Tensor: Estimated gradients
        """
        
        gradient_estimation_method = self.config.get('gradient_estimation_method', 'jacobian')
        
        if gradient_estimation_method == 'jacobian':
            return self._estimate_gradients_jacobian(
                original_queries, perturbed_queries_data,
                original_input_embeddings, perturbed_input_embeddings,
                original_output_embeddings, perturbed_output_embeddings
            )
        elif gradient_estimation_method == 'tensor':
            return self._estimate_gradients_tensor(
                original_queries, perturbed_queries_data,
                original_input_embeddings, perturbed_input_embeddings,
                original_output_embeddings, perturbed_output_embeddings
            )
        else:
            raise ValueError(f"Unsupported gradient estimation method: {gradient_estimation_method}")
    
    def _estimate_gradients_jacobian(self, original_queries, perturbed_queries_data,
                                   original_input_embeddings, perturbed_input_embeddings,
                                   original_output_embeddings, perturbed_output_embeddings):
        """
        Fit a local Jacobian J via Ridge regression: ΔE_out ≈ J · ΔE_in.

        For each original query i, collect its m rewrite pairs to form the dataset
        {(ΔE_in_k, ΔE_out_k)}_{k=1..m} and solve a ridge regression estimate of J_i.
        To facilitate later aggregation, return a flattened J_i vector per sample
        with shape (n_original, D_out * D_in).

        Args:
            original_queries: List of original queries
            perturbed_queries_data: List of rewrite dicts (containing original_query)
            original_input_embeddings: Original input embeddings I_0, tensor [n_original, D_in]
            perturbed_input_embeddings: perturbed input embeddings I_k, tensor [N_rewrites, D_in]
            original_output_embeddings: Original output embeddings O_0, tensor [n_original, D_out]
            perturbed_output_embeddings: perturbed output embeddings O_k, tensor [N_rewrites, D_out]

        Returns:
            torch.Tensor: Flattened Jacobian estimates with shape [n_original, D_out * D_in]
        """

        n_original = len(original_queries)
        m_rewrites = self.config.get('num_rewrites', 5)
        word_sub_config = self.config.get('word_substitution_config', {})
        m_rewrites = word_sub_config.get('n_augmented_samples', 5)
            
        D_in = original_input_embeddings.shape[1]
        D_out = original_output_embeddings.shape[1]
        
        # Get ridge regression parameter
        ridge_alpha = self.config.get('ridge_alpha', 1.0)

        self.logger.info(
            f"Estimating local Jacobians via ridge regression (α={ridge_alpha}) for {n_original} samples with {m_rewrites} rewrites each"
        )

        # Result tensor: one flattened J vector per sample
        device = self.accelerator.device if hasattr(self.accelerator, 'device') else 'cpu'
        estimated_jacobians = torch.zeros(n_original, D_out * D_in, dtype=torch.float32)

        # Map each original query to indices of its rewrites
        query_to_rewrites = {}
        rewrite_idx = 0
        for item in perturbed_queries_data:
            original_query = item['original_query']
            if original_query not in query_to_rewrites:
                query_to_rewrites[original_query] = []
            query_to_rewrites[original_query].append(rewrite_idx)
            rewrite_idx += 1

        # Fit J_i per sample
        for i, original_query in enumerate(original_queries):
            if original_query not in query_to_rewrites:
                self.logger.warning(f"No rewrites found for original query {i}")
                continue

            rewrite_indices = query_to_rewrites[original_query]
            if len(rewrite_indices) != m_rewrites:
                self.logger.warning(
                    f"Expected {m_rewrites} rewrites, got {len(rewrite_indices)} for query {i}"
                )

            # Baseline embeddings
            I_0 = original_input_embeddings[i].detach().cpu().numpy()  # [D_in]
            O_0 = original_output_embeddings[i].detach().cpu().numpy()  # [D_out]

            # Build X (ΔE_in) and Y (ΔE_out)
            X_list = []  # (m, D_in)
            Y_list = []  # (m, D_out)
            for idx in rewrite_indices:
                I_k = perturbed_input_embeddings[idx].detach().cpu().numpy()
                O_k = perturbed_output_embeddings[idx].detach().cpu().numpy()
                d_in = I_k - I_0
                d_out = O_k - O_0

                # Skip too-small perturbations to avoid ill-conditioning
                if np.linalg.norm(d_in, ord=2) <= 1e-8:
                    continue
                X_list.append(d_in)
                Y_list.append(d_out)

            if len(X_list) == 0:
                self.logger.warning(f"No valid rewrite deltas for sample {i}")
                continue

            X = np.stack(X_list, axis=0)  # (m_eff, D_in)
            Y = np.stack(Y_list, axis=0)  # (m_eff, D_out)

            try:
                # Ridge regression solution: (X^T X + α I)^(-1) X^T Y
                # Coef shape (D_in, D_out)
                XTX = X.T @ X  # (D_in, D_in)
                XTY = X.T @ Y  # (D_in, D_out)
                
                # Add ridge regularization: α * I
                ridge_term = ridge_alpha * np.eye(XTX.shape[0])  # (D_in, D_in)
                regularized_XTX = XTX + ridge_term
                
                # Solve the ridge regression system
                Coef = np.linalg.solve(regularized_XTX, XTY)  # (D_in, D_out)
                
                # J = Coef^T, shape (D_out, D_in)
                J = Coef.T
                estimated_jacobians[i] = torch.from_numpy(J.reshape(-1))
            except Exception as e:
                self.logger.warning(f"Ridge regression failed for sample {i}: {e}")
                # Keep zero vector on failure
                continue

        self.logger.info(
            f"Jacobian estimation completed. Final shape: {tuple(estimated_jacobians.shape)} (n, D_out*D_in)"
        )

        return estimated_jacobians
    
    def _estimate_gradients_tensor(self, original_queries, perturbed_queries_data,
                                 original_input_embeddings, perturbed_input_embeddings,
                                 original_output_embeddings, perturbed_output_embeddings):
        """
        Estimate gradients using tensor method: (ΔE_out) / ||ΔE_in||_2.
        
        For each original query i and its m rewrites, compute:
        gradient_i_k = (O_k - O_0) / ||I_k - I_0||_2
        Then average over all rewrites for each original query.
        
        Args:
            original_queries: List of original queries
            perturbed_queries_data: List of rewrite dicts (containing original_query)
            original_input_embeddings: Original input embeddings I_0, tensor [n_original, D_in]
            perturbed_input_embeddings: perturbed input embeddings I_k, tensor [N_rewrites, D_in]
            original_output_embeddings: Original output embeddings O_0, tensor [n_original, D_out]
            perturbed_output_embeddings: perturbed output embeddings O_k, tensor [N_rewrites, D_out]

        Returns:
            torch.Tensor: Gradient estimates with shape [n_original, D_out]
        """

        n_original = len(original_queries)
        m_rewrites = self.config.get('num_rewrites', 5)
        word_sub_config = self.config.get('word_substitution_config', {})
        m_rewrites = word_sub_config.get('n_augmented_samples', 5)
            
        D_out = original_output_embeddings.shape[1]

        self.logger.info(
            f"Estimating gradients via tensor method for {n_original} samples with {m_rewrites} rewrites each"
        )

        # Result tensor: one gradient vector per sample
        device = self.accelerator.device if hasattr(self.accelerator, 'device') else 'cpu'
        estimated_gradients = torch.zeros(n_original, D_out, dtype=torch.float32, device=device)

        # Map each original query to indices of its rewrites
        query_to_rewrites = {}
        rewrite_idx = 0
        for item in perturbed_queries_data:
            original_query = item['original_query']
            if original_query not in query_to_rewrites:
                query_to_rewrites[original_query] = []
            query_to_rewrites[original_query].append(rewrite_idx)
            rewrite_idx += 1

        # Compute gradients per sample
        for i, original_query in enumerate(original_queries):
            if original_query not in query_to_rewrites:
                self.logger.warning(f"No rewrites found for original query {i}")
                continue

            rewrite_indices = query_to_rewrites[original_query]
            if len(rewrite_indices) != m_rewrites:
                self.logger.warning(
                    f"Expected {m_rewrites} rewrites, got {len(rewrite_indices)} for query {i}"
                )

            # Baseline embeddings
            I_0 = original_input_embeddings[i]  # [D_in]
            O_0 = original_output_embeddings[i]  # [D_out]

            # Collect gradients for all rewrites of this original query
            gradient_list = []
            for idx in rewrite_indices:
                I_k = perturbed_input_embeddings[idx]  # [D_in]
                O_k = perturbed_output_embeddings[idx]  # [D_out]
                
                # Compute differences
                d_in = I_k - I_0  # [D_in]
                d_out = O_k - O_0  # [D_out]
                
                # Compute L2 norm of input difference
                d_in_norm = torch.norm(d_in, p=2)
                
                # Skip too-small perturbations to avoid division by zero
                if d_in_norm <= 1e-8:
                    d_in_norm = 1e-8
                
                # Compute gradient: ΔE_out / ||ΔE_in||_2
                gradient_k = d_out / d_in_norm  # [D_out]
                gradient_list.append(gradient_k)

            if len(gradient_list) == 0:
                self.logger.warning(f"No valid rewrite deltas for sample {i}")
                continue

            # Average gradients over all rewrites for this sample
            sample_gradient = torch.stack(gradient_list, dim=0).mean(dim=0)  # [D_out]
            estimated_gradients[i] = sample_gradient.to(device)

        self.logger.info(
            f"Tensor gradient estimation completed. Final shape: {tuple(estimated_gradients.shape)} (n, D_out)"
        )

        return estimated_gradients
    
    def _aggregate_gradients(self, gradients, config=None):
        """
        Aggregate multiple sample gradients into a single fingerprint embedding.
        
        Args:
            gradients: Tensor of shape (n_samples, embedding_dim) containing gradients for each sample
            
        Returns:
            torch.Tensor: Aggregated fingerprint
        """
        aggregation_method = self.config.get('fingerprint_aggregation', 'mean')
        
        self.logger.info(f"Aggregating {gradients.shape[0]} gradients using method: {aggregation_method}")
        
        if aggregation_method == 'mean':
            # Compute mean across samples
            aggregated = torch.mean(gradients, dim=0)
            
        elif aggregation_method == 'max':
            # Compute max across samples for each dimension
            aggregated, _ = torch.max(gradients, dim=0)
            
        elif aggregation_method == 'sum':
            # Compute sum across samples for each dimension
            aggregated = torch.sum(gradients, dim=0)
            
        elif aggregation_method == 'var':
            # Compute variance across samples for each dimension
            aggregated = torch.var(gradients, dim=0, unbiased=True)
            
        elif aggregation_method == 'none':
            # No aggregation, return all gradients
            aggregated = gradients
        elif aggregation_method == 'pca':
            # Apply PCA to reduce to 1D
            pca = PCA(n_components=config.get('n_components', 128), n_jobs=64)
            reduced = pca.fit_transform(gradients.cpu().numpy())  # Shape (n_samples, n_components)
            aggregated = torch.from_numpy(reduced).to(gradients.device).squeeze()
        elif aggregation_method == 'isomap':
            # Apply Isomap to reduce to 1D
            isomap = Isomap(n_components=config.get('n_components', 128), n_jobs=64)
            reduced = isomap.fit_transform(gradients.cpu().numpy())  # Shape (n_samples, n_components)
            aggregated = torch.from_numpy(reduced).to(gradients.device).squeeze()
        else:
            self.logger.warning(f"Unknown aggregation method: {aggregation_method}, using 'mean' as fallback")
            aggregated = torch.mean(gradients, dim=0)
        
        self.logger.info(f"Aggregation completed. Input shape: {gradients.shape}, Output shape: {aggregated.shape}")
        
        return aggregated
    
    def compare_fingerprints(self, base_model, testing_model):
        """
        Compare two models using their fingerprints.

        Args:
            base_model (ModelInterface): The base model to compare against.
            testing_model (ModelInterface): The model to compare.

        Returns:
            float: Similarity score between the two fingerprints in [0,1] range.
        """
        self.logger.info("Comparing two models using ZeroPrint fingerprints")
        self.logger.info(f"Similarity metric: {self.config.get('similarity_metric', 'cosine')}")
        
        # Generate fingerprints for both models
        # print("Generating fingerprint for base model...")
        base_fingerprint = base_model.get_fingerprint()
        
        # print("Generating fingerprint for testing model...")
        testing_fingerprint = testing_model.get_fingerprint()
        
        # Log fingerprint shapes
        # self.logger.info(f"Base model fingerprint shape: {base_fingerprint.shape}")
        # self.logger.info(f"Testing model fingerprint shape: {testing_fingerprint.shape}")
        
        # Handle different aggregation methods
        # aggregation_method = self.config.get('fingerprint_aggregation', 'mean')

        base_aggregated = self._aggregate_gradients(base_fingerprint, self.config) if len(base_fingerprint.shape) > 1 else base_fingerprint
        testing_aggregated = self._aggregate_gradients(testing_fingerprint, self.config) if len(testing_fingerprint.shape) > 1 else testing_fingerprint

        final_similarity = self.fingerprint_helper.compute_similarity(
            base_aggregated, 
            testing_aggregated
        )
        
        self.logger.info(f"Computed similarity between aggregated fingerprints: {final_similarity:.6f}")
        
        print(f"Fingerprint similarity: {final_similarity:.6f}")
        
        return final_similarity
