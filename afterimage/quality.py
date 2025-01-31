from typing import List, Dict, Any, Optional
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import langdetect

from .types import Role
from .storage import DatasetStorage


class QualityChecker:
    """Analyzes and validates conversation dataset quality."""

    def __init__(
        self,
        storage: DatasetStorage,
        min_length: int = 50,
        max_length: int = 2000,
        language: Optional[str] = None,
        embedding_model: str | SentenceTransformer = "altaidevorg/bge-m3-distill-8l",
    ):
        """Initialize quality checker.

        Args:
            storage: Storage backend containing conversations
            min_length: Minimum acceptable response length
            max_length: Maximum acceptable response length
            language: Expected language code (e.g., 'tr', 'en')
            embedding_model: Model name or instance for semantic analysis
        """
        self.storage = storage
        self.min_length = min_length
        self.max_length = max_length
        self.language = language

        # Initialize embedding model
        self.model = (
            embedding_model
            if isinstance(embedding_model, SentenceTransformer)
            else SentenceTransformer(embedding_model)
        )

    def check_length_distribution(self) -> Dict[str, Any]:
        """Analyze response length distribution."""
        lengths = defaultdict(list)

        for conv in self.storage.load_conversations():
            for turn in conv.conversations:
                lengths[turn.role].append(len(turn.content))

        return {
            role: {
                "mean": np.mean(lens),
                "std": np.std(lens),
                "min": min(lens),
                "max": max(lens),
                "outliers": [
                    l
                    for l in lens  # noqa
                    if l < self.min_length or l > self.max_length
                ],
            }
            for role, lens in lengths.items()
        }

    def check_turn_balance(self) -> Dict[str, float]:
        """Check if conversations have balanced turns."""
        role_counts = defaultdict(int)
        total_turns = 0

        for conv in self.storage.load_conversations():
            for turn in conv.conversations:
                role_counts[turn.role] += 1
                total_turns += 1

        return {role: count / total_turns for role, count in role_counts.items()}

    def find_duplicates(
        self,
        similarity_threshold: float = 0.9,
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """Find near-duplicate responses using semantic similarity.

        Args:
            similarity_threshold: Minimum similarity score to consider as duplicate
            batch_size: Batch size for embedding computation

        Returns:
            List of duplicate pairs with similarity scores
        """
        duplicates = []

        # Group by role for more efficient comparison
        responses = defaultdict(list)
        for conv in self.storage.load_conversations():
            for turn in conv.conversations:
                responses[turn.role].append(turn.content)

        for role, texts in responses.items():
            # Compute embeddings in batches
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                embeddings.extend(self.model.encode(batch, normalize_embeddings=True))
            embeddings = np.array(embeddings)

            # Compute similarity matrix
            similarities = np.dot(embeddings, embeddings.T)

            # Find similar pairs (upper triangle only)
            similar_pairs = np.where(np.triu(similarities > similarity_threshold, k=1))

            for idx1, idx2 in zip(*similar_pairs):
                duplicates.append(
                    {
                        "role": role,
                        "text1": texts[idx1],
                        "text2": texts[idx2],
                        "similarity": float(similarities[idx1, idx2]),
                    }
                )

        return duplicates

    def check_language_consistency(self) -> Dict[str, Any]:
        """Check if responses are in expected language."""
        if not self.language:
            return {}

        inconsistent = []
        for conv in self.storage.load_conversations():
            for turn in conv.conversations:
                detected = langdetect.detect(turn.content)
                if detected != self.language:
                    inconsistent.append(
                        {
                            "expected": self.language,
                            "detected": detected,
                            "text": turn.content,
                        }
                    )

        return {"total_inconsistent": len(inconsistent), "examples": inconsistent[:5]}

    def check_response_coherence(self) -> Dict[str, Any]:
        """Check semantic coherence between questions and answers."""
        coherence_scores = []
        low_coherence = []

        for conv in self.storage.load_conversations():
            for i in range(0, len(conv.conversations) - 1, 2):
                question = conv.conversations[i].content
                answer = conv.conversations[i + 1].content

                # Compute semantic similarity between Q&A
                embeddings = self.model.encode(
                    [question, answer], normalize_embeddings=True
                )
                coherence = float(np.dot(embeddings[0], embeddings[1]))
                coherence_scores.append(coherence)

                if coherence < 0.5:  # Configurable threshold
                    low_coherence.append(
                        {"question": question, "answer": answer, "coherence": coherence}
                    )

        return {
            "mean_coherence": np.mean(coherence_scores),
            "std_coherence": np.std(coherence_scores),
            "low_coherence_count": len(low_coherence),
            "examples": low_coherence[:5],
        }

    def check_context_relevance(self) -> Dict[str, Any]:
        """Analyze how well responses utilize the provided context."""
        relevance_scores = []
        low_relevance = []

        for conv in self.storage.load_conversations():
            if not conv.context:
                continue

            context_embedding = self.model.encode(
                conv.context, normalize_embeddings=True
            )

            for turn in conv.conversations:
                if turn.role == Role.ASSISTANT:
                    response_embedding = self.model.encode(
                        turn.content, normalize_embeddings=True
                    )
                    relevance = float(np.dot(context_embedding, response_embedding))
                    relevance_scores.append(relevance)

                    if relevance < 0.3:  # Configurable threshold
                        low_relevance.append(
                            {
                                "context": conv.context,
                                "response": turn.content,
                                "relevance": relevance,
                            }
                        )

        return {
            "mean_relevance": np.mean(relevance_scores),
            "std_relevance": np.std(relevance_scores),
            "low_relevance_count": len(low_relevance),
            "examples": low_relevance[:5],
        }

    def visualize_metrics(
        self, save_dir: Optional[str | Path] = None
    ) -> Dict[str, plt.Figure]:
        """Generate visualizations for quality metrics.

        Args:
            save_dir: Optional directory to save plots

        Returns:
            Dict of matplotlib figures
        """
        figures = {}

        # Set style
        plt.style.use("seaborn")

        # 1. Length Distribution
        length_stats = self.check_length_distribution()
        fig, ax = plt.subplots(figsize=(10, 6))

        for role in length_stats:
            lengths = length_stats[role]["outliers"]
            sns.histplot(lengths, label=role, alpha=0.6, ax=ax)

        ax.set_title("Response Length Distribution")
        ax.set_xlabel("Length (chars)")
        ax.legend()
        figures["length_distribution"] = fig

        # 2. Turn Balance
        turn_balance = self.check_turn_balance()
        fig, ax = plt.subplots(figsize=(8, 8))

        plt.pie(turn_balance.values(), labels=turn_balance.keys(), autopct="%1.1f%%")
        ax.set_title("Turn Distribution")
        figures["turn_balance"] = fig

        # 3. Coherence Distribution
        coherence = self.check_response_coherence()
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.histplot(coherence["coherence_scores"], ax=ax)
        ax.set_title("Question-Answer Coherence Distribution")
        ax.set_xlabel("Coherence Score")
        figures["coherence"] = fig

        # Save if directory provided
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)

            for name, fig in figures.items():
                fig.savefig(save_dir / f"{name}.png")

        return figures

    def generate_report(
        self,
        include_plots: bool = True,
        save_dir: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive quality report with optional visualizations."""
        report = {
            "length_stats": self.check_length_distribution(),
            "turn_balance": self.check_turn_balance(),
            "language_check": self.check_language_consistency(),
            "duplicates": self.find_duplicates(),
            "coherence": self.check_response_coherence(),
            "context_relevance": self.check_context_relevance(),
        }

        if include_plots:
            report["visualizations"] = self.visualize_metrics(save_dir)

        return report
