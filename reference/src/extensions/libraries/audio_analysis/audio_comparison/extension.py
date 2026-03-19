"""
Audio Comparison Extension.

Compare audio files for similarity and detect duplicates using audio embeddings.
Uses CLAP (Contrastive Language-Audio Pretraining) for robust audio embeddings.

Usage:
    from src.extensions.analysis.audio_comparison.extension import register_extension
    extension = register_extension()
    result = extension.compare_audio("file1.wav", "file2.wav")
"""

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ComparisonResult:
    """Result from audio comparison."""

    def __init__(
        self,
        file1: str,
        file2: str,
        similarity: float,
        is_duplicate: bool,
        is_similar: bool,
        metadata: dict | None = None,
    ):
        self.file1 = file1
        self.file2 = file2
        self.similarity = similarity
        self.is_duplicate = is_duplicate
        self.is_similar = is_similar
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "file1": self.file1,
            "file2": self.file2,
            "similarity": round(self.similarity, 4),
            "is_duplicate": self.is_duplicate,
            "is_similar": self.is_similar,
            "metadata": self.metadata,
        }


class BatchComparisonResult:
    """Result from batch audio comparison."""

    def __init__(
        self,
        comparisons: list[ComparisonResult],
        duplicates: list[tuple[str, str]],
        similar_pairs: list[tuple[str, str]],
    ):
        self.comparisons = comparisons
        self.duplicates = duplicates
        self.similar_pairs = similar_pairs

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "comparisons": [c.to_dict() for c in self.comparisons],
            "duplicates": self.duplicates,
            "similar_pairs": self.similar_pairs,
            "total_comparisons": len(self.comparisons),
            "duplicate_count": len(self.duplicates),
            "similar_count": len(self.similar_pairs),
        }


class AudioComparisonExtension:
    """Audio comparison extension using CLAP embeddings."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._model = None
        self._processor = None

    @property
    def name(self) -> str:
        return "audio_comparison"

    @property
    def display_name(self) -> str:
        return "Audio Comparison"

    def _load_model(self):
        """Load the CLAP model (lazy loading)."""
        if self._model is not None:
            return self._model, self._processor

        try:
            from sentence_transformers import SentenceTransformer

            model_name = self.config.get("embedding_model", "laion/larger_clap_music")

            logger.info(f"Loading CLAP model: {model_name}")
            model = SentenceTransformer(model_name)

            if self.config.get("use_gpu", True):
                model = model.to("cuda")
                logger.info("CLAP model moved to GPU")

            self._model = model
            self._processor = None  # SentenceTransformer handles this internally
            return self._model, self._processor

        except ImportError:
            logger.error("sentence-transformers not installed")
            raise RuntimeError("sentence-transformers is required for audio comparison")
        except Exception as e:
            logger.error(f"Failed to load CLAP model: {e}")
            raise

    def _get_embedding(self, audio_path: str | Path) -> np.ndarray:
        """Generate embedding for audio file."""
        audio_path = Path(audio_path)

        # Load model if needed
        model, _ = self._load_model()

        try:
            # Use librosa to load audio
            import librosa

            # Load audio at 48kHz (CLAP's preferred sample rate)
            audio, sr = librosa.load(str(audio_path), sr=48000, duration=30)  # Limit to 30s for efficiency

            # Generate embedding using CLAP
            # CLAP expects audio as a numpy array
            embedding = model.encode_audio(audio)

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding for {audio_path.name}: {e}")
            raise

    def _file_hash(self, file_path: Path) -> str:
        """Generate MD5 hash of file for quick duplicate detection."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def compare_audio(
        self,
        file1: str | Path,
        file2: str | Path,
        similarity_threshold: float | None = None,
        duplicate_threshold: float | None = None,
    ) -> ComparisonResult:
        """
        Compare two audio files for similarity.

        Args:
            file1: Path to first audio file
            file2: Path to second audio file
            similarity_threshold: Threshold for "similar" (default: from config)
            duplicate_threshold: Threshold for "duplicate" (default: from config)

        Returns:
            ComparisonResult
        """
        file1 = Path(file1)
        file2 = Path(file2)

        if not file1.exists():
            raise FileNotFoundError(f"Audio file not found: {file1}")
        if not file2.exists():
            raise FileNotFoundError(f"Audio file not found: {file2}")

        # Use config defaults if not specified
        similarity_threshold = similarity_threshold or self.config.get("similarity_threshold", 0.85)
        duplicate_threshold = duplicate_threshold or self.config.get("duplicate_threshold", 0.95)

        logger.info(f"Comparing: {file1.name} vs {file2.name}")

        try:
            # Quick check: file hash (exact duplicates)
            hash1 = self._file_hash(file1)
            hash2 = self._file_hash(file2)

            if hash1 == hash2:
                return ComparisonResult(
                    file1=str(file1),
                    file2=str(file2),
                    similarity=1.0,
                    is_duplicate=True,
                    is_similar=True,
                    metadata={"method": "file_hash"},
                )

            # Generate embeddings
            emb1 = self._get_embedding(file1)
            emb2 = self._get_embedding(file2)

            # Calculate cosine similarity
            similarity = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

            is_duplicate = similarity >= duplicate_threshold
            is_similar = similarity >= similarity_threshold

            result = ComparisonResult(
                file1=str(file1),
                file2=str(file2),
                similarity=similarity,
                is_duplicate=is_duplicate,
                is_similar=is_similar,
                metadata={
                    "method": "clap_embedding",
                    "model": self.config.get("embedding_model", "laion/larger_clap_music"),
                },
            )

            logger.info(f"Similarity: {similarity:.4f} (similar={is_similar}, duplicate={is_duplicate})")
            return result

        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            raise

    def compare_batch(
        self,
        file_list: list[str | Path],
        similarity_threshold: float | None = None,
        duplicate_threshold: float | None = None,
    ) -> BatchComparisonResult:
        """
        Compare multiple audio files against each other.

        Args:
            file_list: List of audio file paths
            similarity_threshold: Threshold for "similar" (default: from config)
            duplicate_threshold: Threshold for "duplicate" (default: from config)

        Returns:
            BatchComparisonResult with all pairwise comparisons
        """
        if len(file_list) < 2:
            raise ValueError("Need at least 2 files to compare")

        logger.info(f"Batch comparing {len(file_list)} files")

        comparisons = []
        duplicates = []
        similar_pairs = []

        # Compare all pairs
        for i in range(len(file_list)):
            for j in range(i + 1, len(file_list)):
                result = self.compare_audio(
                    file_list[i],
                    file_list[j],
                    similarity_threshold,
                    duplicate_threshold,
                )
                comparisons.append(result)

                if result.is_duplicate:
                    duplicates.append((str(file_list[i]), str(file_list[j])))
                elif result.is_similar:
                    similar_pairs.append((str(file_list[i]), str(file_list[j])))

        batch_result = BatchComparisonResult(
            comparisons=comparisons,
            duplicates=duplicates,
            similar_pairs=similar_pairs,
        )

        logger.info(f"Found {len(duplicates)} duplicates, {len(similar_pairs)} similar pairs")
        return batch_result

    def find_duplicates_in_directory(
        self,
        directory: str | Path,
        pattern: str = "*.wav",
        similarity_threshold: float | None = None,
        duplicate_threshold: float | None = None,
    ) -> BatchComparisonResult:
        """
        Find duplicate/similar audio files in a directory.

        Args:
            directory: Directory to search
            pattern: File pattern to match (default: *.wav)
            similarity_threshold: Threshold for "similar" (default: from config)
            duplicate_threshold: Threshold for "duplicate" (default: from config)

        Returns:
            BatchComparisonResult with duplicates and similar files
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        # Find all matching files
        files = list(directory.glob(pattern))
        if not files:
            logger.warning(f"No files found matching pattern: {pattern}")
            return BatchComparisonResult(comparisons=[], duplicates=[], similar_pairs=[])

        logger.info(f"Found {len(files)} files in {directory}")

        # Compare all files
        return self.compare_batch(files, similarity_threshold, duplicate_threshold)


def register_extension(config: dict | None = None) -> AudioComparisonExtension:
    """Register the audio comparison extension."""
    return AudioComparisonExtension(config)


# Convenience functions for quick usage
def compare_audio_files(
    file1: str | Path,
    file2: str | Path,
    use_gpu: bool = True,
) -> ComparisonResult:
    """
    Quick function to compare two audio files.

    Args:
        file1: Path to first audio file
        file2: Path to second audio file
        use_gpu: Whether to use GPU if available

    Returns:
        ComparisonResult
    """
    config = {"use_gpu": use_gpu}
    extension = register_extension(config)
    return extension.compare_audio(file1, file2)


def find_duplicates(
    directory: str | Path,
    pattern: str = "*.wav",
    use_gpu: bool = True,
) -> BatchComparisonResult:
    """
    Quick function to find duplicate audio files in a directory.

    Args:
        directory: Directory to search
        pattern: File pattern to match (default: *.wav)
        use_gpu: Whether to use GPU if available

    Returns:
        BatchComparisonResult
    """
    config = {"use_gpu": use_gpu}
    extension = register_extension(config)
    return extension.find_duplicates_in_directory(directory, pattern)
