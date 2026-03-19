"""Grid search utility for hyperparameter tuning.

Generates config combinations for grid search experiments.
Based on the experiment_runner from semantic-rag-system.

ARCHITECTURE:
- Data models: GridSearchResult, GridSearchExperiment (domain logic - stay here)
- Local SQLite: GridSearchDatabase (appropriate for local experiment data)
- Config/Result storage: Uses context_engine S3Backend when available
- File I/O: Delegated to context_engine tools via registry

This provides both:
1. Local SQLite for fast queries on experiment results
2. Context Engine storage for persistent config/result files
"""

import itertools
import json
import sqlite3
import uuid
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
import csv

# Import path constants
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from src.constants import DATA_DIR, ROOT_DATA_DIR, DATA_OUTPUTS_DIR


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class GridSearchResult:
    """Structured result from a single grid search run."""
    experiment_id: str
    config_path: str
    config_hash: str
    query: str
    status: str  # "success" or "error"
    execution_time: Optional[float] = None
    output_length: Optional[int] = None
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class GridSearchExperiment:
    """Metadata for a grid search experiment."""
    experiment_id: str
    start_time: str
    end_time: Optional[str] = None
    base_config: Optional[Dict[str, Any]] = None
    param_grid: Optional[Dict[str, List[Any]]] = None
    num_configs: int = 0
    num_queries: int = 0
    status: str = "running"  # "running", "completed", "failed"
    results_file: Optional[str] = None

    def __post_init__(self):
        if self.experiment_id is None:
            self.experiment_id = str(uuid.uuid4())
        if self.start_time is None:
            self.start_time = datetime.now().isoformat()


# =============================================================================
# DATABASE PERSISTENCE (Local SQLite - appropriate for local experiment data)
# =============================================================================

class GridSearchDatabase:
    """SQLite database for grid search results persistence.

    This is LOCAL experiment data, appropriate for SQLite.
    Not moved to context_engine as it's not shared context.

    Provides efficient querying and historical analysis of grid search results.
    """

    def __init__(self, db_path: Union[str, Path] = None):
        """Initialize the database connection.

        Args:
            db_path: Path to SQLite database file (defaults to ROOT_DATA_DIR/grid_search.db)
        """
        if db_path is None:
            db_path = ROOT_DATA_DIR / "grid_search.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    base_config TEXT,
                    param_grid TEXT,
                    num_configs INTEGER DEFAULT 0,
                    num_queries INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'running',
                    results_file TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    config_path TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    query TEXT NOT NULL,
                    status TEXT NOT NULL,
                    execution_time REAL,
                    output_length INTEGER,
                    error TEXT,
                    metrics TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiment_id
                ON results (experiment_id)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_config_hash
                ON results (config_hash)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status
                ON results (status)
            """)

            conn.commit()

    def save_experiment(self, experiment: GridSearchExperiment) -> None:
        """Save experiment metadata to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO experiments
                (experiment_id, start_time, end_time, base_config, param_grid,
                 num_configs, num_queries, status, results_file)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment.experiment_id,
                experiment.start_time,
                experiment.end_time,
                json.dumps(experiment.base_config) if experiment.base_config else None,
                json.dumps(experiment.param_grid) if experiment.param_grid else None,
                experiment.num_configs,
                experiment.num_queries,
                experiment.status,
                experiment.results_file
            ))
            conn.commit()

    def save_result(self, result: GridSearchResult) -> None:
        """Save a single result to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO results
                (experiment_id, config_path, config_hash, query, status,
                 execution_time, output_length, error, metrics, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.experiment_id,
                result.config_path,
                result.config_hash,
                result.query,
                result.status,
                result.execution_time,
                result.output_length,
                result.error,
                json.dumps(result.metrics) if result.metrics else None,
                result.timestamp
            ))
            conn.commit()

    def get_experiment_results(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Get all results for an experiment."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM results
                WHERE experiment_id = ?
                ORDER BY timestamp
            """, (experiment_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_best_configs(
        self,
        metric: str = "execution_time",
        limit: int = 10,
        minimize: bool = True
    ) -> List[Dict[str, Any]]:
        """Get best performing configs across all experiments."""
        order = "ASC" if minimize else "DESC"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(f"""
                SELECT r.*, e.base_config, e.param_grid
                FROM results r
                JOIN experiments e ON r.experiment_id = e.experiment_id
                WHERE r.status = 'success' AND r.{metric} IS NOT NULL
                ORDER BY r.{metric} {order}
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def export_to_csv(self, experiment_id: str, output_path: Union[str, Path]) -> None:
        """Export experiment results to CSV file."""
        results = self.get_experiment_results(experiment_id)

        with open(output_path, 'w', newline='') as f:
            if not results:
                return

            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)


# =============================================================================
# CONTEXT ENGINE STORAGE HELPERS
# =============================================================================

def _get_minio_backend():
    """Get MinioBackend via context_engine or direct connection."""
    try:
        from src.shared.tools.registry import get_tool
        save_raw_file = get_tool("save_raw_file")
        if save_raw_file:
            return None, save_raw_file  # Use tool
    except ImportError:
        pass

    # Fallback: Direct MinioBackend
    try:
        from ..storage.minio_helpers import get_minio_backend
        return get_minio_backend(prefix="experiments"), None
    except ImportError:
        return None, None


# Keep old function name for backwards compatibility
def _get_s3_backend():
    """Get backend via context_engine or direct connection.

    Deprecated: Use _get_minio_backend() instead.
    """
    return _get_minio_backend()


def _save_to_context_engine(path: str, content: str) -> str:
    """Save content to context_engine storage.

    Args:
        path: Storage path (e.g., "/experiments/exp_123/config.yaml")
        content: Content to save

    Returns:
        Path where content was stored
    """
    backend, save_tool = _get_minio_backend()

    if save_tool:
        # Use tool (for async context, this is sync wrapper)
        try:
            result = save_tool.invoke({"path": path, "content": content})
            return path
        except Exception:
            pass  # Fall through to direct backend

    if backend:
        result = backend.write(path, content)
        if not result.error:
            return path

    # Last resort: local file
    local_path = ROOT_DATA_DIR / path.lstrip("/")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(content)
    return str(local_path)


def _read_from_context_engine(path: str) -> Optional[str]:
    """Read content from context_engine storage.

    Args:
        path: Storage path

    Returns:
        Content string or None if not found
    """
    backend, _ = _get_minio_backend()

    if backend:
        try:
            # Use s3_backend directly for raw content (no line numbers)
            return backend.s3_backend.read(path)
        except Exception:
            pass

    # Fallback: local file
    local_path = ROOT_DATA_DIR / path.lstrip("/")
    if local_path.exists():
        return local_path.read_text()

    return None


# =============================================================================
# CONFIG GENERATION AND STORAGE
# =============================================================================

def generate_grid_configs(
    base_config: Dict[str, Any],
    param_grid: Dict[str, List[Any]]
) -> List[Dict[str, Any]]:
    """Generate all config combinations for grid search.

    Args:
        base_config: Base configuration dictionary
        param_grid: Parameter grid
            e.g., {
                "model_configs.spawner.temperature": [0.3, 0.5, 0.7],
                "spawner_config.max_researchers": [2, 3, 4]
            }

    Returns:
        List of config combinations
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))

    configs = []
    for combo in combinations:
        config = _deep_copy_dict(base_config)
        for param_name, param_value in zip(param_names, combo):
            _set_nested_value(config, param_name.split('.'), param_value)
        configs.append(config)

    return configs


def save_grid_configs(
    configs: List[Dict[str, Any]],
    output_dir: str = None,
    base_name: str = "config",
    use_context_engine: bool = True
) -> List[str]:
    """Save grid search configs to storage.

    Uses context_engine S3Backend when available, with local file fallback.

    Args:
        configs: List of config dictionaries
        output_dir: Output directory (defaults to data/outputs/experiments/TIMESTAMP)
        base_name: Base name for config files
        use_context_engine: If True, use context_engine for storage

    Returns:
        List of saved config paths
    """
    import yaml

    # Default: data/outputs/experiments/TIMESTAMP/
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = DATA_OUTPUTS_DIR / "experiments" / timestamp
    else:
        output_dir = Path(output_dir)

    config_paths = []

    for i, config in enumerate(configs):
        config_name = f"{base_name}_{i:03d}.yaml"
        content = yaml.dump(config, default_flow_style=False)

        if use_context_engine:
            # Store in context_engine
            ce_path = f"/experiments/{output_dir.name}/configs/{config_name}"
            stored_path = _save_to_context_engine(ce_path, content)
            config_paths.append(stored_path)
        else:
            # Local file storage
            configs_dir = output_dir / "configs"
            configs_dir.mkdir(parents=True, exist_ok=True)
            config_file = configs_dir / config_name
            config_file.write_text(content)
            config_paths.append(str(config_file))

    return config_paths


def get_experiment_dir() -> Path:
    """Get or create the current experiment directory.

    Returns:
        Path to data/outputs/experiments/TIMESTAMP/
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = DATA_OUTPUTS_DIR / "experiments" / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (exp_dir / "configs").mkdir(exist_ok=True)
    (exp_dir / "runs").mkdir(exist_ok=True)

    return exp_dir


def run_grid_search(
    base_config: Dict[str, Any],
    param_grid: Dict[str, List[Any]],
    queries: List[str],
    run_fn: callable,
    output_dir: str = None,
    use_db: bool = True,
    db_path: str = None,
    use_context_engine: bool = True
) -> Dict[str, Any]:
    """Run a grid search experiment with enhanced persistence.

    Args:
        base_config: Base configuration
        param_grid: Parameter grid to search
        queries: Test queries
        run_fn: Function to run a single experiment (receives config, query)
        output_dir: Output directory (defaults to data/outputs/experiments/TIMESTAMP)
        use_db: If True, save results to SQLite database for historical analysis
        db_path: Path to SQLite database file (defaults to ROOT_DATA_DIR/grid_search.db)
        use_context_engine: If True, use context_engine for config/result storage

    Returns:
        Dict with experiment results and metadata
    """
    import hashlib
    import time
    import yaml

    # Set default db_path if not provided
    if db_path is None:
        db_path = str(ROOT_DATA_DIR / "grid_search.db")

    # Create experiment metadata
    experiment_id = str(uuid.uuid4())
    start_time = datetime.now().isoformat()

    # Get/create experiment directory
    if output_dir is None:
        exp_dir = get_experiment_dir()
    else:
        exp_dir = Path(output_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)

    # Initialize database if enabled
    db = None
    if use_db:
        db = GridSearchDatabase(db_path)
        experiment = GridSearchExperiment(
            experiment_id=experiment_id,
            start_time=start_time,
            base_config=base_config,
            param_grid=param_grid,
            num_configs=0,
            num_queries=len(queries),
            status="running"
        )
        db.save_experiment(experiment)

    # Generate and save configs
    configs = generate_grid_configs(base_config, param_grid)
    config_paths = save_grid_configs(configs, str(exp_dir), use_context_engine=use_context_engine)

    # Update experiment with num_configs
    if db:
        experiment.num_configs = len(configs)
        db.save_experiment(experiment)

    # Run experiments
    results = []
    for i, (config_path, config) in enumerate(zip(config_paths, configs)):
        # Compute config hash for tracking
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()

        # Run experiments for all queries
        config_results = []
        for query in queries:
            result_start = time.time()
            try:
                result = run_fn(config, query)
                execution_time = time.time() - result_start

                result_data = {
                    "query": query,
                    "result": result,
                    "execution_time": execution_time
                }
                config_results.append(result_data)

                # Save to database
                if db:
                    db_result = GridSearchResult(
                        experiment_id=experiment_id,
                        config_path=config_path,
                        config_hash=config_hash,
                        query=query,
                        status="success",
                        execution_time=execution_time,
                        output_length=len(str(result)) if result else None,
                        metrics={"output_length": len(str(result)) if result else 0}
                    )
                    db.save_result(db_result)

            except Exception as e:
                execution_time = time.time() - result_start

                result_data = {
                    "query": query,
                    "error": str(e),
                    "execution_time": execution_time
                }
                config_results.append(result_data)

                # Save error to database
                if db:
                    db_result = GridSearchResult(
                        experiment_id=experiment_id,
                        config_path=config_path,
                        config_hash=config_hash,
                        query=query,
                        status="error",
                        execution_time=execution_time,
                        error=str(e)
                    )
                    db.save_result(db_result)

        results.append({
            "config_path": config_path,
            "config": config,
            "config_hash": config_hash,
            "results": config_results
        })

    # Save results to context_engine or local file
    results_content = json.dumps(results, indent=2, default=str)

    if use_context_engine:
        results_path = f"/experiments/{exp_dir.name}/results.json"
        stored_results_path = _save_to_context_engine(results_path, results_content)
    else:
        results_file = exp_dir / "results.json"
        results_file.write_text(results_content)
        stored_results_path = str(results_file)

    # Update experiment as complete
    end_time = datetime.now().isoformat()
    if db:
        experiment.end_time = end_time
        experiment.status = "completed"
        experiment.results_file = stored_results_path
        db.save_experiment(experiment)

    # Return metadata
    return {
        "experiment_id": experiment_id,
        "experiment_dir": str(exp_dir),
        "results_file": stored_results_path,
        "num_configs": len(configs),
        "num_queries": len(queries),
        "start_time": start_time,
        "end_time": end_time,
        "results": results,
        "db_path": str(db.db_path) if db else None
    }


def _deep_copy_dict(obj: Any) -> Any:
    """Deep copy a dictionary (without using copy.deepcopy)."""
    if isinstance(obj, dict):
        return {k: _deep_copy_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_deep_copy_dict(item) for item in obj]
    else:
        return obj


def _set_nested_value(obj: Dict, path: List[str], value: Any) -> None:
    """Set a nested value in a dict using dot notation path."""
    for key in path[:-1]:
        if key not in obj:
            obj[key] = {}
        obj = obj[key]
    obj[path[-1]] = value


# =============================================================================
# PRE-DEFINED GRIDS
# =============================================================================

RAG_PARAM_GRID = {
    "model_configs.spawner.temperature": [0.3, 0.5, 0.7],
    "model_configs.researcher.temperature": [0.5, 0.7, 0.9],
    "spawner_config.max_researchers": [2, 3, 4],
    "spawner_config.max_iterations": [2, 3, 4],
}

TRADING_PARAM_GRID = {
    "model_configs.spawner.temperature": [0.2, 0.4, 0.6],
    "model_configs.researcher.temperature": [0.3, 0.5, 0.7],
    "spawner_config.max_researchers": [2, 3],
}

OSINT_PARAM_GRID = {
    "model_configs.spawner.temperature": [0.4, 0.6],
    "spawner_config.max_researchers": [3, 4],
}
