"""Thread-safe SQLite connection pool with async support.

Provides efficient database connection management for high-concurrency scenarios.
Uses thread-local storage with connection pooling and async operations via aiosqlite.
"""

import asyncio
import logging
import sqlite3
import threading
import time
from collections import deque
from contextlib import asynccontextmanager, contextmanager
from functools import partial
from pathlib import Path
from typing import Optional, Callable, Any, AsyncIterator, Iterator

logger = logging.getLogger(__name__)


class SQLiteConnectionPool:
    """Thread-safe SQLite connection pool with async support.

    Features:
    - Connection pooling with configurable size
    - Thread-local connections for thread safety
    - WAL mode for better concurrency
    - Connection health checks
    - Automatic reconnection on failure
    - Async support via aiosqlite
    """

    def __init__(
        self,
        db_path: str | Path,
        pool_size: int = 5,
        max_overflow: int = 10,
        timeout: float = 30.0,
        enable_wal: bool = True,
        check_same_thread: bool = False,
    ):
        """Initialize the connection pool.

        Args:
            db_path: Path to SQLite database file
            pool_size: Base number of connections to maintain
            max_overflow: Additional connections allowed under load
            timeout: Maximum time to wait for a connection
            enable_wal: Enable WAL mode for better concurrency
            check_same_thread: Disable check_same_thread for pool usage
        """
        self.db_path = str(db_path)
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.timeout = timeout
        self.enable_wal = enable_wal

        # Thread-local storage for connections
        self._local = threading.local()

        # Connection pool (for overflow)
        self._pool: deque[sqlite3.Connection] = deque()
        self._pool_lock = threading.Lock()

        # Track created connections
        self._created_connections = 0
        self._connection_lock = threading.Lock()

        # Initialize database
        self._init_db()

        logger.info(
            f"SQLite connection pool initialized: {db_path} "
            f"(pool_size={pool_size}, max_overflow={max_overflow})"
        )

    def _init_db(self):
        """Initialize database with optimal settings."""
        conn = self._create_connection()
        try:
            # Enable WAL mode for better concurrency
            if self.enable_wal:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")

            # Optimize for performance
            conn.execute("PRAGMA busy_timeout=30000")  # 30 seconds
            conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
            conn.execute("PRAGMA temp_store=MEMORY")  # In-memory temp tables
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB mmap

            logger.info("Database initialized with optimized settings")
        finally:
            self._return_connection(conn)

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection with optimal settings."""
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=self.timeout,
        )
        conn.row_factory = sqlite3.Row
        return conn

    def _get_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool or create a new one."""
        # Try to get thread-local connection first
        if hasattr(self._local, "connection"):
            conn = self._local.connection
            if self._is_connection_healthy(conn):
                return conn
            # Remove unhealthy connection
            delattr(self._local, "connection")

        # Try to get from pool
        with self._pool_lock:
            if self._pool:
                conn = self._pool.popleft()
                if self._is_connection_healthy(conn):
                    # Store in thread-local
                    self._local.connection = conn
                    return conn
                else:
                    # Close unhealthy connection
                    try:
                        conn.close()
                    except Exception:
                        pass

        # Create new connection if under limit
        with self._connection_lock:
            if self._created_connections < (self.pool_size + self.max_overflow):
                conn = self._create_connection()
                self._created_connections += 1
                self._local.connection = conn
                return conn

        # Wait for available connection
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            with self._pool_lock:
                if self._pool:
                    conn = self._pool.popleft()
                    if self._is_connection_healthy(conn):
                        self._local.connection = conn
                        return conn
            time.sleep(0.01)

        raise TimeoutError(f"Could not acquire database connection within {self.timeout}s")

    def _return_connection(self, conn: sqlite3.Connection) -> None:
        """Return a connection to the pool."""
        # Keep in thread-local if possible
        if hasattr(self._local, "connection") and self._local.connection is conn:
            return

        # Return to pool
        with self._pool_lock:
            if len(self._pool) < self.pool_size:
                self._pool.append(conn)
                return

        # Close if pool is full
        try:
            conn.close()
            with self._connection_lock:
                self._created_connections -= 1
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")

    def _is_connection_healthy(self, conn: sqlite3.Connection) -> bool:
        """Check if a connection is healthy."""
        try:
            conn.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False

    @contextmanager
    def get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get a connection from the pool (context manager)."""
        conn = self._get_connection()
        try:
            yield conn
        finally:
            self._return_connection(conn)

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a SQL statement and return the cursor."""
        with self.get_connection() as conn:
            return conn.execute(sql, params)

    def executemany(self, sql: str, params_list: list[tuple]) -> sqlite3.Cursor:
        """Execute a SQL statement with multiple parameter sets."""
        with self.get_connection() as conn:
            return conn.executemany(sql, params_list)

    def executescript(self, script: str) -> sqlite3.Cursor:
        """Execute a SQL script."""
        with self.get_connection() as conn:
            return conn.executescript(script)

    def commit(self) -> None:
        """Commit transactions on all connections."""
        # Commit thread-local connection
        if hasattr(self._local, "connection"):
            try:
                self._local.connection.commit()
            except Exception as e:
                logger.warning(f"Error committing: {e}")

        # Commit pooled connections
        with self._pool_lock:
            for conn in self._pool:
                try:
                    conn.commit()
                except Exception as e:
                    logger.warning(f"Error committing pooled connection: {e}")

    def close(self) -> None:
        """Close all connections in the pool."""
        logger.info("Closing SQLite connection pool...")

        # Close thread-local connection
        if hasattr(self._local, "connection"):
            try:
                self._local.connection.close()
            except Exception:
                pass
            delattr(self._local, "connection")

        # Close pooled connections
        with self._pool_lock:
            while self._pool:
                conn = self._pool.popleft()
                try:
                    conn.close()
                except Exception:
                    pass

        logger.info("SQLite connection pool closed")


class AsyncSQLiteConnectionPool:
    """Async SQLite connection pool using aiosqlite.

    Provides async database operations for better performance
    in async contexts like FastAPI endpoints.
    """

    def __init__(
        self,
        db_path: str | Path,
        pool_size: int = 10,
        timeout: float = 30.0,
        enable_wal: bool = True,
    ):
        """Initialize the async connection pool.

        Args:
            db_path: Path to SQLite database file
            pool_size: Number of connections to maintain
            timeout: Maximum time to wait for a connection
            enable_wal: Enable WAL mode for better concurrency
        """
        self.db_path = str(db_path)
        self.pool_size = pool_size
        self.timeout = timeout
        self.enable_wal = enable_wal

        # Connection pool
        self._pool: asyncio.Queue[Optional["aiosqlite.Connection"]] = asyncio.Queue(maxsize=pool_size)
        self._created = 0
        self._lock = asyncio.Lock()

        logger.info(f"Async SQLite connection pool initialized: {db_path}")

    async def _create_connection(self) -> "aiosqlite.Connection":
        """Create a new aiosqlite connection."""
        import aiosqlite

        conn = await aiosqlite.connect(self.db_path)
        await conn.execute("PRAGMA foreign_keys = ON")

        if self.enable_wal:
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA synchronous=NORMAL")

        await conn.execute("PRAGMA busy_timeout=30000")
        await conn.execute("PRAGMA cache_size=-64000")

        return conn

    async def _get_connection(self) -> "aiosqlite.Connection":
        """Get a connection from the pool or create a new one."""
        try:
            # Try to get from pool (non-blocking)
            conn = self._pool.get_nowait()
            if conn is not None:
                return conn
        except asyncio.QueueEmpty:
            pass

        # Create new connection if under limit
        async with self._lock:
            if self._created < self.pool_size:
                self._created += 1
                return await self._create_connection()

        # Wait for available connection
        try:
            conn = await asyncio.wait_for(self._pool.get(), timeout=self.timeout)
            if conn is not None:
                return conn
        except asyncio.TimeoutError:
            raise TimeoutError(f"Could not acquire database connection within {self.timeout}s")

        raise RuntimeError("No available database connection")

    async def _return_connection(self, conn: "aiosqlite.Connection") -> None:
        """Return a connection to the pool."""
        try:
            self._pool.put_nowait(conn)
        except asyncio.QueueFull:
            # Pool is full, close the connection
            asyncio.create_task(self._close_connection(conn))
            async with self._lock:
                self._created -= 1

    async def _close_connection(self, conn: "aiosqlite.Connection") -> None:
        """Close a connection."""
        try:
            await conn.close()
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator["aiosqlite.Connection"]:
        """Acquire a connection from the pool (async context manager)."""
        conn = await self._get_connection()
        try:
            yield conn
        finally:
            await self._return_connection(conn)

    async def execute(
        self,
        sql: str,
        params: tuple = (),
    ) -> "aiosqlite.Cursor":
        """Execute a SQL statement."""
        async with self.acquire() as conn:
            return await conn.execute(sql, params)

    async def executemany(
        self,
        sql: str,
        params_list: list[tuple],
    ) -> "aiosqlite.Cursor":
        """Execute a SQL statement with multiple parameter sets."""
        async with self.acquire() as conn:
            return await conn.executemany(sql, params_list)

    async def executescript(self, script: str) -> "aiosqlite.Cursor":
        """Execute a SQL script."""
        async with self.acquire() as conn:
            return await conn.executescript(script)

    async def close(self) -> None:
        """Close all connections in the pool."""
        logger.info("Closing async SQLite connection pool...")

        while not self._pool.empty():
            conn = await self._pool.get()
            if conn is not None:
                await self._close_connection(conn)

        logger.info("Async SQLite connection pool closed")


# Global pool instances
_pools: dict[str, SQLiteConnectionPool] = {}
_pools_lock = threading.Lock()
_async_pools: dict[str, AsyncSQLiteConnectionPool] = {}


def get_pool(
    db_path: str | Path,
    pool_size: int = 5,
    **kwargs
) -> SQLiteConnectionPool:
    """Get or create a connection pool for the given database path.

    Args:
        db_path: Path to SQLite database
        pool_size: Base pool size
        **kwargs: Additional arguments for SQLiteConnectionPool

    Returns:
        SQLiteConnectionPool instance
    """
    db_path_str = str(db_path)

    with _pools_lock:
        if db_path_str not in _pools:
            _pools[db_path_str] = SQLiteConnectionPool(
                db_path=db_path_str,
                pool_size=pool_size,
                **kwargs
            )
        return _pools[db_path_str]


def get_async_pool(
    db_path: str | Path,
    pool_size: int = 10,
    **kwargs
) -> AsyncSQLiteConnectionPool:
    """Get or create an async connection pool for the given database path.

    Args:
        db_path: Path to SQLite database
        pool_size: Pool size
        **kwargs: Additional arguments for AsyncSQLiteConnectionPool

    Returns:
        AsyncSQLiteConnectionPool instance
    """
    db_path_str = str(db_path)

    if db_path_str not in _async_pools:
        _async_pools[db_path_str] = AsyncSQLiteConnectionPool(
            db_path=db_path_str,
            pool_size=pool_size,
            **kwargs
        )
    return _async_pools[db_path_str]


async def close_all_pools() -> None:
    """Close all connection pools."""
    for pool in _pools.values():
        pool.close()
    for pool in _async_pools.values():
        await pool.close()
    _pools.clear()
    _async_pools.clear()
