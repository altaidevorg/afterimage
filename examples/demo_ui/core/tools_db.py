"""
SQLite storage for custom tool definitions.

Provides CRUD operations for storing and retrieving FunctionDefinition objects.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .function_parser import FunctionDefinition, ParsedFunction


# Default database path
DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "custom_tools.db"


class ToolsDatabase:
    """SQLite database for storing custom tool definitions."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the database connection.
        
        Args:
            db_path: Path to SQLite database file. If None, uses default path.
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self):
        """Initialize database tables."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS custom_tools (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    parameters TEXT,
                    required TEXT,
                    source_code TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def save_tool(self, parsed: ParsedFunction) -> bool:
        """
        Save a parsed function to the database.
        
        Args:
            parsed: ParsedFunction containing FunctionDefinition and source code
            
        Returns:
            True if saved successfully, False if error occurred
        """
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO custom_tools 
                    (name, description, parameters, required, source_code, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        parsed.definition.name,
                        parsed.definition.description,
                        json.dumps(parsed.definition.parameters),
                        json.dumps(parsed.definition.required),
                        parsed.source_code,
                        datetime.now().isoformat(),
                    ),
                )
                conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return False
    
    def get_all_tools(self) -> List[ParsedFunction]:
        """
        Get all tool definitions from the database.
        
        Returns:
            List of ParsedFunction objects
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT name, description, parameters, required, source_code FROM custom_tools ORDER BY name"
            )
            rows = cursor.fetchall()
        
        results = []
        for row in rows:
            func_def = FunctionDefinition(
                name=row["name"],
                description=row["description"] or "",
                parameters=json.loads(row["parameters"]) if row["parameters"] else {},
                required=json.loads(row["required"]) if row["required"] else [],
            )
            results.append(ParsedFunction(
                definition=func_def,
                source_code=row["source_code"] or "",
            ))
        return results
    
    def get_tool(self, name: str) -> Optional[ParsedFunction]:
        """
        Get a specific tool by name.
        
        Args:
            name: Tool name to retrieve
            
        Returns:
            ParsedFunction if found, None otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT name, description, parameters, required, source_code FROM custom_tools WHERE name = ?",
                (name,),
            )
            row = cursor.fetchone()
        
        if row is None:
            return None
        
        func_def = FunctionDefinition(
            name=row["name"],
            description=row["description"] or "",
            parameters=json.loads(row["parameters"]) if row["parameters"] else {},
            required=json.loads(row["required"]) if row["required"] else [],
        )
        return ParsedFunction(
            definition=func_def,
            source_code=row["source_code"] or "",
        )
    
    def delete_tool(self, name: str) -> bool:
        """
        Delete a tool by name.
        
        Args:
            name: Tool name to delete
            
        Returns:
            True if deleted, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM custom_tools WHERE name = ?",
                (name,),
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def tool_exists(self, name: str) -> bool:
        """
        Check if a tool exists.
        
        Args:
            name: Tool name to check
            
        Returns:
            True if exists, False otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM custom_tools WHERE name = ?",
                (name,),
            )
            return cursor.fetchone() is not None
    
    def get_tool_names(self) -> List[str]:
        """
        Get list of all tool names.
        
        Returns:
            List of tool names
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT name FROM custom_tools ORDER BY name")
            return [row["name"] for row in cursor.fetchall()]
    
    def clear_all(self) -> int:
        """
        Delete all tools from the database.
        
        Returns:
            Number of tools deleted
        """
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM custom_tools")
            conn.commit()
            return cursor.rowcount


# Singleton instance for easy access
_db_instance: Optional[ToolsDatabase] = None


def get_tools_db() -> ToolsDatabase:
    """Get the singleton database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = ToolsDatabase()
    return _db_instance
