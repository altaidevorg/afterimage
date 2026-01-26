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
        self.seed_builtin_tools()
    
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
                    category TEXT DEFAULT 'Uncategorized',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            
            # Migration: Add category column if it doesn't exist (for existing databases)
            self._migrate_add_category(conn)
    
    def _migrate_add_category(self, conn: sqlite3.Connection):
        """Add category column to existing databases that don't have it."""
        cursor = conn.execute("PRAGMA table_info(custom_tools)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if "category" not in columns:
            conn.execute("ALTER TABLE custom_tools ADD COLUMN category TEXT DEFAULT 'Uncategorized'")
            conn.commit()
    
    def seed_builtin_tools(self):
        """Seed the database with built-in tools from schemas.py."""
        # Import here to avoid circular dependencies
        from schemas import AVAILABLE_TOOLS
        import inspect
        
        # Built-in tool category mapping
        BUILTIN_CATEGORIES = {
            "turn_on_light": "Smart Home",
            "turn_off_light": "Smart Home",
            "set_thermostat": "Smart Home",
            "play_music": "Smart Home",
            "lock_door": "Smart Home",
            "check_weather": "Smart Home",
        }
        
        print("Seeding built-in tools...")
        for tool_cls in AVAILABLE_TOOLS:
            try:
                # Create a schema from the Pydantic model
                schema = tool_cls.model_json_schema()
                
                # Extract name
                name_prop = schema.get("properties", {}).get("name", {})
                name = name_prop.get("default", tool_cls.__name__)
                
                # Check if tool already exists - skip if it does to preserve customizations
                if self.tool_exists(name):
                    continue
                
                description = schema.get("description", tool_cls.__doc__ or "")
                
                # Extract parameters
                parameters = {}
                required = []
                
                defs = schema.get("$defs", {})
                props = schema.get("properties", {})
                
                if "arguments" in props:
                    args_ref = props["arguments"].get("$ref", "")
                    if args_ref:
                        args_model_name = args_ref.split("/")[-1]
                        
                        if args_model_name in defs:
                            args_schema = defs[args_model_name]
                            parameters = args_schema.get("properties", {})
                            required = args_schema.get("required", [])
                
                clean_params = {
                    "type": "object",
                    "properties": parameters,
                    "required": required
                }
                
                # Get source code
                try:
                    source_code = inspect.getsource(tool_cls)
                except (OSError, TypeError):
                    source_code = f"# Source code not available for {name}"

                # Determine category for built-in tool
                category = BUILTIN_CATEGORIES.get(name, "Uncategorized")

                # Save to DB with category
                print(f"Adding new built-in tool: {name} (category: {category})")
                self.save_tool(
                    ParsedFunction(
                        definition=FunctionDefinition(
                            name=name,
                            description=description,
                            parameters=clean_params,
                            required=required
                        ),
                        source_code=source_code,
                        category=category
                    ),
                    category=category
                )
                
            except Exception as e:
                print(f"Failed to seed tool {tool_cls}: {e}")

    def save_tool(self, parsed: ParsedFunction, category: str = "Uncategorized") -> bool:
        """
        Save a parsed function to the database.
        
        Args:
            parsed: ParsedFunction containing FunctionDefinition and source code
            category: Category name for grouping tools
            
        Returns:
            True if saved successfully, False if error occurred
        """
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO custom_tools 
                    (name, description, parameters, required, source_code, category, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        parsed.definition.name,
                        parsed.definition.description,
                        json.dumps(parsed.definition.parameters),
                        json.dumps(parsed.definition.required),
                        parsed.source_code,
                        category or "Uncategorized",
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
            List of ParsedFunction objects with category info
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT name, description, parameters, required, source_code, category FROM custom_tools ORDER BY category, name"
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
                category=row["category"] or "Uncategorized",
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
                "SELECT name, description, parameters, required, source_code, category FROM custom_tools WHERE name = ?",
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
            category=row["category"] or "Uncategorized",
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
    
    def get_categories(self) -> List[str]:
        """
        Get list of all unique categories.
        
        Returns:
            List of category names, sorted alphabetically
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT DISTINCT category FROM custom_tools WHERE category IS NOT NULL ORDER BY category"
            )
            return [row["category"] for row in cursor.fetchall()]
    
    def get_tools_by_category(self) -> dict[str, List[ParsedFunction]]:
        """
        Get all tools grouped by category.
        
        Returns:
            Dictionary mapping category names to lists of ParsedFunction objects
        """
        tools = self.get_all_tools()
        grouped: dict[str, List[ParsedFunction]] = {}
        
        for tool in tools:
            category = tool.category or "Uncategorized"
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(tool)
        
        # Sort categories with "Uncategorized" at the end
        sorted_grouped = {}
        for cat in sorted(grouped.keys()):
            if cat != "Uncategorized":
                sorted_grouped[cat] = grouped[cat]
        if "Uncategorized" in grouped:
            sorted_grouped["Uncategorized"] = grouped["Uncategorized"]
        
        return sorted_grouped
    
    def update_tool_category(self, name: str, category: str) -> bool:
        """
        Update the category of an existing tool.
        
        Args:
            name: Tool name
            category: New category name
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "UPDATE custom_tools SET category = ? WHERE name = ?",
                    (category or "Uncategorized", name),
                )
                conn.commit()
                return cursor.rowcount > 0
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return False
    
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
