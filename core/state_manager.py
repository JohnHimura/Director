import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from . import constants as C # Assuming constants.py is in the same directory or accessible

logger = logging.getLogger(__name__)

DB_FILE_DEFAULT_NAME = "bot_state.sqlite"

class StateManager:
    def __init__(self, db_file_name: Optional[str] = None):
        # Determine database file path (e.g., in a 'data' subdirectory or project root)
        # For simplicity, placing it in the project root for now.
        # A more robust solution might use a dedicated data directory.
        self.db_file = Path(db_file_name or DB_FILE_DEFAULT_NAME).resolve()
        self.db_file.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists

        logger.info(f"StateManager initialized with database file: {self.db_file}")
        self._create_tables()

    def _connect(self) -> sqlite3.Connection:
        """Establishes a connection to the SQLite database."""
        try:
            # `check_same_thread=False` is important if StateManager methods
            # might be called from different threads (e.g. main bot thread and shutdown handler thread)
            # For simple sequential use, it's less critical but good practice for libraries.
            conn = sqlite3.connect(self.db_file, check_same_thread=False)
            conn.row_factory = sqlite3.Row # Access columns by name
            return conn
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database {self.db_file}: {e}", exc_info=True)
            raise # Re-raise after logging, or handle more gracefully depending on requirements

    def _create_tables(self) -> None:
        """Creates the necessary tables in the database if they don't already exist."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()

                # Open Positions Table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS open_positions (
                    ticket INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    volume REAL NOT NULL,
                    open_price REAL NOT NULL,
                    open_time INTEGER NOT NULL,
                    position_type INTEGER NOT NULL,
                    sl REAL,
                    tp REAL,
                    magic_number INTEGER,
                    comment TEXT
                )
                """)

                # Bot Variables Table (Key-Value Store)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS bot_variables (
                    key TEXT PRIMARY KEY,
                    value_text TEXT,
                    value_real REAL,
                    value_int INTEGER
                )
                """)
                conn.commit()
                logger.info("Database tables checked/created successfully.")
        except sqlite3.Error as e:
            logger.error(f"Error creating database tables: {e}", exc_info=True)
            # Depending on severity, this might be a critical error
            raise

    # --- Position Management Methods ---
    def save_open_positions(self, positions: List[Dict[str, Any]]) -> None:
        """Clears and saves all currently open positions to the database."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM open_positions") # Clear existing

                for pos in positions:
                    # Ensure all required keys are present, provide defaults if some are optional
                    # In MT5, sl/tp can be 0.0 if not set.
                    cursor.execute("""
                    INSERT INTO open_positions (
                        ticket, symbol, volume, open_price, open_time, position_type,
                        sl, tp, magic_number, comment
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        pos.get(C.POSITION_TICKET), pos.get(C.POSITION_SYMBOL), pos.get(C.POSITION_VOLUME),
                        pos.get(C.POSITION_OPEN_PRICE), pos.get(C.POSITION_TIME), pos.get(C.POSITION_TYPE),
                        pos.get(C.POSITION_SL, 0.0), pos.get(C.POSITION_TP, 0.0),
                        pos.get(C.POSITION_MAGIC), pos.get(C.POSITION_COMMENT)
                    ))
                conn.commit()
                logger.info(f"Saved {len(positions)} open positions to database.")
        except sqlite3.Error as e:
            logger.error(f"Error saving open positions: {e}", exc_info=True)

    def load_open_positions(self) -> List[Dict[str, Any]]:
        """Loads all open positions from the database."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM open_positions")
                rows = cursor.fetchall()
                # Convert sqlite3.Row objects to dictionaries
                positions = [dict(row) for row in rows]
                logger.info(f"Loaded {len(positions)} open positions from database.")
                return positions
        except sqlite3.Error as e:
            logger.error(f"Error loading open positions: {e}", exc_info=True)
            return [] # Return empty list on error

    def add_position(self, position_dict: Dict[str, Any]) -> None:
        """Adds a single new position to the database."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT OR REPLACE INTO open_positions (
                    ticket, symbol, volume, open_price, open_time, position_type,
                    sl, tp, magic_number, comment
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    position_dict.get(C.POSITION_TICKET), position_dict.get(C.POSITION_SYMBOL),
                    position_dict.get(C.POSITION_VOLUME), position_dict.get(C.POSITION_OPEN_PRICE),
                    position_dict.get(C.POSITION_TIME), position_dict.get(C.POSITION_TYPE),
                    position_dict.get(C.POSITION_SL, 0.0), position_dict.get(C.POSITION_TP, 0.0),
                    position_dict.get(C.POSITION_MAGIC), position_dict.get(C.POSITION_COMMENT)
                ))
                conn.commit()
                logger.info(f"Added/Replaced position {position_dict.get(C.POSITION_TICKET)} in database.")
        except sqlite3.Error as e:
            logger.error(f"Error adding position {position_dict.get(C.POSITION_TICKET)}: {e}", exc_info=True)

    def remove_position(self, ticket: int) -> None:
        """Removes a position from the database by its ticket."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM open_positions WHERE ticket = ?", (ticket,))
                conn.commit()
                if cursor.rowcount > 0:
                    logger.info(f"Removed position {ticket} from database.")
                else:
                    logger.warning(f"Attempted to remove position {ticket}, but it was not found in database.")
        except sqlite3.Error as e:
            logger.error(f"Error removing position {ticket}: {e}", exc_info=True)

    def update_position_sl_tp(self, ticket: int, sl: Optional[float], tp: Optional[float]) -> None:
        """Updates the SL and TP for a given position ticket."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                UPDATE open_positions
                SET sl = ?, tp = ?
                WHERE ticket = ?
                """, (sl if sl is not None else 0.0, tp if tp is not None else 0.0, ticket))
                conn.commit()
                if cursor.rowcount > 0:
                    logger.info(f"Updated SL/TP for position {ticket} in database. SL: {sl}, TP: {tp}")
                else:
                    logger.warning(f"Attempted to update SL/TP for position {ticket}, but it was not found.")
        except sqlite3.Error as e:
            logger.error(f"Error updating SL/TP for position {ticket}: {e}", exc_info=True)

    # --- Bot Variable Methods ---
    def save_variable(self, key: str, value: Any) -> None:
        """Saves a variable to the key-value store. Determines type for storage."""
        value_text, value_real, value_int = None, None, None
        if isinstance(value, str):
            value_text = value
        elif isinstance(value, float):
            value_real = value
        elif isinstance(value, int): # bool is a subclass of int
            value_int = value
        elif isinstance(value, bool): # Explicitly handle bool to store as int (0 or 1)
             value_int = int(value)
        elif value is None: # Store None as NULL in all type-specific columns
            pass
        else: # For other types (list, dict), serialize to JSON string
            try:
                value_text = json.dumps(value)
            except TypeError as e:
                logger.error(f"Could not serialize value for key '{key}' to JSON: {e}. Value not saved.", exc_info=True)
                return

        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT OR REPLACE INTO bot_variables (key, value_text, value_real, value_int)
                VALUES (?, ?, ?, ?)
                """, (key, value_text, value_real, value_int))
                conn.commit()
                logger.debug(f"Saved variable '{key}': {value}")
        except sqlite3.Error as e:
            logger.error(f"Error saving variable '{key}': {e}", exc_info=True)

    def load_variable(self, key: str, default_value: Any = None) -> Any:
        """
        Loads a variable from the key-value store.
        Tries to deserialize JSON strings if applicable.
        """
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT value_text, value_real, value_int FROM bot_variables WHERE key = ?", (key,))
                row = cursor.fetchone()

                if row:
                    if row["value_int"] is not None:
                        # Could be an int or a bool. If we need to distinguish, more info is needed
                        # For now, assume if it was saved as int (e.g. from bool), it's fine to return as int.
                        # Or, add type hinting during save, e.g. another column 'value_type'
                        # For this iteration, a common use case is peak_equity (float) or flags (bool as int)
                        # or date string (text).
                        # Example: if key implies boolean, cast: if key == C.SOME_BOOLEAN_FLAG_KEY: return bool(row["value_int"])
                        return row["value_int"]
                    if row["value_real"] is not None:
                        return row["value_real"]
                    if row["value_text"] is not None:
                        try:
                            # Attempt to deserialize if it's JSON (e.g., for lists/dicts)
                            return json.loads(row["value_text"])
                        except json.JSONDecodeError:
                            # If not JSON, return as plain text (e.g. date string)
                            return row["value_text"]
                    # If all are NULL (e.g. value was None)
                    return None
                else:
                    logger.debug(f"Variable '{key}' not found, returning default: {default_value}")
                    return default_value
        except sqlite3.Error as e:
            logger.error(f"Error loading variable '{key}': {e}. Returning default.", exc_info=True)
            return default_value

    def remove_variable(self, key: str) -> None:
        """Removes a variable from the key-value store."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM bot_variables WHERE key = ?", (key,))
                conn.commit()
                if cursor.rowcount > 0:
                    logger.info(f"Removed variable '{key}' from database.")
                else:
                    logger.debug(f"Attempted to remove variable '{key}', but it was not found.")
        except sqlite3.Error as e:
            logger.error(f"Error removing variable '{key}': {e}", exc_info=True)

    def clear_all_state(self) -> None:
        """Removes all data from all tables. For testing or full reset."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM open_positions")
                cursor.execute("DELETE FROM bot_variables")
                conn.commit()
                logger.info("Cleared all state from database (open_positions, bot_variables).")
        except sqlite3.Error as e:
            logger.error(f"Error clearing all state: {e}", exc_info=True)

```
