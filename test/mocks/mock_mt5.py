import time
from typing import Dict, List, Optional, Any, Tuple, Union
from unittest.mock import MagicMock # For emulating MT5 objects if needed
import pandas as pd
from datetime import datetime, timezone

# MT5 Constants (can be expanded as needed)
TRADE_RETCODE_DONE = 10009
TRADE_RETCODE_REQUOTE = 10004
TRADE_RETCODE_CONNECTION = 10007 # Example for connection error

ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1

TRADE_ACTION_DEAL = 1 # For opening positions
TRADE_ACTION_SLTP = 2 # For modifying SL/TP

class MockMT5:
    """
    A mock implementation of the MetaTrader5 library for testing purposes.
    """

    # --- Constants for convenience within the mock ---
    ORDER_TYPE_BUY = ORDER_TYPE_BUY
    ORDER_TYPE_SELL = ORDER_TYPE_SELL
    TRADE_ACTION_DEAL = TRADE_ACTION_DEAL
    TRADE_ACTION_SLTP = TRADE_ACTION_SLTP
    # Add other TIMEFRAME constants if needed by tests directly using them
    TIMEFRAME_M1 = 1
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_M30 = 30
    TIMEFRAME_H1 = 16385
    TIMEFRAME_H4 = 16388
    TIMEFRAME_D1 = 16408


    def __init__(self):
        self._initialized = False
        self._connected_login = None # Stores the login if connected
        self._last_error_code = TRADE_RETCODE_DONE
        self._last_error_message = "Success"

        # Internal state for simulation
        self._account_state = {
            'login': 0,
            'name': "Test Account",
            'server': "TestServer",
            'balance': 10000.0,
            'equity': 10000.0,
            'margin': 0.0,
            'margin_free': 10000.0,
            'margin_level': 0.0,
            'currency': "USD"
        }
        self._symbol_states: Dict[str, Dict[str, Any]] = {}
        self._historical_data: Dict[str, Dict[int, pd.DataFrame]] = {}
        self._open_positions: List[Dict[str, Any]] = []
        self._trade_history_deals: List[Dict[str, Any]] = []
        self._next_ticket_id = 1
        self._next_order_id = 1 # Separate from position ticket

        # Configuration for failures
        self.should_initialize_fail = False
        self.should_login_fail = False
        self.should_terminal_info_fail = False
        self.should_account_info_fail = False
        self.order_send_should_fail = False
        self.order_send_fail_retcode = TRADE_RETCODE_CONNECTION
        self.order_send_fail_message = "Mocked order send failure"

        # Mock objects for complex return types
        self.mock_terminal_info_obj = self._create_mock_terminal_info()
        self.mock_account_info_obj = self._create_mock_account_info()

        self.print_logs = False # Set to True to print mock actions

    def _log_action(self, message: str, params: Optional[Dict] = None):
        if self.print_logs:
            log_entry = f"[MockMT5] {message}"
            if params:
                log_entry += f" | Params: {params}"
            print(log_entry)

    def _create_mock_terminal_info(self):
        info = MagicMock()
        info.connected = False # Will be set by initialize/login
        info.dll_path = "mock_mt5_dll_path"
        info.data_path = "mock_mt5_data_path"
        info.login = 0 # Will be set by login
        # Add other attributes as needed by the bot
        return info

    def _create_mock_account_info(self):
        info = MagicMock()
        for key, value in self._account_state.items():
            setattr(info, key, value)
        return info

    def _update_mock_account_info(self):
        for key, value in self._account_state.items():
            setattr(self.mock_account_info_obj, key, value)

    # --- Connection & Info ---
    def initialize(self, path=None, login=None, password=None, server=None, timeout=None, portable=False) -> bool:
        self._log_action("initialize called", {'path': path, 'login': login, 'server': server, 'timeout': timeout, 'portable': portable})
        if self.should_initialize_fail:
            self._last_error_code = 1 # Generic error
            self._last_error_message = "Mocked initialize failure"
            self._initialized = False
            self.mock_terminal_info_obj.connected = False
            return False
        self._initialized = True
        self._last_error_code = TRADE_RETCODE_DONE
        self._last_error_message = "Success"
        # If path is provided, it might be stored or used
        if path: self.mock_terminal_info_obj.path = path
        if portable: self.mock_terminal_info_obj.portable_mode = True
        # Note: Real mt5.initialize() does not log in.
        # self.mock_terminal_info_obj.connected reflects terminal server connection, not login.
        # For simplicity, we'll say initialize makes it "connectable"
        self.mock_terminal_info_obj.connected = True # Terminal is responsive
        return True

    def shutdown(self) -> bool:
        self._log_action("shutdown called")
        self._initialized = False
        self._connected_login = None
        self.mock_terminal_info_obj.connected = False
        self.mock_terminal_info_obj.login = 0
        return True

    def version(self) -> Tuple[int, int, int]:
        self._log_action("version called")
        return (5, 0, 3000) # Example version

    def last_error(self) -> Tuple[int, str]:
        self._log_action(f"last_error called, returning: ({self._last_error_code}, '{self._last_error_message}')")
        return (self._last_error_code, self._last_error_message)

    def terminal_info(self) -> Optional[Any]: # Actually returns TerminalInfo object
        self._log_action("terminal_info called")
        if self.should_terminal_info_fail or not self._initialized:
            self._last_error_code = 2 # Another generic error
            self._last_error_message = "Mocked terminal_info failure or not initialized"
            return None
        # Update connected status based on internal state
        # self.mock_terminal_info_obj.connected = self._initialized # More like terminal server connection
        return self.mock_terminal_info_obj

    def login(self, login: int, password: Optional[str] = None, server: Optional[str] = None, timeout: Optional[int] = None) -> bool:
        self._log_action("login called", {'login': login, 'server': server})
        if not self._initialized:
            self._last_error_code = 3
            self._last_error_message = "Not initialized"
            return False
        if self.should_login_fail:
            self._last_error_code = 4 # Generic login error
            self._last_error_message = "Mocked login failure"
            self._connected_login = None
            self.mock_terminal_info_obj.login = 0
            return False

        # Simulate successful login
        self._connected_login = login
        self._account_state['login'] = login
        if server: self._account_state['server'] = server
        self._update_mock_account_info() # Update mock object with new state
        self.mock_terminal_info_obj.login = login # TerminalInfo also shows login

        self._last_error_code = TRADE_RETCODE_DONE
        self._last_error_message = "Success"
        return True

    def account_info(self) -> Optional[Any]: # Actually returns AccountInfo object
        self._log_action("account_info called")
        if self.should_account_info_fail or not self._initialized or not self._connected_login:
            self._last_error_code = 5 # Error
            self._last_error_message = "Mocked account_info failure or not logged in"
            return None
        # Return the mock object reflecting current state
        return self.mock_account_info_obj

    # --- Configuration methods for tests ---
    def set_last_error(self, code: int, message: str):
        self._last_error_code = code
        self._last_error_message = message

    def set_account_details(self, login: int, name: str, server: str, balance: float, equity: float):
        self._account_state = {
            'login': login, 'name': name, 'server': server,
            'balance': balance, 'equity': equity,
            'margin': self._account_state.get('margin',0.0), # Keep existing or default
            'margin_free': self._account_state.get('margin_free', equity),
            'margin_level': self._account_state.get('margin_level',0.0),
            'currency': self._account_state.get('currency', "USD")
        }
        self._update_mock_account_info() # Apply to mock object
        if self._connected_login == login: # If currently "connected" to this account
             self.mock_terminal_info_obj.login = login
             self.mock_terminal_info_obj.server = server

    # --- Placeholder for other methods to be implemented ---
    def symbol_info(self, symbol_name: str) -> Optional[Any]:
        self._log_action(f"symbol_info called for {symbol_name}")
        if symbol_name in self._symbol_states:
            info = MagicMock()
            for k, v in self._symbol_states[symbol_name].items():
                setattr(info, k, v)
            info.name = symbol_name # Ensure name attribute is set
            return info
        self._set_last_error(6, f"Symbol {symbol_name} not found in mock")
        return None

    def symbol_info_tick(self, symbol_name: str) -> Optional[Any]:
        self._log_action(f"symbol_info_tick called for {symbol_name}")
        if symbol_name in self._symbol_states and 'bid' in self._symbol_states[symbol_name] and 'ask' in self._symbol_states[symbol_name]:
            tick_data = self._symbol_states[symbol_name]
            tick = MagicMock()
            tick.time = int(time.time() * 1000) # Current time in ms
            tick.bid = tick_data['bid']
            tick.ask = tick_data['ask']
            tick.last = tick_data.get('last', (tick_data['bid'] + tick_data['ask']) / 2) # Simulate last
            tick.volume = tick_data.get('volume', 0)
            tick.flags = tick_data.get('flags', 0)
            return tick
        self._set_last_error(7, f"Tick info for {symbol_name} not available in mock")
        return None

    def copy_rates_from_pos(self, symbol: str, timeframe: int, start_pos: int, count: int) -> Optional[List[Tuple]]:
        self._log_action(f"copy_rates_from_pos for {symbol}, TF {timeframe}, Start {start_pos}, Count {count}")
        if symbol in self._historical_data and timeframe in self._historical_data[symbol]:
            df = self._historical_data[symbol][timeframe]
            # Simulate MT5 returning records as tuples: (time, open, high, low, close, tick_volume, spread, real_volume)
            # Ensure DataFrame has these columns or adapt
            required_cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
            # For simplicity, assume df has these columns. If not, this mock needs more robust data generation.
            # Convert time to Unix timestamp for the tuple
            df_slice = df.iloc[start_pos : start_pos + count]

            # Convert to list of tuples in the structure MT5 returns
            # This requires the DataFrame to have columns named exactly as MT5 fields for copy_rates_*
            # MT5 returns: time, open, high, low, close, tick_volume, spread, real_volume
            # Our load_historical_data renames to Time, Open, High, Low, Close, Volume.
            # For mock, we need to provide what MT5 library would return.
            # Let's assume the stored DFs are already in the correct format for this output.
            # If self._historical_data stores DFs with 'Time', 'Open', etc., they need to be converted.
            # For now, assuming it stores them with MT5-like column names.

            # Example: if df has 'time' as datetime, convert to timestamp for output
            # df_slice['time'] = (df_slice['time'] - pd.Timestamp("1970-01-01", tz='UTC')) // pd.Timedelta('1s')
            # This is complex. Let's assume the stored data is a list of tuples already.

            # Simplified: return a list of tuples from the DataFrame rows
            # This requires careful structuring of data in self._historical_data
            # For now, returning a simplified version assuming data is pre-formatted as list of tuples
            # Or, if it's a DataFrame, convert its records.

            if isinstance(df_slice, pd.DataFrame):
                 # Convert DataFrame rows to list of tuples
                 # Ensure the columns are in the order MT5 expects for rates:
                 # time (Unix timestamp), open, high, low, close, tick_volume, spread, real_volume
                 # This part is tricky because the df structure needs to be precise.
                 # For simplicity, returning empty list if not pre-formatted as list of tuples.
                 # A real test setup would have proper data fixtures.
                 # The historical data added via add_historical_data should ideally have:
                 # 'time': datetime objects (will be converted to timestamp)
                 # 'open', 'high', 'low', 'close': numeric
                 # 'tick_volume': numeric (defaults to 0)
                 # 'spread': int (optional, defaults to 0)
                 # 'real_volume': numeric (optional, defaults to 0)
                 output_records = []
                 for _, row in df_slice.iterrows():
                     time_val = row['time']
                     if isinstance(time_val, datetime):
                         timestamp = int(time_val.timestamp())
                     elif pd.api.types.is_datetime64_any_dtype(time_val):
                         # Convert pandas Timestamp to seconds timestamp
                         timestamp = int(time_val.value / 10**9) if hasattr(time_val, 'value') else int(pd.Timestamp(time_val).timestamp())
                     else: # Assume it's already a numeric timestamp if not datetime
                         timestamp = int(time_val)

                     record_tuple = (
                         timestamp,
                         row['open'],
                         row['high'],
                         row['low'],
                         row['close'],
                         int(row.get('tick_volume', 0)),
                         int(row.get('spread', 0)),
                         int(row.get('real_volume', 0))
                     )
                     output_records.append(record_tuple)
                 return output_records
            return []
        return None

    # --- Test configuration methods ---
    def add_symbol_info(self, symbol_name: str, info: Dict[str, Any]):
        """Allows tests to set up symbol information."""
        self._symbol_states[symbol_name] = info

    def add_historical_data(self, symbol: str, timeframe: int, data: pd.DataFrame):
        """Allows tests to set up historical data for a symbol/timeframe."""
        if symbol not in self._historical_data:
            self._historical_data[symbol] = {}
        # Ensure DataFrame has 'time' as datetime objects, 'open', 'high', 'low', 'close', 'tick_volume'
        # For copy_rates_from_pos, 'time' should be convertible to Unix timestamp.
        # 'tick_volume' is expected, others like 'spread', 'real_volume' are optional but good to have.
        self._historical_data[symbol][timeframe] = data.copy()

    def add_open_position(self, position_data: Dict[str, Any]):
        """Adds a position to the internal list of open positions."""
        # Ensure required fields are present, assign ticket if not
        if 'ticket' not in position_data:
            position_data['ticket'] = self._next_ticket_id
            self._next_ticket_id +=1
        self._open_positions.append(position_data)

    def get_open_positions(self) -> List[Dict[str,Any]]: # Helper for tests to inspect state
        return self._open_positions

    def clear_open_positions(self):
        self._open_positions = []
        self._next_ticket_id = 1 # Reset ticket counter for positions

    def positions_get(self, symbol: Optional[str] = None, ticket: Optional[int] = None) -> Optional[List[Any]]: # Returns List[PositionInfo]
        self._log_action("positions_get called", {'symbol': symbol, 'ticket': ticket})
        if not self._initialized or not self._connected_login:
            self._set_last_error(8, "Not connected for positions_get")
            return None

        results = []
        for pos_data in self._open_positions:
            add_pos = True
            if symbol is not None and pos_data.get('symbol') != symbol:
                add_pos = False
            if ticket is not None and pos_data.get('ticket') != ticket:
                add_pos = False

            if add_pos:
                # Convert dict to MagicMock emulating PositionInfo object
                pos_obj = MagicMock()
                for k, v in pos_data.items():
                    setattr(pos_obj, k, v)
                results.append(pos_obj)

        if not results and (symbol or ticket): # If filtering criteria led to empty, but positions exist
             # This behavior might differ from real MT5; real MT5 returns empty tuple if no match.
             # If no positions exist at all, it returns None or empty tuple depending on version/situation.
             # For mock: return empty list if no match, or None if that's more representative.
             pass # results is already []

        self._set_last_error(TRADE_RETCODE_DONE, "Success")
        return results if results else []

    def order_send(self, request: Dict[str, Any]) -> Optional[Any]: # Returns OrderSendResult
        self._log_action("order_send called", request)
        if self.order_send_should_fail:
            self._set_last_error(self.order_send_fail_retcode, self.order_send_fail_message)
            result_mock = MagicMock()
            result_mock.retcode = self.order_send_fail_retcode
            result_mock.comment = self.order_send_fail_message
            result_mock.request = request # Attach original request
            result_mock.order = 0 # No order ticket generated
            return result_mock

        action = request.get('action')
        symbol = request.get('symbol')
        volume = request.get('volume')
        order_type = request.get('type') # e.g., ORDER_TYPE_BUY
        price = request.get('price', self._symbol_states.get(symbol, {}).get('ask' if order_type == ORDER_TYPE_BUY else 'bid', 0)) # Simulate market price
        sl = request.get('sl', 0.0)
        tp = request.get('tp', 0.0)
        magic = request.get('magic', 0)
        comment = request.get('comment', "")
        position_ticket_to_modify = request.get('position') # For close/modify

        retcode = TRADE_RETCODE_DONE
        message = "Order executed successfully"
        new_order_id = self._next_order_id
        self._next_order_id +=1

        if action == TRADE_ACTION_DEAL: # Opening a new position
            if not symbol or not volume or order_type is None:
                retcode = 10013 # TRADE_RETCODE_INVALID_REQUEST or similar
                message = "Invalid request for opening position"
            else:
                new_pos_ticket = self._next_ticket_id
                self._next_ticket_id += 1
                position_data = {
                    'ticket': new_pos_ticket, 'symbol': symbol, 'volume': volume,
                    'type': order_type, 'price_open': price, 'time': int(time.time()),
                    'sl': sl, 'tp': tp, 'magic': magic, 'comment': comment
                }
                self._open_positions.append(position_data)
                # Add a deal to history
                deal_data = {
                    'ticket': new_pos_ticket, 'order': new_order_id, 'symbol': symbol, 'type': order_type,
                    'entry': 0, # 0 for IN, 1 for OUT, 2 for IN/OUT
                    'volume': volume, 'price': price, 'profit': 0.0, 'commission': -0.1, 'swap': 0.0, # Example
                    'time': int(time.time())
                }
                self._trade_history_deals.append(deal_data)
                message = f"Position #{new_pos_ticket} opened"

        elif action == TRADE_ACTION_SLTP: # Modifying SL/TP
            found_pos = None
            for pos in self._open_positions:
                if pos['ticket'] == position_ticket_to_modify:
                    found_pos = pos
                    break
            if found_pos:
                if sl is not None: found_pos['sl'] = sl
                if tp is not None: found_pos['tp'] = tp
                message = f"Position #{position_ticket_to_modify} SL/TP modified"
            else:
                retcode = 10015 # TRADE_RETCODE_INVALID_POSITION or similar
                message = f"Position #{position_ticket_to_modify} not found for SL/TP modification"

        # Simplified close: Assume full close if action is DEAL and position ticket is provided
        # A real close would be a counter-order. This is very simplified.
        elif action == TRADE_ACTION_DEAL and position_ticket_to_modify: # Simplified close
            idx_to_remove = -1
            closed_pos_data = None
            for i, pos in enumerate(self._open_positions):
                if pos['ticket'] == position_ticket_to_modify:
                    idx_to_remove = i
                    closed_pos_data = pos
                    break
            if idx_to_remove != -1:
                closed_position = self._open_positions.pop(idx_to_remove)
                # Simulate profit/loss (very basic)
                profit = 0
                close_price = price # Assume current price passed in request is close price
                if closed_pos_data['type'] == ORDER_TYPE_BUY:
                    profit = (close_price - closed_pos_data['price_open']) * closed_pos_data['volume'] * self._symbol_states.get(symbol,{}).get('trade_contract_size',100000)
                else: # SELL
                    profit = (closed_pos_data['price_open'] - close_price) * closed_pos_data['volume'] * self._symbol_states.get(symbol,{}).get('trade_contract_size',100000)

                deal_data = {
                    'ticket': position_ticket_to_modify, 'order': new_order_id, 'symbol': symbol,
                    'type': ORDER_TYPE_SELL if closed_pos_data['type'] == ORDER_TYPE_BUY else ORDER_TYPE_BUY, # Counter order
                    'entry': 1, # 1 for OUT
                    'volume': closed_pos_data['volume'], 'price': price, 'profit': profit,
                    'commission': -0.1, 'swap': 0.0, # Example
                    'time': int(time.time())
                }
                self._trade_history_deals.append(deal_data)
                message = f"Position #{position_ticket_to_modify} closed"
            else:
                retcode = 10015
                message = f"Position #{position_ticket_to_modify} not found for closing"


        result_obj = MagicMock()
        result_obj.retcode = retcode
        result_obj.comment = message
        result_obj.order = new_order_id # This is the order ticket
        # The request object in MT5's OrderSendResult is the original request dict
        result_obj.request = request
        # Other fields like volume, price, bid, ask, etc., might be on result depending on retcode
        result_obj.volume = volume if retcode == TRADE_RETCODE_DONE else 0
        result_obj.price = price if retcode == TRADE_RETCODE_DONE else 0
        # If the action resulted in a position ticket (e.g. new position), it's often found via deal/history.
        # For simplicity, if a new position was opened, we can add its ticket to the result if helpful for mock.
        if action == TRADE_ACTION_DEAL and retcode == TRADE_RETCODE_DONE and not position_ticket_to_modify:
             # This part is tricky as MT5 result.order is the order ticket,
             # not necessarily the position ticket directly.
             # For mock, we can assume the last opened position's ticket if needed for tests.
             if self._open_positions:
                 result_obj.position_ticket = self._open_positions[-1]['ticket']


        self._set_last_error(retcode, message)
        return result_obj

    def history_deals_get(self, date_from: Union[int, datetime], date_to: Union[int, datetime],
                          group: Optional[str] = None, ticket: Optional[int] = None,
                          order: Optional[int] = None, position: Optional[int] = None) -> Optional[List[Any]]:
        self._log_action("history_deals_get called", {'from':date_from, 'to':date_to, 'ticket':ticket, 'order':order, 'position':position})
        if not self._initialized or not self._connected_login:
            self._set_last_error(9, "Not connected for history_deals_get")
            return None

        # Convert datetimes to timestamps if they are not already
        if isinstance(date_from, datetime): from_timestamp = int(date_from.timestamp())
        else: from_timestamp = date_from
        if isinstance(date_to, datetime): to_timestamp = int(date_to.timestamp())
        else: to_timestamp = date_to

        results = []
        for deal_data in self._trade_history_deals:
            add_deal = True
            if deal_data['time'] < from_timestamp or deal_data['time'] > to_timestamp:
                add_deal = False
            if ticket is not None and deal_data.get('ticket') != ticket: # 'ticket' here is position ticket
                add_deal = False
            if order is not None and deal_data.get('order') != order:
                add_deal = False
            if position is not None and deal_data.get('ticket') != position: # 'position' filter usually refers to position_id/ticket
                add_deal = False
            # Group filter is complex, not implemented for this mock

            if add_deal:
                deal_obj = MagicMock()
                for k,v in deal_data.items():
                    setattr(deal_obj, k, v)
                results.append(deal_obj)

        self._set_last_error(TRADE_RETCODE_DONE, "Success")
        return results if results else []

    def history_orders_get(self, ticket: Optional[int] = None, position: Optional[int] = None,
                           date_from: Optional[Union[int, datetime]] = None,
                           date_to: Optional[Union[int, datetime]] = None,
                           group: Optional[str] = None) -> Optional[List[Any]]:
        self._log_action("history_orders_get called", {'ticket': ticket, 'position': position})
        # This is a placeholder. A full implementation would store and filter order history.
        # For now, returning an empty list, assuming it's not critically blocking tests.
        # Tests can mock this method's return value on the MockMT5 instance if specific order history is needed.
        logger.warning("MockMT5.history_orders_get is a placeholder and returns empty list.")
        return []

    def symbols_get(self, group: Optional[str] = None) -> Optional[List[Any]]:
        self._log_action("symbols_get called", {'group': group})
        if not self._initialized:
            self._set_last_error(10, "Not initialized for symbols_get")
            return None

        # Return mock symbols based on what's configured in _symbol_states
        mock_symbols_list = []
        for name, data in self._symbol_states.items():
            s_obj = MagicMock()
            s_obj.name = name
            # Populate other common SymbolInfo attributes if needed by tests
            for k,v in data.items():
                setattr(s_obj, k, v)
            mock_symbols_list.append(s_obj)
        return mock_symbols_list

```
