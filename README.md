# Director - Proyecto de Trading Algorítmico

Este proyecto contiene un conjunto de herramientas para trading algorítmico integrado con MetaTrader 5.

## Estructura del Proyecto

El proyecto contiene los siguientes módulos principales:

- **mt5_connector.py**: Gestiona la conexión con MetaTrader 5.
- **config_manager.py**: Maneja la configuración del sistema.
- **trading_operations.py**: Implementa operaciones de trading.
- **indicator_calculator.py**: Calcula indicadores técnicos.
- **strategy_engine.py**: Motor para ejecutar estrategias de trading.
- **sr_handler.py**: Manejador de soportes y resistencias.
- **risk_manager.py**: Gestiona el riesgo en operaciones.
- **logging_setup.py**: Configuración del sistema de logging.

## Requisitos

- Python 3.10.11
- MetaTrader 5

## Features

- **Modular Architecture**: Easily extendable with custom strategies and indicators
- **Multiple Timeframe Analysis**: Analyze and trade across different timeframes
- **Risk Management**: Built-in position sizing and risk management
- **Backtesting**: Test strategies on historical data before going live
- **Support for Multiple Symbols**: Trade multiple currency pairs simultaneously
- **Technical Indicators**: Comprehensive library of technical indicators
- **Support/Resistance Detection**: Identify key price levels
- **Logging**: Comprehensive logging for monitoring and debugging

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/director-trading-bot.git
   cd director-trading-bot
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure the bot by editing `config.json` (see Configuration section below)

## Configuration

Edit the `config.json` file to configure the bot. Here's an example configuration:

```json
{
    "meta_trader": {
        "path": "C:/Program Files/MetaTrader 5/terminal64.exe",
        "server": "YourBrokerServer",
        "login": 12345678,
        "password": "yourpassword"
    },
    "trading": {
        "symbols": ["EURUSD", "GBPUSD"],
        "timeframes": ["M15", "H1", "H4"],
        "risk_per_trade": 1.0,
        "max_daily_drawdown": 5.0,
        "max_open_trades": 5
    },
    "strategy": {
        "name": "confluencia",
        "params": {
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9
        }
    },
    "logging": {
        "level": "INFO",
        "file": "logs/trading_bot.log",
        "max_size_mb": 10,
        "backup_count": 5
    }
}
```

## Usage

### Running the Bot

To start the trading bot:

```bash
python main_bot.py
```

### Backtesting

To run a backtest:

```python
from backtester import Backtester
import pandas as pd

# Load historical data
data = pd.read_csv('path/to/your/data.csv', parse_dates=['Date'], index_col='Date')

# Initialize and run backtest
backtester = Backtester()
results = backtester.run_backtest(
    data=data,
    symbol='EURUSD',
    start_date='2022-01-01',
    end_date='2023-01-01',
    initial_cash=10000.0,
    commission=0.0005
)

# Print summary
print(backtester.get_summary())

# Plot equity curve
backtester.plot_equity_curve('equity_curve.png')
```

## Project Structure

```
director-trading-bot/
├── config.json           # Main configuration file
├── main_bot.py          # Main bot script
├── backtester.py        # Backtesting engine
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── core/                # Core functionality
│   ├── __init__.py
│   ├── config_manager.py  # Configuration management
│   ├── logging_setup.py   # Logging configuration
│   ├── mt5_connector.py   # MetaTrader 5 interface
│   ├── indicator_calculator.py  # Technical indicators
│   ├── sr_handler.py      # Support/resistance detection
│   └── strategy_engine.py # Strategy implementation
└── strategies/          # Trading strategies
    ├── __init__.py
    └── confluencia_strategy.py  # Example strategy
```

## Strategy Development

To create a new strategy:

1. Create a new file in the `strategies` directory (e.g., `my_strategy.py`)
2. Implement your strategy by extending the `BaseStrategy` class
3. Update the configuration to use your new strategy

Example strategy template:

```python
from typing import Dict, Any, Optional
import pandas as pd
from .base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    """My custom trading strategy."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "my_strategy"
    
    def analyze(
        self,
        symbol: str,
        data: Dict[str, pd.DataFrame],
        position_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze the market and generate trading signals.
        
        Args:
            symbol: Trading symbol
            data: Dictionary of DataFrames for different timeframes
            position_info: Information about current position (if any)
            
        Returns:
            Dictionary with analysis results and trading signals
        """
        # Your strategy logic here
        return {
            'signal': 'NONE',  # 'BUY', 'SELL', or 'NONE'
            'stop_loss': None,
            'take_profit': None,
            'message': 'No signal',
            'indicators': {}
        }
```

## Risk Management

The bot includes several risk management features:

- Position sizing based on account balance and risk per trade
- Maximum daily drawdown protection
- Maximum number of open trades
- Stop loss and take profit management
- Trailing stops

## Logging

The bot logs to both console and file by default. Log files are rotated when they reach 10MB, keeping up to 5 backup files.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is for educational purposes only. Use it at your own risk. The authors are not responsible for any financial losses incurred while using this software. Always test thoroughly with a demo account before trading with real money.
