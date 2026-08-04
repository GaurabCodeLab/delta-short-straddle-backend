# Delta BTC Options Short Strangle Bot

Python async trading bot for Delta Exchange BTC options that:

- Detects an existing short strangle automatically
- Detects an active short straddle automatically
- Watches Delta BTC index price against both option strikes
- Continuously monitors combined short-straddle/strangle PnL and exits when profit or loss thresholds are reached

## Key Features

- Modular architecture:
  - `exchange_client.py` (Delta API wrapper + retries)
  - `position_manager.py` (position discovery and live state)
  - `strategy_engine.py` (trigger and dynamic adjustment logic)
  - `order_executor.py` (market order placement + fill confirmation)
- Uses Delta index price (`BTCUSDT`) for dynamic structure monitoring
- Detects short straddle ↔ short strangle transitions and closes positions on exit conditions
- Detailed structured logging

## Strategy Behavior

### 1) Initial Position Discovery

The bot scans open option positions and identifies a valid short strangle or short straddle with the same expiry.

### 2) Dynamic Adjustment Trigger

Trigger activates when index touches/crosses a relevant option strike:

- crossing: `(prev_index - strike) * (curr_index - strike) <= 0`
- touch: `curr_index == strike`

At trigger:

- compute current PnL from best bid/ask and entry prices
- if overall strategy PnL reaches the configured profit target, exit all positions
- otherwise, adjust between short straddle and short strangle as needed

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Copy env template and set credentials:

```bash
copy .env.example .env
```

3. Run bot:

```bash
python -m src.main
```

4. Run tests:

```bash
python -m pytest tests/test_strategy_workflow.py
```

## Testing

Run the focused strategy workflow test file:

```bash
python -m pytest tests/test_strategy_workflow.py
```

Run the full test suite:

```bash
python -m pytest
```

Run tests with coverage reporting for the `src/` package:

```bash
python -m pytest --cov=src --cov-report=term-missing
```

Generate both terminal and HTML coverage reports:

```bash
python -m pytest --cov=src --cov-report=term-missing --cov-report=html
```

The HTML report is written to `htmlcov/index.html`.

Run a single test case by name:

```bash
python -m pytest tests/test_strategy_workflow.py -k test_short_straddle_monitor_exits_on_profit_target
```

## Environment Variables

- `DELTA_API_KEY`
- `DELTA_API_SECRET`
- `DELTA_BASE_URL` (default: `https://cdn-ind.testnet.deltaex.org`)
- `DELTA_SSL_VERIFY` (default: `false`)
- `POLL_INTERVAL_SECONDS` (default: `0.2`)
- `STRIKE_ADJUSTMENT_THRESHOLD` (default: `1000`)
- `PROFIT_TARGET` (default: `50`)
- `STOP_LOSS` (default: `100`)
- `LOG_LEVEL` (default: `INFO`)

Heartbeat logs run on the same interval as `POLL_INTERVAL_SECONDS`.

## Notes

- The code uses `delta-rest-client` and wraps sync methods with `asyncio.to_thread` for compatibility.
- Delta REST field names can vary by account/product type. Mapping logic is centralized in `src/models.py` and `src/exchange_client.py`.
- Test first on a non-production Delta environment.
