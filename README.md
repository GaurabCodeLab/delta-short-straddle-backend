# Delta BTC Options Short Strangle Bot

Python async trading bot for Delta Exchange BTC options that:

- Detects an existing short strangle automatically
- Watches Delta BTC index price against both short strikes
- On first trigger, either exits all positions (`PnL >= 5 USD`) or converts to an iron fly via a short straddle adjustment (`PnL < 5 USD`)

## Key Features

- Modular architecture:
  - `exchange_client.py` (Delta API wrapper + retries)
  - `position_manager.py` (position discovery and live state)
  - `strategy_engine.py` (trigger + adjustment logic)
  - `risk_manager.py` (guards and notional limits)
  - `order_executor.py` (market order placement + fill confirmation)
- Uses Delta index price (`BTCUSDT`) as the trigger reference
- Handles partial fills via fill reconciliation loops
- Retries transient API failures with exponential backoff
- Detailed structured logging

## Strategy Behavior

### 1) Initial Position Discovery

The bot scans open option positions and identifies a valid short strangle where both legs are short, same expiry, and both legs are out-of-the-money relative to BTC index price.

### 2) First Adjustment Trigger

Trigger activates when index touches/crosses the short strike:

- crossing: `(prev_index - strike) * (curr_index - strike) <= 0`
- touch: `curr_index == strike`

At trigger:

- Compute total unrealized PnL from best bid/ask and entry prices
- If `PnL >= 5` USD: close all positions and stop
- If `PnL < 5` USD:
  - buy back the opposite short leg
  - keep the breached short leg in place
  - open the opposite short leg at the same strike to form a short straddle
  - buy iron fly wings to convert the short straddle into an iron fly

Resulting structure: iron fly after adjustment via a short straddle

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

## Environment Variables

- `DELTA_API_KEY`
- `DELTA_API_SECRET`
- `DELTA_BASE_URL` (default: `https://cdn-ind.testnet.deltaex.org`)
- `DELTA_SSL_VERIFY` (default: `false` for testnet environments with missing CA chain)
- `POLL_INTERVAL_SECONDS` (default: `2`)
- `MAX_SINGLE_ORDER_QTY` (default: `10`)
- `MAX_TOTAL_OPTION_NOTIONAL` (default: `100000`)
- `PROFIT_CAPTURE_RATIO` (default: `0.5` for 50% of total premium received)

Heartbeat logs run on the same interval as `POLL_INTERVAL_SECONDS`.

## Notes

- The code uses `delta-rest-client` and wraps sync calls with `asyncio.to_thread`.
- Delta REST field names can vary by account/product type. Mapping logic is centralized in `src/models.py` and `src/exchange_client.py`.
- Test only on testnet first.
