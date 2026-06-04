# Delta BTC Options Ratio-Spread Bot

Python async trading bot for Delta Exchange BTC options that:

- Detects an existing `1x long + 2x short` ratio spread automatically
- Watches Delta BTC index price against short strike
- On first trigger, either exits all positions (`PnL >= 5 USD`) or converts to a short straddle (`PnL < 5 USD`)

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

The bot scans open option positions and identifies a valid ratio spread where all legs have:

- same strike
- same expiry
- same option type (`call` or `put`)
- net structure: `+1` long, `-2` short (in equivalent quantity units)

### 2) First Adjustment Trigger

Trigger activates when index touches/crosses the short strike:

- crossing: `(prev_index - strike) * (curr_index - strike) <= 0`
- touch: `curr_index == strike`

At trigger:

- Compute total unrealized PnL from best bid/ask and entry prices
- If `PnL >= 5` USD: close all positions and stop
- If `PnL < 5` USD:
  - close long leg
  - close one short leg (half of the initial short side)
  - open new short leg of opposite option type at same strike/expiry

Resulting structure: short straddle (`1x short call + 1x short put`)

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

Heartbeat logs run on the same interval as `POLL_INTERVAL_SECONDS`.

## Notes

- The code uses `delta-rest-client` and wraps sync calls with `asyncio.to_thread`.
- Delta REST field names can vary by account/product type. Mapping logic is centralized in `src/models.py` and `src/exchange_client.py`.
- Test only on testnet first.
