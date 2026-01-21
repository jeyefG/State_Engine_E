"""MetaTrader 5 connectivity for OHLCV data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import MetaTrader5 as mt5
import pandas as pd


_TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M2": mt5.TIMEFRAME_M2,
    "M3": mt5.TIMEFRAME_M3,
    "M4": mt5.TIMEFRAME_M4,
    "M5": mt5.TIMEFRAME_M5,
    "M6": mt5.TIMEFRAME_M6,
    "M10": mt5.TIMEFRAME_M10,
    "M12": mt5.TIMEFRAME_M12,
    "M15": mt5.TIMEFRAME_M15,
    "M20": mt5.TIMEFRAME_M20,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H2": mt5.TIMEFRAME_H2,
    "H3": mt5.TIMEFRAME_H3,
    "H4": mt5.TIMEFRAME_H4,
    "H6": mt5.TIMEFRAME_H6,
    "H8": mt5.TIMEFRAME_H8,
    "H12": mt5.TIMEFRAME_H12,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}


def _normalize_timeframe(timeframe: str) -> str:
    return str(timeframe).upper()


def _resolve_timeframe(timeframe: str | int) -> tuple[int, str]:
    if isinstance(timeframe, int):
        return timeframe, str(timeframe)
    key = _normalize_timeframe(timeframe)
    if key not in _TIMEFRAME_MAP:
        raise ValueError(f"Unsupported MT5 timeframe: {timeframe}")
    return _TIMEFRAME_MAP[key], key


@dataclass
class MT5Connector:
    """Simple MT5 connector for OHLCV retrieval."""

    def __post_init__(self) -> None:
        if not mt5.initialize():
            raise RuntimeError(f"No se pudo conectar a MetaTrader 5: {mt5.last_error()}")

    def shutdown(self) -> None:
        mt5.shutdown()

    def obtener_ohlcv(
        self,
        symbol: str,
        timeframe: str | int,
        fecha_inicio: datetime,
        fecha_fin: datetime,
    ) -> pd.DataFrame:
        """Obtener velas en el timeframe dado."""
        timeframe_value, timeframe_label = _resolve_timeframe(timeframe)
        rates = mt5.copy_rates_range(symbol, timeframe_value, fecha_inicio, fecha_fin)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No se pudieron obtener datos {timeframe_label} desde MT5.")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        return df

    def obtener_h1(
        self,
        symbol: str,
        fecha_inicio: datetime,
        fecha_fin: datetime,
    ) -> pd.DataFrame:
        """Obtener velas H1 en el rango dado."""
        return self.obtener_ohlcv(symbol, "H1", fecha_inicio, fecha_fin)

    def obtener_m5(
        self,
        symbol: str,
        fecha_inicio: datetime,
        fecha_fin: datetime,
    ) -> pd.DataFrame:
        """Obtener velas M5 en el rango dado."""
        return self.obtener_ohlcv(symbol, "M5", fecha_inicio, fecha_fin)
    
    def server_now(self, symbol: str) -> pd.Timestamp:
        """
        Return current MT5 server time inferred from last tick (naive timestamp).
        This is the closest we can get to server clock using the Python MT5 API.
        """
        # Ensure symbol is selected so ticks update
        mt5.symbol_select(symbol, True)
    
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError("No se pudo obtener tick para inferir hora del servidor MT5.")
    
        # tick.time is seconds since epoch (server-side timestamp)
        return pd.to_datetime(int(tick.time), unit="s")

__all__ = ["MT5Connector"]
