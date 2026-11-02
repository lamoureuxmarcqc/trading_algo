"""
Memory-efficient Batch trainer using StockModelTrain streaming API.
Places: trading_algo/batch/trainer.py
"""
from __future__ import annotations

import os
import time
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
import glob

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

from trading_algo.models.stockmodeltrain import StockModelTrain
from trading_algo.models.base_model import ImprovedLSTMPredictorMultiOutput

logger = logging.getLogger(__name__)


class BatchTrainer:
    def __init__(
        self,
        symbols: List[str],
        period: str = "20y",
        lookback: int = 60,
        epochs: int = 50,
        batch_size: int = 64,
        output_dir: str = "models_saved/batch",
        max_symbols: int = 500,
        min_rows: int = 300,
        val_ratio: float = 0.1,          # fraction des séquences pour validation
        clip_quantile: float = 0.01,      # clipping des outliers
        max_retries_per_symbol: int = 2,  # réessaie en cas d'échec fetch
    ):
        self.symbols = [s.upper() for s in symbols][:max_symbols]
        self.period = period
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.min_rows = min_rows
        self.val_ratio = val_ratio
        self.clip_quantile = clip_quantile
        self.max_retries_per_symbol = max_retries_per_symbol

        os.makedirs(self.output_dir, exist_ok=True)
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        # populated during run()
        self.common_feature_cols: List[str] = []
        self.common_target_cols: List[str] = []
        self.scaled_map: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}  # (X_scaled, y_scaled)

    # ----------------------------------------------------------------------
    # 1. Data collection with retry
    # ----------------------------------------------------------------------
    def _fetch_symbol_data(self, symbol: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Fetch features/targets for a single symbol with retries."""
        for attempt in range(1, self.max_retries_per_symbol + 1):
            try:
                sst = StockModelTrain(symbol, period=self.period)
                ok = sst.fetch_data()
                if not ok:
                    raise RuntimeError(f"fetch_data returned False")
                feats = getattr(sst, "features", None)
                targs = getattr(sst, "targets", None)
                if feats is None or feats.empty or targs is None or targs.empty:
                    raise RuntimeError("Empty features or targets")
                # Align indices
                idx = feats.index.intersection(targs.index)
                if len(idx) < self.min_rows:
                    raise RuntimeError(f"Only {len(idx)} aligned rows (<{self.min_rows})")
                return feats.loc[idx].copy(), targs.loc[idx].copy()
            except Exception as e:
                logger.warning(f"Attempt {attempt}/{self.max_retries_per_symbol} failed for {symbol}: {e}")
                if attempt == self.max_retries_per_symbol:
                    logger.error(f"Skipping {symbol} after {attempt} attempts")
                    return None
                time.sleep(2 ** attempt)  # simple exponential backoff
        return None

    def collect_symbol_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        For each symbol: fetch data using StockModelTrain.
        Return mapping symbol -> {'features': df, 'targets': df}
        """
        collected = {}
        for i, sym in enumerate(self.symbols, 1):
            logger.info(f"[{i}/{len(self.symbols)}] Fetching {sym} ({self.period})")
            result = self._fetch_symbol_data(sym)
            if result is None:
                continue
            feats, targs = result
            collected[sym] = {"features": feats, "targets": targs}
            logger.info(f"Collected {sym}: {len(feats)} rows, features={feats.shape[1]}, targets={targs.shape[1]}")
        return collected

    # ----------------------------------------------------------------------
    # 2. Common columns
    # ----------------------------------------------------------------------
    def determine_common_columns(self, collected: Dict[str, Dict[str, pd.DataFrame]]):
        """Compute intersection of feature and target columns across symbols."""
        common_feat = None
        common_targ = None
        for v in collected.values():
            fcols = set(v["features"].columns)
            tcols = set(v["targets"].columns)
            common_feat = fcols if common_feat is None else (common_feat & fcols)
            common_targ = tcols if common_targ is None else (common_targ & tcols)

        if not common_feat or not common_targ:
            raise RuntimeError("No common feature/target columns across symbols")

        self.common_feature_cols = sorted(list(common_feat))
        self.common_target_cols = sorted(list(common_targ))
        logger.info(f"Common features: {len(self.common_feature_cols)} cols")
        logger.info(f"Common targets : {len(self.common_target_cols)} cols")

    # ----------------------------------------------------------------------
    # 3. Global scalers with clipping
    # ----------------------------------------------------------------------
    def fit_global_scalers(self, collected: Dict[str, Dict[str, pd.DataFrame]]):
        """Fit scalers on concatenated per-symbol data after clipping outliers."""
        all_X = pd.concat([v["features"][self.common_feature_cols] for v in collected.values()], axis=0)
        all_y = pd.concat([v["targets"][self.common_target_cols] for v in collected.values()], axis=0)

        # sanitize infinities
        all_X = all_X.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
        all_y = all_y.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

        # clip extreme outliers
        lower = all_X.quantile(self.clip_quantile)
        upper = all_X.quantile(1 - self.clip_quantile)
        all_X = all_X.clip(lower, upper, axis=1)

        lower_y = all_y.quantile(self.clip_quantile)
        upper_y = all_y.quantile(1 - self.clip_quantile)
        all_y = all_y.clip(lower_y, upper_y, axis=1)

        # fit scalers
        self.feature_scaler.fit(all_X.values)
        self.target_scaler.fit(all_y.values)

        # save scalers immediately (for recovery)
        scaler_feat_path = os.path.join(self.output_dir, "feature_scaler.pkl")
        scaler_targ_path = os.path.join(self.output_dir, "target_scaler.pkl")
        with open(scaler_feat_path, "wb") as f:
            pickle.dump(self.feature_scaler, f)
        with open(scaler_targ_path, "wb") as f:
            pickle.dump(self.target_scaler, f)
        logger.info(f"Fitted and saved global scalers (clip={self.clip_quantile})")

    # ----------------------------------------------------------------------
    # 4. Precompute scaled arrays (memory efficient)
    # ----------------------------------------------------------------------
    def prepare_scaled_map(self, collected: Dict[str, Dict[str, pd.DataFrame]]):
        """
        Transform each symbol's data to scaled numpy arrays.
        Releases original DataFrames to free memory.
        """
        scaled_map = {}
        for sym, data in collected.items():
            X_df = data["features"][self.common_feature_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
            y_df = data["targets"][self.common_target_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

            if len(X_df) <= self.lookback:
                logger.warning(f"Skipping {sym}: only {len(X_df)} rows, need >{self.lookback}")
                continue

            X_scaled = self.feature_scaler.transform(X_df.values).astype(np.float32)
            y_scaled = self.target_scaler.transform(y_df.values).astype(np.float32)
            scaled_map[sym] = (X_scaled, y_scaled)

            # free original DataFrame memory
            del data
        self.scaled_map = scaled_map
        logger.info(f"Prepared scaled data for {len(scaled_map)} symbols")

    # ----------------------------------------------------------------------
    # 5. Train/validation split indices
    # ----------------------------------------------------------------------
    @staticmethod
    def _split_sequences(n_sequences: int, val_ratio: float) -> Tuple[int, int, int, int]:
        """
        Returns (train_start, train_end, val_start, val_end) indices.
        Validation takes the last val_ratio fraction of sequences.
        """
        val_len = max(1, int(n_sequences * val_ratio))
        train_len = n_sequences - val_len
        return (0, train_len, train_len, n_sequences)

    def _make_generator(self, scaled_map: Dict[str, Tuple[np.ndarray, np.ndarray]],
                        split_range: Tuple[int, int], batch_size: int, lookback: int,
                        shuffle: bool = False) -> Callable[[], Any]:
        """
        Returns a generator factory that yields batches from sequences whose
        start indices fall into [split_start, split_end) (exclusive end).
        split_range = (start_idx, end_idx) where indices are positions in the
        sequence list (0-based, each sequence = window ending at t).
        """
        # Pre-build list of (symbol, seq_start_idx) for all valid sequences in range
        seq_items = []
        for sym, (X, y) in scaled_map.items():
            n = len(X)
            max_start = n - lookback  # last index that can start a sequence
            # sequence start indices are lookback .. n-1  (window ending at t)
            # We map t (the target index) -> sequence start = t - lookback
            # But easier: iterate over t from lookback to n-1, then check if t is in split_range?
            # split_range is defined on the sequence index (t - lookback) or on t?
            # Let's define split_range on the *target index* t (the end of window).
            # Then sequence is X[t-lookback:t], y[t].
            # So for each t in [lookback, n-1], if t in split_range (after mapping), include.
            # However split_range is given as (start_idx, end_idx) in terms of target indices.
            # We'll implement directly in the generator loop.
            # To avoid recomputing each time, we precompute the list of t indices.
            # But to keep generator lightweight, we'll just iterate over t and filter.
            # We'll store the list of t indices for this symbol that fall in range.
            t_start, t_end = split_range
            for t in range(lookback, n):
                if t_start <= t < t_end:
                    seq_items.append((sym, t))
        if shuffle:
            np.random.shuffle(seq_items)

        def generator():
            # infinite loop for Keras
            while True:
                for i in range(0, len(seq_items), batch_size):
                    batch_X = []
                    batch_y = []
                    for sym, t in seq_items[i:i+batch_size]:
                        X_s, y_s = scaled_map[sym]
                        batch_X.append(X_s[t - lookback:t])
                        batch_y.append(y_s[t])
                    if batch_X:
                        yield np.array(batch_X, dtype=np.float32), np.array(batch_y, dtype=np.float32)
        return generator

    def create_train_val_generators(self) -> Tuple[Callable, Callable, int, int]:
        """
        Create train and validation generator factories.
        Returns (train_gen_factory, val_gen_factory, train_steps, val_steps)
        where steps are computed as ceil(total_train_sequences / batch_size).
        """
        # First compute total sequences per split to set steps_per_epoch
        total_train_seqs = 0
        total_val_seqs = 0
        seq_ranges = {}  # per symbol: (train_start, train_end, val_start, val_end) in target index terms

        for sym, (X, y) in self.scaled_map.items():
            n = len(X)
            if n <= self.lookback:
                continue
            n_seqs = n - self.lookback  # number of possible target indices t from lookback to n-1
            train_start, train_end, val_start, val_end = self._split_sequences(n_seqs, self.val_ratio)
            # Convert to target index space:
            # target index t ranges from lookback to n-1, mapping: seq_index = t - lookback
            # So t = lookback + seq_index
            train_t_start = self.lookback + train_start
            train_t_end   = self.lookback + train_end
            val_t_start   = self.lookback + val_start
            val_t_end     = self.lookback + val_end
            total_train_seqs += (train_t_end - train_t_start)
            total_val_seqs   += (val_t_end - val_t_start)
            seq_ranges[sym] = (train_t_start, train_t_end, val_t_start, val_t_end)

        if total_train_seqs == 0 or total_val_seqs == 0:
            raise RuntimeError(f"No sequences for training ({total_train_seqs}) or validation ({total_val_seqs})")

        train_steps = max(1, total_train_seqs // self.batch_size)
        val_steps   = max(1, total_val_seqs // self.batch_size)

        logger.info(f"Total train sequences: {total_train_seqs} -> steps/epoch: {train_steps}")
        logger.info(f"Total val sequences  : {total_val_seqs} -> validation steps: {val_steps}")

        # Build generators for train and val using the same scaled_map but different t ranges
        # We need to create a generator that yields only sequences with t in the given range.
        # Instead of building a list per generator, we'll create a closure that captures the range.
        def make_train_gen():
            # Build item list for train range
            items = []
            for sym, (X, y) in self.scaled_map.items():
                n = len(X)
                if n <= self.lookback:
                    continue
                _, _, val_s, val_e = seq_ranges.get(sym, (0,0,0,0))
                # train range is everything before val_s? Actually train range is from lookback to val_s-1
                # but we already stored train_t_start, train_t_end in seq_ranges? Let's store them.
                # Simpler: recompute per symbol
                n_seqs = n - self.lookback
                train_start, train_end, _, _ = self._split_sequences(n_seqs, self.val_ratio)
                t_start = self.lookback + train_start
                t_end   = self.lookback + train_end
                for t in range(t_start, t_end):
                    items.append((sym, t))
            # shuffle for better training
            np.random.shuffle(items)
            while True:
                for i in range(0, len(items), self.batch_size):
                    batch_X, batch_y = [], []
                    for sym, t in items[i:i+self.batch_size]:
                        X_s, y_s = self.scaled_map[sym]
                        batch_X.append(X_s[t - self.lookback:t])
                        batch_y.append(y_s[t])
                    if batch_X:
                        yield np.array(batch_X, dtype=np.float32), np.array(batch_y, dtype=np.float32)

        def make_val_gen():
            items = []
            for sym, (X, y) in self.scaled_map.items():
                n = len(X)
                if n <= self.lookback:
                    continue
                n_seqs = n - self.lookback
                _, _, val_start, val_end = self._split_sequences(n_seqs, self.val_ratio)
                t_start = self.lookback + val_start
                t_end   = self.lookback + val_end
                for t in range(t_start, t_end):
                    items.append((sym, t))
            # No shuffle for validation
            while True:
                for i in range(0, len(items), self.batch_size):
                    batch_X, batch_y = [], []
                    for sym, t in items[i:i+self.batch_size]:
                        X_s, y_s = self.scaled_map[sym]
                        batch_X.append(X_s[t - self.lookback:t])
                        batch_y.append(y_s[t])
                    if batch_X:
                        yield np.array(batch_X, dtype=np.float32), np.array(batch_y, dtype=np.float32)

        return make_train_gen, make_val_gen, train_steps, val_steps

    # ----------------------------------------------------------------------
    # 6. Main run
    # ----------------------------------------------------------------------
    def run(self):
        """Main entry: collect data, fit scalers, precompute, then stream-train."""
        # Step 1: collect raw data
        collected = self.collect_symbol_data()
        if not collected:
            logger.error("No data collected; aborting")
            return

        # Step 2: determine common columns
        self.determine_common_columns(collected)

        # Step 3: fit global scalers (with clipping) and save them
        self.fit_global_scalers(collected)

        # Step 4: precompute scaled arrays (releases original data)
        self.prepare_scaled_map(collected)
        if not self.scaled_map:
            logger.error("No symbols with enough rows after scaling prep")
            return

        # Step 5: create train/val generators with proper split
        train_gen_factory, val_gen_factory, train_steps, val_steps = self.create_train_val_generators()

        # Step 6: train using StockModelTrain's streaming logic (supports checkpoint resume)
        smt = StockModelTrain("BATCH", period=self.period)
        smt.feature_scaler = self.feature_scaler
        smt.target_scaler = self.target_scaler
        smt.feature_columns = self.common_feature_cols
        smt.target_columns = self.common_target_cols
        smt.sequence_length = self.lookback
        smt.lookback_days = self.lookback
        smt.batch_size = self.batch_size

        logger.info(f"Starting streaming training: epochs={self.epochs}, train_steps={train_steps}, val_steps={val_steps}")

        history = smt.train_with_generator(
            data_gen=train_gen_factory,
            steps_per_epoch=train_steps,
            validation_gen=val_gen_factory,
            validation_steps=val_steps,
            epochs=self.epochs,
            batch_size=self.batch_size,
            model_dir=self.output_dir,
            additional_callbacks=None,
            use_checkpoint=True,   # enables auto-resume on crash
        )

        # Step 7: save metadata and training history
        ts = time.strftime("%Y%m%d_%H%M%S")
        metadata = {
            "symbols_count": len(self.scaled_map),
            "symbols": list(self.scaled_map.keys()),
            "common_feature_cols": self.common_feature_cols,
            "common_target_cols": self.common_target_cols,
            "lookback": self.lookback,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "val_ratio": self.val_ratio,
            "clip_quantile": self.clip_quantile,
            "training_steps_per_epoch": train_steps,
            "validation_steps": val_steps,
            "model_dir": self.output_dir,
            "timestamp": ts,
        }
        meta_path = os.path.join(self.output_dir, f"batch_metadata_{ts}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Save training history if available
        if history is not None:
            hist_path = os.path.join(self.output_dir, f"training_history_{ts}.json")
            # Convert numpy values to Python types
            hist_dict = {k: [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in vals]
                         for k, vals in history.history.items()}
            with open(hist_path, "w", encoding="utf-8") as f:
                json.dump(hist_dict, f, indent=2)

        logger.info("Batch training finished.")
        logger.info(f"Metadata saved: {meta_path}")
        if history is not None:
            logger.info(f"Training history saved: {hist_path}")

        return history