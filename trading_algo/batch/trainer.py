"""
Memory-efficient Batch trainer using StockModelTrain streaming API.
Places: trading_algo/batch/trainer.py
"""
from __future__ import annotations

import os
import time
import json
import logging
from typing import List, Dict, Any, Optional

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
    ):
        self.symbols = [s.upper() for s in symbols][:max_symbols]
        self.period = period
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.min_rows = min_rows

        os.makedirs(self.output_dir, exist_ok=True)
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        # populated during run()
        self.common_feature_cols: List[str] = []
        self.common_target_cols: List[str] = []

    def collect_symbol_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        For each symbol: use StockModelTrain.fetch_data() to populate .features and .targets.
        Return mapping symbol -> {'features': df, 'targets': df}
        """
        collected: Dict[str, Dict[str, pd.DataFrame]] = {}
        for i, sym in enumerate(self.symbols, 1):
            logger.info(f"[{i}/{len(self.symbols)}] Fetching {sym} ({self.period})")
            try:
                sst = StockModelTrain(sym, period=self.period)
                ok = sst.fetch_data()
                if not ok:
                    logger.warning(f"Skipping {sym}: fetch_data failed")
                    continue

                feats = getattr(sst, "features", None)
                targs = getattr(sst, "targets", None)
                if feats is None or feats.empty or targs is None or targs.empty:
                    logger.warning(f"Skipping {sym}: no features/targets")
                    continue

                # Ensure at least min_rows after alignment
                idx = feats.index.intersection(targs.index)
                if len(idx) < self.min_rows:
                    logger.warning(f"Skipping {sym}: insufficient aligned rows ({len(idx)})")
                    continue

                collected[sym] = {"features": feats.loc[idx].copy(), "targets": targs.loc[idx].copy()}
                logger.info(f"Collected {sym}: {len(idx)} rows, features={feats.shape[1]}, targets={targs.shape[1]}")

            except Exception as e:
                logger.exception(f"Error collecting {sym}: {e}")
                continue

        return collected

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
        logger.info(f"Common features: {len(self.common_feature_cols)}, targets: {len(self.common_target_cols)}")

    def fit_global_scalers(self, collected: Dict[str, Dict[str, pd.DataFrame]]):
        """Fit scalers on concatenated per-symbol data (common columns only)."""
        all_X = pd.concat([v["features"][self.common_feature_cols] for v in collected.values()], axis=0)
        all_y = pd.concat([v["targets"][self.common_target_cols] for v in collected.values()], axis=0)

        # sanitize
        all_X = all_X.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
        all_y = all_y.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

        # Fit scalers
        self.feature_scaler.fit(all_X.values)
        self.target_scaler.fit(all_y.values)
        logger.info("Fitted global scalers")

    def compute_total_sequences(self, collected: Dict[str, Dict[str, pd.DataFrame]]) -> int:
        """Compute total sliding-window sequences across collected symbols (no allocation)."""
        total = 0
        for sym, data in collected.items():
            n = len(data["features"])
            seqs = max(0, n - self.lookback)
            total += seqs
        return total

    def run(self):
        """Main entry: collect data, fit scalers, then stream-train a unified model."""
        collected = self.collect_symbol_data()
        if not collected:
            logger.error("No data collected; aborting")
            return

        # determine common columns and fit scalers
        self.determine_common_columns(collected)
        self.fit_global_scalers(collected)

        total_sequences = self.compute_total_sequences(collected)
        if total_sequences <= 0:
            logger.error("No sequences available for training")
            return

        logger.info(f"Total available sequences (approx): {total_sequences}")

        # set steps per epoch to cover the full dataset once per epoch
        steps_per_epoch = max(1, total_sequences // self.batch_size)
        val_fraction = 0.1
        validation_steps = max(1, int(steps_per_epoch * val_fraction))

        # Build generator factory for Keras (callable that returns a new generator)
        def gen_factory():
            return StockModelTrain.combined_data_generator(
                collected=collected,
                common_feature_cols=self.common_feature_cols,
                common_target_cols=self.common_target_cols,
                feature_scaler=self.feature_scaler,
                target_scaler=self.target_scaler,
                lookback_days=self.lookback,
                batch_size=self.batch_size,
            )

        # Use StockModelTrain centralized training logic (streaming)
        smt = StockModelTrain("BATCH", period=self.period)
        # inject scalers/metadata so it can save them after training
        smt.feature_scaler = self.feature_scaler
        smt.target_scaler = self.target_scaler
        smt.feature_columns = self.common_feature_cols
        smt.target_columns = self.common_target_cols
        smt.sequence_length = self.lookback
        smt.lookback_days = self.lookback
        smt.batch_size = self.batch_size

        logger.info(f"Starting streaming training: epochs={self.epochs}, steps_per_epoch={steps_per_epoch}, batch_size={self.batch_size}")

        history = smt.train_with_generator(
            data_gen=gen_factory,
            steps_per_epoch=steps_per_epoch,
            validation_gen=gen_factory,
            validation_steps=validation_steps,
            epochs=self.epochs,
            batch_size=self.batch_size,
            model_dir=self.output_dir,
            additional_callbacks=None,
        )

        # Save metadata
        ts = time.strftime("%Y%m%d_%H%M%S")
        metadata = {
            "symbols_count": len(collected),
            "symbols": list(collected.keys()),
            "common_feature_cols": self.common_feature_cols,
            "common_target_cols": self.common_target_cols,
            "lookback": self.lookback,
            "epochs": self.epochs,
            "training_shape_estimated_sequences": int(total_sequences),
            "model_dir": self.output_dir,
            "timestamp": ts,
        }
        meta_path = os.path.join(self.output_dir, f"batch_metadata_{ts}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info("Batch training finished.")
        logger.info(f"Metadata saved: {meta_path}")
