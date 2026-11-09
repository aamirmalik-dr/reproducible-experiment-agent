"""The trusted split service: row partitions, integrity checks, visible summary.

Partitions are assigned once, when a task is generated, by
``assign_partitions``. Rows are grouped into units (the group column for a
grouped task, otherwise identical-content rows) and whole units go to one
partition, so neither a group nor a duplicated row ever straddles train, dev
and holdout. Holdout rows are written only to evaluator/; this module only
ever loads the train and dev rows.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schemas import TaskManifest

ROW_ID = "__row_id"
PARTITION = "__partition"
FRACTIONS = (("train", 0.6), ("dev", 0.2), ("holdout", 0.2))


def content_key(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Hash of each row's content over ``columns`` (missing values included)."""
    sub = df[columns].astype(object)
    text = sub.where(sub.notna(), "<NA>").map(str).agg("|".join, axis=1)
    return text.map(lambda s: hashlib.sha256(s.encode()).hexdigest()[:16])


def assign_partitions(
    df: pd.DataFrame, content_columns: list[str], group_column: str | None, seed: int
) -> tuple[pd.Series, int]:
    """Assign every row to train, dev or holdout by whole units.

    Args:
        df: Full task table.
        content_columns: Columns that define duplicate rows.
        group_column: Group column for grouped tasks, else None.
        seed: Shuffle seed for the unit order.

    Returns:
        The partition label per row and the number of rows that were merged
        into a unit because they duplicate another row.
    """
    keys = content_key(df, content_columns)
    dup_rows = int(keys.duplicated(keep="first").sum())
    units = df[group_column].astype(str) if group_column else keys
    order = pd.Series(units.unique())
    order = order.sample(frac=1.0, random_state=seed).tolist()
    sizes = units.value_counts()
    total, cum, label_of = len(df), 0, {}
    bounds = np.cumsum([f for _, f in FRACTIONS]) * total
    for unit in order:
        idx = int(np.searchsorted(bounds, cum, side="right"))
        label_of[unit] = FRACTIONS[min(idx, len(FRACTIONS) - 1)][0]
        cum += int(sizes[unit])
    labels = units.map(label_of)
    straddle = pd.DataFrame({"k": keys, "p": labels}).groupby("k")["p"].nunique()
    if (straddle > 1).any():
        raise AssertionError("a duplicated row straddles partitions")
    return labels, dup_rows


def csv_bytes(df: pd.DataFrame) -> bytes:
    """Canonical CSV serialisation used for every committed task table."""
    return df.to_csv(index=False, lineterminator="\n").encode("utf-8")


@dataclass
class Partitions:
    """Train and dev rows of one task. Holdout rows are not here by design."""

    manifest: TaskManifest
    train: pd.DataFrame
    dev: pd.DataFrame

    @property
    def sha256(self) -> str:
        """Hash binding a trial to the exact row-id lists it used."""
        ids = ",".join(map(str, self.train[ROW_ID])) + "|" + ",".join(map(str, self.dev[ROW_ID]))
        return hashlib.sha256(ids.encode()).hexdigest()

    def summary(self) -> dict[str, Any]:
        """The visible split summary: counts, strategy and hashes, never holdout rows."""
        m = self.manifest
        out: dict[str, Any] = {
            "strategy": m.partition_strategy,
            "train_rows": len(self.train),
            "dev_rows": len(self.dev),
            "holdout_rows": m.rows["holdout"],
            "holdout_access": "none; scored once, outside the agent loop, after freeze",
            "duplicate_rows_kept_on_one_side": m.duplicate_rows_merged,
            "partition_sha256": self.sha256[:16],
            "train_dev_sha256": m.train_dev_sha256[:16],
        }
        if m.group_column:
            out["group_column"] = m.group_column
            out["train_groups"] = int(self.train[m.group_column].nunique())
            out["dev_groups"] = int(self.dev[m.group_column].nunique())
            out["groups_shared_between_train_and_dev"] = len(
                set(self.train[m.group_column]) & set(self.dev[m.group_column])
            )
        return out


def load_partitions(task_dir: Path, manifest: TaskManifest) -> Partitions:
    """Load train and dev rows and verify them against the manifest hash."""
    raw = (task_dir / "train_dev.csv").read_bytes()
    df = pd.read_csv(io.BytesIO(raw), keep_default_na=True)
    if hashlib.sha256(raw).hexdigest() != manifest.train_dev_sha256:
        raise ValueError(f"{manifest.task_id}: train_dev.csv does not match its manifest hash")
    train = df[df[PARTITION] == "train"].drop(columns=[PARTITION]).reset_index(drop=True)
    dev = df[df[PARTITION] == "dev"].drop(columns=[PARTITION]).reset_index(drop=True)
    return Partitions(manifest, train, dev)
