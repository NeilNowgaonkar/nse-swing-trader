# backup_trades.py
# Protects your trades table against GitHub Actions cache eviction.
#
# Two modes:
#   python backup_trades.py --backup   → reads trades table → writes data/trades_backup.json
#   python backup_trades.py --restore  → reads trades_backup.json → inserts missing rows back
#
# This file (trades_backup.json) IS committed to the repo.
# It only contains your trade entries — NOT the large price history.
# It is safe to commit. It contains: entry price, qty, SL, target, P&L.
# It does NOT contain any secrets or credentials.
#
# Why this exists:
#   GitHub Actions cache is evicted after 7 days of no runs (weekends, holidays).
#   Without this backup, a cache miss = losing your entire trade history.
#   With this, worst case = you lose the trades entered since the last run.

import argparse
import json
import os
import sqlite3
from datetime import datetime
from loguru import logger

from config import DB_PATH, DATA_DIR
from create_db import get_connection, create_tables

BACKUP_PATH = os.path.join(DATA_DIR, "trades_backup.json")


def backup(conn: sqlite3.Connection) -> int:
    """
    Dump the entire trades table to JSON.
    Returns number of rows backed up.
    """
    rows = conn.execute("""
        SELECT id, symbol, bucket, direction, status,
               entry_price, entry_date, qty, stop_loss, target,
               exit_price, exit_date, pnl, pnl_pct, exit_reason,
               score_at_entry, regime_at_entry, notes,
               created_at, updated_at
        FROM trades
        ORDER BY id ASC
    """).fetchall()

    columns = [
        "id", "symbol", "bucket", "direction", "status",
        "entry_price", "entry_date", "qty", "stop_loss", "target",
        "exit_price", "exit_date", "pnl", "pnl_pct", "exit_reason",
        "score_at_entry", "regime_at_entry", "notes",
        "created_at", "updated_at"
    ]

    data = {
        "backup_time": datetime.now().isoformat(),
        "row_count":   len(rows),
        "trades":      [dict(zip(columns, row)) for row in rows],
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(BACKUP_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"Backup complete: {len(rows)} trades → {BACKUP_PATH}")
    return len(rows)


def restore(conn: sqlite3.Connection) -> int:
    """
    Read trades_backup.json and INSERT any trades not already in the DB.
    Uses INSERT OR IGNORE so it never creates duplicates.
    Returns number of rows inserted.
    """
    if not os.path.exists(BACKUP_PATH):
        logger.warning(f"No backup file found at {BACKUP_PATH}. Skipping restore.")
        return 0

    with open(BACKUP_PATH, "r") as f:
        data = json.load(f)

    trades     = data.get("trades", [])
    inserted   = 0

    for t in trades:
        try:
            conn.execute("""
                INSERT OR IGNORE INTO trades
                (id, symbol, bucket, direction, status,
                 entry_price, entry_date, qty, stop_loss, target,
                 exit_price, exit_date, pnl, pnl_pct, exit_reason,
                 score_at_entry, regime_at_entry, notes,
                 created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                t["id"], t["symbol"], t["bucket"], t["direction"], t["status"],
                t["entry_price"], t["entry_date"], t["qty"],
                t["stop_loss"], t["target"],
                t.get("exit_price"), t.get("exit_date"),
                t.get("pnl"), t.get("pnl_pct"), t.get("exit_reason"),
                t.get("score_at_entry"), t.get("regime_at_entry"),
                t.get("notes"), t.get("created_at"), t.get("updated_at"),
            ))
            inserted += conn.execute("SELECT changes()").fetchone()[0]
        except Exception as e:
            logger.error(f"Restore error for trade ID {t.get('id')}: {e}")

    conn.commit()
    logger.info(f"Restore complete: {inserted} new rows inserted from {len(trades)} in backup.")
    return inserted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup",  action="store_true", help="Backup trades to JSON")
    parser.add_argument("--restore", action="store_true", help="Restore trades from JSON")
    args = parser.parse_args()

    conn = get_connection()
    create_tables(conn)

    if args.backup:
        n = backup(conn)
        print(f"✅ Backed up {n} trades to {BACKUP_PATH}")

    elif args.restore:
        n = restore(conn)
        if n > 0:
            print(f"✅ Restored {n} trades from backup.")
        else:
            print("ℹ️  No new trades to restore (DB already up to date or no backup exists).")

    else:
        print("Usage: python backup_trades.py --backup | --restore")

    conn.close()
