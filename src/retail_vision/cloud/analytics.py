"""Cloud-side event store and analytics queries (DuckDB).

Only events reach the cloud, so a single analytical database per region can
serve hundreds of stores. DuckDB keeps the POC self-contained; the SQL is
plain enough to move to BigQuery/Snowflake unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import duckdb

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_type   VARCHAR,
    store_id     VARCHAR,
    camera_id    VARCHAR,
    ts           DOUBLE,
    global_id    VARCHAR,
    role         VARCHAR,
    zone         VARCHAR,
    zone_kind    VARCHAR,
    employee_id  VARCHAR,
    confidence   DOUBLE,
    needs_review BOOLEAN,
    dwell_seconds DOUBLE,
    payload      JSON
);
"""


class EventStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.con = duckdb.connect(str(path))
        self.con.execute(SCHEMA)

    # Writes ------------------------------------------------------------------------------

    def insert(self, events: Iterable[dict]) -> int:
        rows = []
        for e in events:
            payload = e.get("payload") or {}
            rows.append(
                (
                    e["event_type"],
                    e["store_id"],
                    e["camera_id"],
                    float(e["timestamp"]),
                    e["global_id"],
                    e.get("role"),
                    e.get("zone"),
                    payload.get("zone_kind"),
                    e.get("employee_id"),
                    e.get("confidence"),
                    bool(payload.get("needs_review", False)),
                    payload.get("dwell_seconds"),
                    json.dumps(payload),
                )
            )
        if rows:
            self.con.executemany(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
            )
        return len(rows)

    def load_jsonl(self, path: str | Path) -> int:
        with open(path, encoding="utf-8") as fh:
            return self.insert(json.loads(line) for line in fh if line.strip())

    # Reads -------------------------------------------------------------------------------

    def _rows(self, sql: str, params: list | None = None) -> list[dict]:
        cur = self.con.execute(sql, params or [])
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]

    def summary(self, store_id: str | None = None) -> dict:
        where = "WHERE store_id = ?" if store_id else ""
        params = [store_id] if store_id else []
        row = self._rows(
            f"""
            SELECT
                COUNT(*)                                                  AS events,
                COUNT(DISTINCT global_id)                                 AS unique_people,
                COUNT(DISTINCT global_id) FILTER (WHERE role = 'customer') AS unique_customers,
                COUNT(DISTINCT employee_id) FILTER (WHERE employee_id IS NOT NULL)
                                                                          AS employees_seen,
                COUNT(*) FILTER (WHERE event_type = 'product_interaction')  AS product_interactions,
                COUNT(*) FILTER (WHERE needs_review)                      AS needs_review,
                MIN(ts) AS first_ts, MAX(ts) AS last_ts
            FROM events {where}
            """,
            params,
        )[0]
        return row

    def store_visits(self) -> list[dict]:
        """Unique customers that crossed an entrance zone, per store."""
        return self._rows(
            """
            SELECT store_id, COUNT(DISTINCT global_id) AS visitors
            FROM events
            WHERE event_type = 'zone_entered' AND zone_kind = 'entrance' AND role = 'customer'
            GROUP BY store_id ORDER BY store_id
            """
        )

    def zone_dwell(self, store_id: str | None = None) -> list[dict]:
        where = "AND store_id = ?" if store_id else ""
        params = [store_id] if store_id else []
        return self._rows(
            f"""
            SELECT store_id, zone, zone_kind, role,
                   COUNT(*)                        AS visits,
                   ROUND(AVG(dwell_seconds), 2)    AS avg_dwell_s,
                   ROUND(MAX(dwell_seconds), 2)    AS max_dwell_s
            FROM events
            WHERE event_type IN ('zone_exited', 'seat_released') {where}
            GROUP BY ALL ORDER BY store_id, zone, role
            """,
            params,
        )

    def shelf_engagement(self) -> list[dict]:
        """Per shelf: how many customers stopped by vs how many interacted (conversion)."""
        return self._rows(
            """
            WITH visits AS (
                SELECT store_id, zone, COUNT(DISTINCT global_id) AS visitors
                FROM events
                WHERE event_type = 'zone_entered' AND zone_kind = 'shelf' AND role = 'customer'
                GROUP BY ALL
            ), inter AS (
                SELECT store_id, zone, COUNT(DISTINCT global_id) AS interacting
                FROM events
                WHERE event_type = 'product_interaction' AND role = 'customer'
                GROUP BY ALL
            )
            SELECT v.store_id, v.zone, v.visitors, COALESCE(i.interacting, 0) AS interacting,
                   ROUND(COALESCE(i.interacting, 0) * 100.0 / v.visitors, 1) AS interaction_rate_pct
            FROM visits v LEFT JOIN inter i USING (store_id, zone)
            ORDER BY v.store_id, v.zone
            """
        )

    def employee_presence(self) -> list[dict]:
        """Time span each identified employee was observed and how many zones/cameras saw them."""
        return self._rows(
            """
            SELECT store_id, employee_id,
                   ROUND(MAX(ts) - MIN(ts), 1)     AS observed_span_s,
                   COUNT(DISTINCT camera_id)       AS cameras,
                   COUNT(DISTINCT zone)            AS zones,
                   ROUND(SUM(dwell_seconds) FILTER (WHERE zone_kind = 'seat'), 1) AS seated_s,
                   ROUND(AVG(confidence), 3)       AS avg_confidence
            FROM events
            WHERE employee_id IS NOT NULL
            GROUP BY ALL ORDER BY store_id, employee_id
            """
        )

    def cross_camera_journeys(self, limit: int = 20) -> list[dict]:
        """People seen by more than one camera: the evidence that ReID is deduplicating."""
        return self._rows(
            """
            SELECT store_id, global_id, role,
                   COUNT(DISTINCT camera_id) AS cameras,
                   STRING_AGG(DISTINCT camera_id, ',' ORDER BY camera_id) AS camera_list,
                   ROUND(MAX(ts) - MIN(ts), 1) AS journey_s
            FROM events
            GROUP BY ALL HAVING COUNT(DISTINCT camera_id) > 1
            ORDER BY journey_s DESC LIMIT ?
            """,
            [limit],
        )

    def review_queue(self, limit: int = 50) -> list[dict]:
        """Low-confidence identity decisions to label; feeds the ReID retraining loop."""
        return self._rows(
            """
            SELECT store_id, camera_id, ts, global_id, role, employee_id, confidence, payload
            FROM events
            WHERE needs_review AND event_type = 'person_seen'
            ORDER BY confidence ASC, ts ASC LIMIT ?
            """,
            [limit],
        )

    def close(self) -> None:
        self.con.close()
