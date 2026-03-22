"""Data Ingester - Multi-source financial data ingestion."""

import logging
import pandas as pd
from pathlib import Path
logger = logging.getLogger(__name__)


class DataIngester:
    def __init__(self, config: dict):
        self.sources = config.get("sources", [])

    def ingest(self) -> dict:
        datasets = {}
        for source in self.sources:
            stype = source["type"]
            try:
                if source.get("path") and Path(source["path"]).exists():
                    data = pd.read_csv(source["path"]) if source["path"].endswith(".csv") else pd.read_json(source["path"])
                    datasets[stype] = data
                    logger.info(f"Ingested {stype}: {len(data)} records")
                else:
                    logger.warning(f"Source {stype} not found at {source.get('path')}, using empty DataFrame")
                    datasets[stype] = pd.DataFrame()
            except Exception as e:
                logger.error(f"Failed to ingest {stype}: {e}")
                datasets[stype] = pd.DataFrame()
        return datasets
