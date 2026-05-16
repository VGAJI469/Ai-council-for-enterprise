"""Audit recorder for session-level event capture and report generation."""
import json
import os
from datetime import datetime
from typing import List, Dict
from utils.logger import get_logger, LOG_DIR, REPORT_DIR

logger = get_logger("audit")


class AuditRecorder:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.events: List[Dict] = []
        self.started_at = datetime.utcnow()

    def record_event(self, event_type: str, payload: Dict):
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": event_type,
            "payload": payload,
        }
        self.events.append(entry)
        # Also write a lightweight audit entry to audit log
        logger.info(f"audit_event: {event_type}", extra={"extra": {"session_id": self.session_id, "event": event_type, "payload": payload}})

    def record_error(self, error: Exception, context: Dict = None):
        ctx = context or {}
        self.record_event("error", {"error": str(error), "context": ctx})

    def write_report(self):
        os.makedirs(REPORT_DIR, exist_ok=True)
        report_path = os.path.join(REPORT_DIR, f"session_{self.session_id}.json")
        report = {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat() + "Z",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "events": self.events,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Wrote audit report", extra={"extra": {"session_id": self.session_id, "report_path": report_path}})
        return report_path
