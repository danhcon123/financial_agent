from enum import Enum

class SignalType(str, Enum):
    """Investment signal recommendations"""
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"
    NO_TRADE = "NO_TRADE"
    
class CritiqueSeverity(str, Enum):
    """Severity levels for critic issue"""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class StepStatus(str, Enum):
    """Execution step status"""
    START = "start"
    END = "end"
    ERROR = "error"

class IssueType(str, Enum):
    """Types of issues identified by critic"""
    REASONING = "REASONING" # fixable by analyst revision
    EVIDENCE_GAP = "EVIDENCE_GAP" # requires new data gathering

class TaskType(str, Enum):
    """Types of research tasks the Planner can emit"""
    PRICE_DATA = "PRICE_DATA"
    NEWS_SEARCH = "NEWS_SEARCH"
    FUNDAMENTALS = "FUNDAMENTALS"
    PEER_COMPARE = "PEER_COMPARE"
    EARNINGS_CHECK = "EARNINGS_CHECK"
    FILING_SUMMARY = "FILING_SUMMARY"

class TaskStatus(str, Enum):
    """Lifecycle status of a research task"""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    DONE = "DONE"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    