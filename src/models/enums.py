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