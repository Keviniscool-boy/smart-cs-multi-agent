"""Decision rules for the simulated after-sales workflow."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml

POLICY_PATH = Path(__file__).resolve().parents[1] / "knowledge_base" / "POL-AFTER-MVP-001.md"


def _policy_reference(section: str) -> str:
    try:
        parts = POLICY_PATH.read_text(encoding="utf-8").split("---", 2)
        if len(parts) != 3 or parts[0].strip():
            return ""
        policy = yaml.safe_load(parts[1])
        if not isinstance(policy, dict) or policy.get("status") != "published" or policy.get("superseded_by") or policy.get("rule_set") != "after_sales_mvp_1":
            return ""
        start = date.fromisoformat(str(policy["effective_date"]))
        end = date.fromisoformat(str(policy["expiration_date"])) if policy.get("expiration_date") else None
        if start > date.today() or (end and end < date.today()) or not policy.get("doc_id") or not policy.get("version"):
            return ""
        return f"{policy['doc_id']}-v{policy['version']}#{section}"
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
        return ""


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    reason: str
    policy_ref: str


def evaluate(application_type: str | None, product_type: str | None,
             signed_at: datetime | None, now: datetime) -> PolicyDecision:
    """Determine eligibility for submission; accepted requests still need review."""
    if application_type not in {"no_reason", "quality_issue"}:
        return PolicyDecision("manual_review", "申请类型未确认", "")
    if product_type != "normal":
        return PolicyDecision("manual_review", "商品类型未确认或需要特殊审核", "")
    reference = _policy_reference("质量问题" if application_type == "quality_issue" else "无理由退货")
    if not reference:
        return PolicyDecision("manual_review", "售后政策不可用，需要人工核验", "")
    if application_type == "quality_issue":
        return PolicyDecision("manual_review", "质量问题需人工核验证据", reference)
    if (signed_at is None or signed_at.tzinfo is None or signed_at.utcoffset() is None
            or now.tzinfo is None or now.utcoffset() is None or signed_at > now):
        return PolicyDecision("manual_review", "签收时间无法确认", reference)
    if now - signed_at > timedelta(days=7):
        return PolicyDecision("reject", "无理由退货申请已超过签收后 7 天", reference)
    return PolicyDecision("allow_submission", "符合无理由退货申请时限，仍需人工审核", reference)
