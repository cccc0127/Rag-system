import logging
import re
from typing import Dict, Tuple


logger = logging.getLogger(__name__)


class PrivacyScorer:
    """Estimate chunk privacy sensitivity and map it to epsilon/delta."""

    def __init__(
        self,
        model: str = "valhalla/distilbart-mnli-12-1",
        device: int = -1,
        enable_nlp: bool = True,
        epsilon_min: float = 0.1,
        epsilon_max: float = 10.0,
    ):
        self.model = model
        self.device = device
        self.enable_nlp = enable_nlp
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self._classifier = None
        self._classifier_failed = False

        self.candidate_labels = [
            "personal sensitive information",
            "financial and banking records",
            "corporate confidential data",
            "public general information",
        ]

        self.regex_patterns: Dict[str, re.Pattern] = {
            "ID": re.compile(r"(?<!\d)(?:\d{17}[\dXx]|\d{15})(?!\d)"),
            "Phone": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
            "Bank": re.compile(r"(?<!\d)(?:4\d{15}|5[1-5]\d{14}|62\d{14,17})(?!\d)"),
            "Money": re.compile(
                r"(\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?\s?(?:元|万元|亿元|USD|CNY|RMB|dollars)",
                re.IGNORECASE,
            ),
            "Email": re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
        }

        self.keyword_patterns: Dict[str, re.Pattern] = {
            "PersonalStrong": re.compile(r"(身份证号?|手机号|电话号码|银行卡号?|住址|家庭住址|病历|诊断记录)"),
            "PersonalWeak": re.compile(r"(参会人员|人员名单|联系人|客户|员工|姓名|包括[\u4e00-\u9fff]{2,4})"),
            "CorporateStrong": re.compile(
                r"(内部资料|请勿外传|商业秘密|机密|保密|未公开|客户名单|源代码|研发计划|投标文件)"
            ),
            "CorporateWeak": re.compile(r"(营收目标|战略规划|成本明细|利润率|报价|预算|合同|供应商)"),
            "Public": re.compile(r"(首都|人口|面积|地理|历史|百科|公开|新闻|常识|天气|城市|国家|省会)"),
        }

    @property
    def classifier(self):
        if not self.enable_nlp or self._classifier_failed:
            return None
        if self._classifier is None:
            try:
                from transformers import pipeline

                self._classifier = pipeline(
                    "zero-shot-classification",
                    model=self.model,
                    device=self.device,
                )
            except Exception as exc:
                self._classifier_failed = True
                logger.warning("Privacy zero-shot classifier is unavailable: %s", exc)
                return None
        return self._classifier

    def get_sensitivity_score(self, text_chunk: str) -> float:
        """Return raw sensitivity score in [0.1, 10.0]; higher means more sensitive."""
        final_sensitivity = self._estimate_sensitivity(text_chunk)
        return _clip(0.1 + 9.9 * final_sensitivity, 0.1, 10.0)

    def get_privacy_profile(self, text_chunk: str) -> Dict[str, float]:
        """Evaluate a chunk once and expose both raw score and legacy DP params."""
        final_sensitivity = self._estimate_sensitivity(text_chunk)
        final_eps = self._sensitivity_to_epsilon(final_sensitivity)
        dynamic_delta = 2.0 * (1.0 + 0.5 * final_sensitivity)

        return {
            "raw_sensitivity_score": float(_clip(0.1 + 9.9 * final_sensitivity, 0.1, 10.0)),
            "privacy_epsilon": float(final_eps),
            "dynamic_delta": float(dynamic_delta),
        }

    def get_privacy_params(self, text_chunk: str) -> Tuple[float, float]:
        profile = self.get_privacy_profile(text_chunk)
        return profile["privacy_epsilon"], profile["dynamic_delta"]

    def _estimate_sensitivity(self, text_chunk: str) -> float:
        regex_score = self._quick_regex_scan(text_chunk)
        keyword_score = self._keyword_scan(text_chunk)
        heuristic_score = self._heuristic_semantic_scan(text_chunk)

        rule_score = _clip(regex_score + keyword_score + heuristic_score, 0.0, 1.0)

        # Performance balance: direct identifiers are already reliable enough.
        # Skip expensive NLP, but keep a continuous score instead of forcing eps=0.1.
        if regex_score > 0.8:
            final_sensitivity = rule_score
        else:
            nlp_score = self._semantic_sensitivity(text_chunk)
            if self._looks_public(text_chunk) and rule_score < 0.2:
                nlp_score = min(nlp_score, 0.12)
            final_sensitivity = _clip(rule_score * 0.65 + nlp_score * 0.35, 0.0, 1.0)

        return float(final_sensitivity)

    def _semantic_sensitivity(self, text: str) -> float:
        classifier = self.classifier
        if classifier is None or not text.strip():
            return 0.0

        try:
            result = classifier(text, self.candidate_labels)
        except Exception as exc:
            logger.warning("Privacy zero-shot classification failed: %s", exc)
            return 0.0

        scores = dict(zip(result["labels"], result["scores"]))
        public_score = scores.get("public general information", 0.0)
        sensitive_score = max(
            scores.get("personal sensitive information", 0.0),
            scores.get("financial and banking records", 0.0),
            scores.get("corporate confidential data", 0.0),
        )

        return _clip(sensitive_score - public_score, 0.0, 1.0)

    def _quick_regex_scan(self, text: str) -> float:
        weights = {
            "ID": 0.82,
            "Phone": 0.68,
            "Bank": 0.78,
            "Money": 0.18,
            "Email": 0.35,
        }
        score = 0.0
        for name, pattern in self.regex_patterns.items():
            matches = len(pattern.findall(text))
            if matches:
                score += weights[name] * matches
        return _clip(score, 0.0, 1.0)

    def _keyword_scan(self, text: str) -> float:
        score = 0.0
        if self.keyword_patterns["PersonalStrong"].search(text):
            score += 0.22
        if self.keyword_patterns["PersonalWeak"].search(text):
            score += 0.34
        if self.keyword_patterns["CorporateStrong"].search(text):
            score += 0.45
        if self.keyword_patterns["CorporateWeak"].search(text):
            score += 0.28
        return _clip(score, 0.0, 1.0)

    def _heuristic_semantic_scan(self, text: str) -> float:
        score = 0.0
        has_person_name = re.search(r"[\u4e00-\u9fff]{2,3}", text) is not None
        has_private_context = re.search(r"(会议|参会|名单|联系人|客户|员工|候选人)", text) is not None
        has_public_context = self._looks_public(text)

        if has_person_name and has_private_context and not has_public_context:
            score += 0.12
        if re.search(r"(余额|收入|薪资|工资|借款|贷款|资产)", text):
            score += 0.18
        if re.search(r"(内部|外传|保密|目标|规划|未公开)", text):
            score += 0.12

        return _clip(score, 0.0, 1.0)

    def _looks_public(self, text: str) -> bool:
        return self.keyword_patterns["Public"].search(text) is not None

    def _sensitivity_to_epsilon(self, sensitivity: float) -> float:
        sensitivity = _clip(sensitivity, 0.0, 1.0)
        # Smooth continuous mapping:
        # 0.00 -> 10.0, 0.35 -> about 4.3, 0.65 -> about 1.3, 1.00 -> 0.1.
        eps = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * ((1.0 - sensitivity) ** 2.2)
        return _clip(eps, self.epsilon_min, self.epsilon_max)


def _clip(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, float(value)))
