"""Vietnamese text utilities for chat — diacritics folding, feature labels, sanitization."""
from __future__ import annotations

import re

_VIET_TRANS = str.maketrans(
    {
        "à": "a", "á": "a", "ạ": "a", "ả": "a", "ã": "a",
        "â": "a", "ầ": "a", "ấ": "a", "ậ": "a", "ẩ": "a", "ẫ": "a",
        "ă": "a", "ằ": "a", "ắ": "a", "ặ": "a", "ẳ": "a", "ẵ": "a",
        "đ": "d",
        "è": "e", "é": "e", "ẹ": "e", "ẻ": "e", "ẽ": "e",
        "ê": "e", "ề": "e", "ế": "e", "ệ": "e", "ể": "e", "ễ": "e",
        "ì": "i", "í": "i", "ị": "i", "ỉ": "i", "ĩ": "i",
        "ò": "o", "ó": "o", "ọ": "o", "ỏ": "o", "õ": "o",
        "ô": "o", "ồ": "o", "ố": "o", "ộ": "o", "ổ": "o", "ỗ": "o",
        "ơ": "o", "ờ": "o", "ớ": "o", "ợ": "o", "ở": "o", "ỡ": "o",
        "ù": "u", "ú": "u", "ụ": "u", "ủ": "u", "ũ": "u",
        "ư": "u", "ừ": "u", "ứ": "u", "ự": "u", "ử": "u", "ữ": "u",
        "ỳ": "y", "ý": "y", "ỵ": "y", "ỷ": "y", "ỹ": "y",
    }
)

_FEATURE_LABELS: dict[str, str] = {
    "nim_pct": "biên lãi thuần",
    "other_income": "thu nhập ngoài lãi",
    "interbank_placements": "quy mô cho vay/đặt vốn liên ngân hàng",
    "assets_to_equity": "đòn bẩy tài chính (tài sản/vốn chủ sở hữu)",
    "gdp_ty_dong_log": "môi trường tăng trưởng kinh tế",
    "fee_and_commission_income": "thu nhập từ phí dịch vụ",
    "profit_after_tax": "mặt bằng lợi nhuận sau thuế nền",
    "revenue_current": "mặt bằng doanh thu hiện tại",
    "net_interest_income": "thu nhập lãi thuần",
    "cpi_inflation_yoy_pp": "lạm phát CPI",
    "usd_vnd_log": "tỷ giá USD/VND",
    "interest_deposit_12m_pct": "lãi suất tiền gửi 12 tháng",
    "interest_loan_midlong_pct": "lãi suất cho vay trung dài hạn",
    "interest_loan_short_pct": "lãi suất cho vay ngắn hạn",
    "hrc_log": "giá thép cuộn cán nóng HRC",
    "iron_ore_log": "giá quặng sắt",
    "coal_log": "giá than",
    "oil_brent_log": "giá dầu Brent",
    "nat_gas_log": "giá khí tự nhiên",
    "gold_gc_log": "giá vàng",
    "roe": "tỷ suất lợi nhuận trên vốn chủ (ROE)",
    "roa": "tỷ suất lợi nhuận trên tổng tài sản (ROA)",
    "gross_margin": "biên lợi nhuận gộp",
    "net_margin": "biên lợi nhuận ròng",
    "total_assets_reported": "tổng tài sản",
    "equity": "vốn chủ sở hữu",
    "net_debt_to_equity_pct": "nợ ròng/vốn chủ sở hữu",
    "asset_growth_yoy": "tăng trưởng tài sản YoY",
    "revenue_yoy_pct": "tăng trưởng doanh thu YoY",
    "profit_yoy_pct": "tăng trưởng lợi nhuận YoY",
    "revenue_momentum_pct": "đà tăng trưởng doanh thu",
    "profit_momentum_pct": "đà tăng trưởng lợi nhuận",
    "vnindex_growth_yoy_pct": "tăng trưởng VN-Index YoY",
}

_FUZZY_LABELS: list[tuple[str, str]] = [
    ("nim", "biên lãi thuần"),
    ("interbank", "quy mô cho vay/đặt vốn liên ngân hàng"),
    ("fee", "thu nhập từ phí dịch vụ"),
    ("commission", "thu nhập từ phí dịch vụ"),
    ("other_income", "thu nhập ngoài lãi"),
    ("gdp", "bối cảnh kinh tế vĩ mô"),
    ("macro", "bối cảnh kinh tế vĩ mô"),
    ("profit", "mặt bằng lợi nhuận nền"),
    ("revenue", "mặt bằng doanh thu nền"),
    ("interest", "lãi suất"),
    ("vnindex", "chỉ số VN-Index"),
    ("spread", "biên chênh giá"),
]


def to_ascii_lower(text: str) -> str:
    """Fast Vietnamese diacritics fold + lowercase."""
    return text.lower().translate(_VIET_TRANS)


def feature_label_vi(feature_name: str) -> str:
    """Map technical feature name to Vietnamese label for user-facing display."""
    key = (feature_name or "").strip().lower()
    if key in _FEATURE_LABELS:
        return _FEATURE_LABELS[key]
    for substr, label in _FUZZY_LABELS:
        if substr in key:
            return label
    return "một số chỉ tiêu tài chính nội tại khác"


def impact_level_vi(abs_score: float | None) -> str:
    """Classify SHAP/importance score into Vietnamese impact level."""
    s = abs(abs_score or 0.0)
    if s >= 0.04:
        return "mạnh"
    if s >= 0.02:
        return "vừa"
    return "nhẹ"


def sanitize_user_facing_message(text: str) -> str:
    """Clean technical leaks from LLM response while preserving markdown formatting."""
    if not text:
        return text

    def _repl_snake(match: re.Match[str]) -> str:
        token = match.group(0)
        label = feature_label_vi(token)
        if not label or "một số chỉ tiêu" in label:
            return "một số chỉ tiêu tài chính"
        return label

    cleaned = re.sub(r"\b[a-zA-Z]+(?:_[a-zA-Z0-9]+)+\b", _repl_snake, text)

    # Rewrite technical causal phrases into investor-friendly language.
    replacements = [
        (r"\btác động ngược chiều\b", "tạo áp lực lên"),
        (r"\btác động cùng chiều\b", "hỗ trợ"),
        (r"\btác động trung tính\b", "ảnh hưởng chưa rõ"),
        (r"\btác động tiêu cực\b", "gây áp lực"),
        (r"\btác động tích cực\b", "hỗ trợ"),
    ]
    for pattern, repl in replacements:
        cleaned = re.sub(pattern, repl, cleaned, flags=re.IGNORECASE)

    # Remove markdown bold (**text**), italic (*text*), and combinations.
    # Must be done in order: bold first (** before *).
    cleaned = re.sub(r'\*\*(.+?)\*\*', r'\1', cleaned)
    cleaned = re.sub(r'\*(.+?)\*', r'\1', cleaned)
    cleaned = re.sub(r'__(.+?)__', r'\1', cleaned)
    cleaned = re.sub(r'(?<!\w)_(.+?)_(?!\w)', r'\1', cleaned)

    # Remove markdown headers (# Text -> Text)
    cleaned = re.sub(r'^#{1,6}\s+', '', cleaned, flags=re.MULTILINE)

    # Remove orphan references to generic labels in parens.
    cleaned = re.sub(r"\(\s*một số chỉ tiêu tài chính\s*\)", "", cleaned)
    # Collapse multiple spaces but preserve markdown line breaks.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()
