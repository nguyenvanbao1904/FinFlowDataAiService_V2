"""
Danh sách mã chứng khoán VN30 và VN100 dùng cho eval pipeline.
Cập nhật khi HOSE thay đổi rổ (thường mỗi 6 tháng).

Usage:
    from symbol_lists import VN30, VN100, VN100_CSV

    # Hoặc chạy trực tiếp để in ra:
    python scripts/financial_training/symbol_lists.py
"""

VN30 = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
]

VN100 = [
    "ACB", "BCM", "BID", "BMP", "BSI", "BSR", "BVH", "BWE", "CII", "CMG",
    "CTD", "CTG", "CTR", "CTS", "DBC", "DCM", "DGC", "DGW", "DIG", "DPM",
    "DSE", "DXG", "DXS", "EIB", "EVF", "FPT", "FRT", "FTS", "GAS", "GEE",
    "GEX", "GMD", "GVR", "HAG", "HCM", "HDB", "HDC", "HDG", "HHV", "HPG",
    "HSG", "HT1", "IDC", "IMP", "KBC", "KDC", "KDH", "KOS", "LPB", "MBB",
    "MSB", "MSN", "MWG", "NAB", "NKG", "NLG", "NT2", "NVL", "OCB", "PAN",
    "PC1", "PDR", "PHR", "PLX", "PNJ", "POW", "PVD", "PVT", "REE", "SAB",
    "SBT", "SCS", "SHB", "SIP", "SJS", "SSB", "SSI", "STB", "SZC", "TCB",
    "TCH", "TPB", "VCB", "VCG", "VCI", "VGC", "VHC", "VHM", "VIB", "VIC",
    "VIX", "VJC", "VND", "VNM", "VPB", "VPI", "VPL", "VRE", "VSC", "VTP",
]

VN30_CSV = ",".join(VN30)
VN100_CSV = ",".join(VN100)

if __name__ == "__main__":
    print(f"VN30  ({len(VN30)} mã):  {VN30_CSV}")
    print(f"VN100 ({len(VN100)} mã): {VN100_CSV}")
    print()
    print("--- Lệnh train ---")
    print()
    print("# Eval pipeline (đánh giá model, so sánh predict vs actual):")
    print("venv/bin/python scripts/financial_training/run_final_model_pipeline.py \\")
    print("  --train-target-year-max 2024 --predict-target-year 2025 \\")
    print(f'  --symbols "{VN100_CSV}" \\')
    print("  --out-dir artifacts/models/eval_pipeline")
    print()
    print("# Production pipeline (chatbot dùng):")
    print("venv/bin/python scripts/financial_training/run_final_model_pipeline.py \\")
    print("  --train-target-year-max 2025 --predict-target-year 2026 \\")
    print("  --out-dir artifacts/models/production_pipeline")
