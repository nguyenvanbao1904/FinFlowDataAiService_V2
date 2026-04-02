# FireAnt — tên công ty & mô tả (thay cho scrape trang charts)

Trang ví dụ `https://fireant.vn/charts/content/symbols/FPT` là **SPA**: dữ liệu tab **Hồ sơ** lấy từ API, không nên crawl HTML (Cloudflare, JS, rủi ro vi phạm ToS).

## API chính thức

- Base (OpenAPI): [https://restv2.fireant.vn/](https://restv2.fireant.vn/) — spec: `/swagger/docs/v1`
- Endpoint: `GET /symbols/{symbol}/profile`  
  Trả về `CompanyInfo`: `companyName`, `shortName`, `businessAreas` (ngành/lĩnh vực dạng text), `overview` (giới thiệu), …
- Cây phân ngành ICB: `GET /icb` — thường có `industryCode` / `level` / `name` (field có thể khác tên theo spec; crawler chuẩn hoá trong `icb_tree_sync.py`).
- **Bắt buộc**: header `Authorization: Bearer <token>`  
  OAuth2 implicit / scopes — cần quyền **`symbols-read`** (xem định nghĩa trong swagger).

## Cấu hình crawler (FinFlow)

Trong `data_ai_service/.env` (đã có sẵn key `FIREANT_ACCESS_TOKEN=`) hoặc shell:

```bash
export FIREANT_ACCESS_TOKEN="..."   # Bearer token (có scope symbols-read)
# Tuỳ chọn:
# export FIREANT_API_BASE="https://restv2.fireant.vn"
```

Alias: `FIREANT_BEARER_TOKEN` (cùng ý nghĩa).

Khi **không** set token, crawler chỉ dùng vnstock VCI như trước.  
Khi **có** token, FireAnt được **ưu tiên** cho từng field; field trống sẽ **bù** từ VCI (ví dụ nhãn ngành / path ICB khi thiếu `businessAreas`).

**Lưu ý quan trọng (mã ICB số):** `Company.overview()` VCI **không** trả các cột `icb_code1`–`icb_code4`, chỉ có `icb_name2/3/4`. Khi **không** có `FIREANT_ACCESS_TOKEN`, mã số để khớp `industry_nodes` được lấy từ **`Listing.symbols_by_industries()`** (cache một lần / process). Khi **có** token, crawler **không** dùng listing VCI cho mã ICB — chỉ dùng FireAnt `GET /symbols/{symbol}/profile` (`icbCode`) + cây `GET /icb` để đồng nhất.

Cây ngành chuẩn: **`POST .../industry-nodes`**. Nếu có token: **chỉ** FireAnt **`GET /icb`** (không fallback VCI). Nếu không có token: vnstock VCI **`industries_icb`**.  
Công ty: **`POST .../companies`** gửi `industryIcbCode` (và/hoặc `industryNodeId`) — backend gán **`companies.industry_node_id`**; nhãn/mã ICB lấy qua join nút lá.

## Lấy token

Đăng ký / tạo ứng dụng OAuth theo hướng dẫn trên nền tảng FireAnt (trang API / tài khoản). Không hard-code token lên git; chỉ dùng biến môi trường hoặc secret store.
