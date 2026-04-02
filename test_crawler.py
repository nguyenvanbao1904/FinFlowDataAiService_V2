from app.services.crawler_service import VnstockCrawlerService

crawler = VnstockCrawlerService()
symbol = "VCB"
is_bank = True

print("=== Testing Financial Indicators ===")
inds, warnings = crawler.get_financial_indicators(symbol, is_bank)
print(f"Count: {len(inds)}, Warnings: {warnings}")
if inds:
    ind = inds[0]
    print(f"First record: year={ind.year}, pe={ind.pe}, roe={ind.roe}, roa={ind.roa}")
else:
    print("No data returned!")

print("\n=== Testing Income Statement ===")
incomes, warnings = crawler.get_income_statement(symbol, is_bank)
print(f"Count: {len(incomes)}, Warnings: {warnings}")
if incomes:
    inc = incomes[0]
    print(f"First record: year={inc.year}, profitAfterTax={inc.profitAfterTax}, netProfit={inc.netProfit}")
else:
    print("No data returned!")

print("\n=== Testing Balance Sheet ===")
balances, warnings = crawler.get_balance_sheet(symbol, is_bank)
print(f"Count: {len(balances)}, Warnings: {warnings}")
if balances:
    bal = balances[0]
    print(f"First record: year={bal.year}, totalAssets={bal.totalAssets}")
else:
    print("No data returned!")
