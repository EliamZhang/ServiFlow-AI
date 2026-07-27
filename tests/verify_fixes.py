import pandas as pd
import re
from collections import Counter

df = pd.read_excel('output/classification_report.xlsx')

def norm(text):
    if pd.isna(text): return ''
    return re.sub(r'\s+', ' ', str(text)).lower().strip()

# Complete rule set with all modifications
all_rules = [
    # HIGH_CONFIDENCE
    (r'^direct debit$', 'et'),
    (r'\btransferwise sydney\b', 'et'),
    (r'^debit card purchase (wise|taptap send) sydney', 'et'),
    (r'^fast pymt in', 'et'),
    (r'^withdrawal mobile \d+ tfr westpac cho loan$', 'et'),
    (r'^imt \d+', 'et'),  # NEW
    (r'^anz mobile banking payment \d+ to [a-z].*$', 'et'),
    (r'\b(payid|osko|npp)\b', 'et'),
    (r'\bfast transfer\b', 'et'),
    (r'\b(paypal australia|worldremit)\b', 'et'),
    (r'\bcommbank app\b', 'et'),
    (r'^direct debit \d+ paypal australia \d+$', 'et'),
    (r'^payment by authority to paypal australia \d+$', 'et'),
    (r'^payment to paypal australia \d+$', 'et'),
    (r'^paypal australia \d+$', 'et'),
    (r'\btfr (?:to|from) .*\b(mob|mobile)\b', 'et'),
    (r'\bwithdrawal mobile\b.*\bpymt\b', 'et'),
    (r'^transfer credit (?!online\b)', 'et'),
    (r'^transfer debit (?!online\b)', 'et'),
    (r'^internet (?:withdrawal|deposit|external transfer)\b', 'et'),  # FIXED
    (r'\btfr (?:to|from)\b', 'et'),
    (r'\bnabpay\d+', 'et'),
    (r'\bphone/internet tfr\b', 'et'),
    (r'\btransferred (?:to|from) \d', 'et'),
    # MEDIUM_CONFIDENCE
    (r'\bwestpa\s', 'et'),
    (r'\btransferred to \d{3,6} \d+\b', 'et'),
    (r'^transfer (to|from) sav \d+ net#\d+$', 'et'),
    (r'^scheduled payment to a cba account \d+ \d+$', 'et'),
    (r'^anz m-banking transfer \d+ from \d+$', 'et'),
    (r'^internet transfer credit from \d+ ref no \d+$', 'et'),
    (r'^internet transfer debit to \d+ reference no \d+$', 'et'),
    (r'^ib transfer \d+ to \d{3}-\d{3}-\d+ \d+:\d+(?:am|pm) tfd$', 'et'),
    (r'^ib transfer \d+ from \d{3}-\d{3}-\d+ \d+:\d+(?:am|pm) tfc$', 'et'),
    (r'^transfer to cba a/c commbank app\b', 'et'),  # FIXED
    (r'^transfer from commbank app\b', 'et'),  # FIXED
    (r'\btransfer (to|from) xx\d{4}\b', 'et'),
    (r'^\. tfd$', 'et'),
    (r'^\. tfc$', 'et'),
    (r'^me tfd$', 'et'),
    (r'^x tfc$', 'et'),
    (r'^save tfc$', 'et'),
    (r'^j tfd$', 'et'),
    (r'^b tfd$', 'et'),  # NEW
    (r'^b tfc$', 'et'),  # NEW
    (r'^n tfd$', 'et'),  # NEW
    (r'^h tfd$', 'et'),  # NEW
    (r'^automatic payment$', 'et'),  # NEW
    (r'^periodic transfer from\b', 'et'),  # NEW
    (r'^rtgs funds credit$', 'et'),  # NEW
    (r'^transfer (debit|credit) (?!online )[a-z].*?[a-z]\d{10,}', 'et'),
]

# === ET verification ===
et_mask = (df['category'] == 'External Transfers') & (df['finv_category'].isna())
et_empty = df[et_mask]

newly_covered = 0
still_missed = []
for idx, row in et_empty.iterrows():
    text_norm = norm(row['text'])
    matched = False
    for pattern, _ in all_rules:
        if re.search(pattern, text_norm, re.IGNORECASE):
            matched = True
            break
    if matched:
        newly_covered += 1
    else:
        still_missed.append(str(row['text'])[:120])

patterns = Counter()
for text in still_missed:
    t = text.upper()
    if 'DIRECT DEBIT' in t:
        patterns['DIRECT_DEBIT (illion misclassify - utility bill)'] += 1
    elif 'PAYMENT TO FLEET' in t:
        patterns['PAYMENT_TO_FLEET (illion misclassify)'] += 1
    elif 'BPAY' in t:
        patterns['BPAY (illion misclassify)'] += 1
    elif 'BILL PAY' in t:
        patterns['BILL_PAY (illion misclassify)'] += 1
    elif 'DEBIT CARD PURCHASE RMTLY' in t:
        patterns['REMITLY (intl remittance - can add later)'] += 1
    elif 'WESTERN UNION' in t:
        patterns['WESTERN_UNION (intl remittance - can add later)'] += 1
    elif 'TRANSFER' in t:
        patterns['TRANSFER_OTHER (edge cases)'] += 1
    elif 'MISCELLANEOUS' in t:
        patterns['MISC (uncertain)'] += 1
    else:
        patterns['OTHER'] += 1

print('=' * 60)
print('External Transfers Fix Results')
print('=' * 60)
print(f'Original gap: 487')
print(f'Newly covered: {newly_covered} ({newly_covered/487*100:.1f}%)')
print(f'Still missed: {len(still_missed)} ({len(still_missed)/487*100:.1f}%)')
print()
print('Still missed breakdown:')
for p, c in patterns.most_common():
    print(f'  {p}: {c}')

# === Fees verification ===
def norm_fee(text):
    if pd.isna(text): return ''
    return re.sub(r'\s+', ' ', str(text)).strip()

interest_rules = [
    (r'^INTEREST\s+CHARGES\s+-\s+PUR', 'fee'),
    (r'^INTEREST\s+CHARGES\s+-\s+CAS', 'fee'),
    (r'^Interest\s+Charges\s+-\s+Purch', 'fee'),
    (r'^Interest\s+Charges\s+-\s+Cash', 'fee'),
    (r'^INTEREST\s+ON\s+CASH\s+ADV', 'fee'),
    (r'^CASH\s+ADVANCE\s+INTEREST', 'fee'),
    (r'^VISA\s+PURCHASE\s+INTEREST', 'fee'),
    (r'^INSTALMENT\s+PLAN\s+INTEREST', 'fee'),
    (r'^INTEREST\s+CHARGED\s+ON\s+PURCHASES', 'fee'),
    (r'^INTEREST\s+-\s+BASE\s+PLAN', 'fee'),
    (r'^INTEREST\s+CHARGED\s+INTEREST\s+CHARGED', 'fee'),
    (r'^INTEREST\s+CHARGED$', 'fee'),
    (r'^DEBIT\s+INTEREST\s+CHARGED', 'fee'),
    (r'^Debit\s+Interest$', 'fee'),
    (r'^Interest\s+charged$', 'fee'),
    (r'^INTEREST\s+CHARGES$', 'fee'),
    (r'^INTEREST\s+DEBIT$', 'fee'),
    (r'^INTEREST$', 'fee'),
    (r'^Interest$', 'fee'),
]

fees_mask = (df['category'] == 'Fees') & (df['finv_category'].isna())
fees_empty = df[fees_mask]
fee_covered = 0
fee_missed = []
for idx, row in fees_empty.iterrows():
    tn = norm_fee(row['text'])
    matched = False
    for pat, _ in interest_rules:
        if re.search(pat, tn):
            matched = True
            break
    if matched:
        fee_covered += 1
    else:
        fee_missed.append(str(row['text'])[:120])

print()
print('=' * 60)
print('Fees Fix Results')
print('=' * 60)
print(f'Original gap: 161')
print(f'Newly covered: {fee_covered} ({fee_covered/161*100:.1f}%)')
print(f'Still missed: {len(fee_missed)}')
for t in fee_missed:
    print(f'  [{t}]')

print()
print('=' * 60)
print(f'SUMMARY: Total covered {newly_covered + fee_covered} rows')
print(f'  ET: {newly_covered}/487 = {newly_covered/487*100:.1f}%')
print(f'  Fees: {fee_covered}/161 = {fee_covered/161*100:.1f}%')
