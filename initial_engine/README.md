# Initial Classification Engine

Merchant keyword-matching engine that classifies bank transactions by matching
their `text` field against a pre-built merchant knowledge base.

## How it works

1. **Load** `merchant_kb.csv` in chunks (~395 MB, ~3.58M rows).
2. **Filter** to ~9k rows that have a non-empty `category`.
3. **Build** a case-insensitive Aho-Corasick automaton from the pipe-separated
   keyword variants (~30k keywords total).
4. **Search** each transaction's `text` in a single pass through the automaton.
5. When multiple keywords match, the **longest** keyword wins (most specific).

## Categories owned

Dining Out, Retail, Groceries, Health, Automotive, Entertainment,
Home Improvement, Travel, Information, Personal Care, Transport, Education,
Gambling, Gyms and other memberships, Pet Care, Donations, Utilities,
Telecommunications, Rent, Department Stores, Insurance, Debt Consolidation,
Debt Collection, Subscription TV

## Pipeline priority

50 — runs before income (100), liability (200), and transfer (300).
