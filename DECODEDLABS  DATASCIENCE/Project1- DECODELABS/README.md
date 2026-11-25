# Project 1: Advanced EDA & Feature Engineering
**DecodeLabs Data Science Industrial Training, 2026 Batch**

## Overview

This project takes a raw, 1,200-row e-commerce orders dataset and transforms it into a clean, feature-rich dataset ready for machine learning, without silently deleting, guessing, or distorting the underlying data.

The dataset covers order details (product, quantity, pricing), customer behavior (cart size vs. actual purchase), payment method, order status, and coupon usage across a January 2023 to June 2025 window.

## Objectives

- Handle missing data using statistical and business reasoning, not blind imputation
- Detect outliers using the IQR method and evaluate whether they are genuine data errors or legitimate values
- Engineer at least 3 new predictive features from existing columns

## 1. Missing Value Handling

**Finding:** 309 of 1,200 rows (25.75%) had a blank `CouponCode`. Every other column was fully populated.

**Reasoning:** A missing coupon code does not mean "unknown value to estimate", it means no coupon was applied to that order. Statistically, a >20% missing rate would typically call for KNN-based estimation. Applying that here would have meant inventing a fake coupon code for 309 real orders, corrupting the dataset rather than cleaning it.

**Decision:** Filled blanks with an explicit `"No Coupon"` label, preserving it as a legitimate category alongside the three real coupon codes (`FREESHIP`, `WINTER15`, `SAVE10`).

```python
df["CouponCode"] = df["CouponCode"].fillna("No Coupon")
```

## 2. Outlier Detection (IQR Method)

Checked `Quantity`, `UnitPrice`, and `TotalPrice` using the standard IQR fence:

| Column | Q1 | Q3 | IQR | Fence | Outliers Found |
|---|---|---|---|---|---|
| Quantity | 2.00 | 4.00 | 2.00 | [-1.00, 7.00] | 0 |
| UnitPrice | 186.06 | 521.57 | 335.51 | [-317.20, 1024.83] | 0 |
| TotalPrice | 410.52 | 1578.47 | 1167.95 | [-1341.41, 3330.41] | 8 |

**Investigation:** All 8 `TotalPrice` outliers shared the same pattern: `Quantity = 5` (the dataset's maximum) paired with a high `UnitPrice`. Verified the math on each: e.g. `670.75 × 5 = 3353.75`, exactly matching the recorded `TotalPrice`.

**Decision:** Left untouched. These are legitimate high-value orders, not data errors. `TotalPrice` is a derived field (`Quantity × UnitPrice`); capping it at the statistical fence would break its own formula and introduce a worse inconsistency than the one being addressed. Statistically unusual is not the same as factually wrong, and this project treats that distinction as a first-class decision, not an afterthought.

## 3. Feature Engineering

| Feature | Formula | Business Question It Answers |
|---|---|---|
| `OrderMonth` | `Date.dt.month` | Does order volume shift by month or season? |
| `HasCoupon` | `1` if `CouponCode != "No Coupon"` else `0` | Do coupon users behave differently from non-users? |
| `ItemsRemovedFromCart` | `ItemsInCart − Quantity` | How many items does a customer add but never buy, and does that gap predict cancellations or returns? |

`ItemsRemovedFromCart` came directly from a pattern noticed while exploring the data: `ItemsInCart` is always greater than or equal to `Quantity` across every single row, meaning cart abandonment behavior was sitting in the raw data the whole time, just never surfaced as its own feature. A sanity check confirms the value is never negative across all 1,200 rows, verifying the assumption held for the full dataset.

## Tech Stack

Python, Pandas, NumPy

## Files

- `project1_cleaning.py` — full cleaning and feature engineering script
- `dataset_cleaned_final.xlsx` — final output dataset

## Key Takeaway

Every fix in this project was made after asking what the data actually represents, not just what a formula outputs. A missing value, a statistical outlier, or a raw column difference can each mean something real about customer behavior if you check before you clean.
