# =========================================
# PROJECT 1: Advanced EDA & Feature Engineering
# DecodeLabs Data Science Industrial Training, 2026
# =========================================

import pandas as pd

# STEP 1: Load the dataset
df = pd.read_excel("Dataset_for_Data_Analytics.xlsx")

print(df.head())
print(df.shape)
print(df.isnull().sum())

# =========================================
# STEP 2: Handle missing values (CouponCode)
# =========================================
# Found: 309 of 1200 rows (25.75%) had a blank CouponCode.
# Decision: a blank coupon code means "no coupon was used",
# not an unknown value to estimate. Labeled explicitly instead
# of guessing a fake code with KNN imputation.

df["CouponCode"] = df["CouponCode"].fillna("No Coupon")

print(df["CouponCode"].isnull().sum())   # 0
print(df["CouponCode"].value_counts())

# =========================================
# STEP 3: Outlier detection (IQR method)
# =========================================
for col in ["Quantity", "UnitPrice", "TotalPrice"]:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower_fence) | (df[col] > upper_fence)]
    print(f"{col}: {len(outliers)} outliers found")

# RESULT: Quantity 0, UnitPrice 0, TotalPrice 8 outliers.
# All 8 have Quantity=5 with high UnitPrice; TotalPrice = Quantity x UnitPrice
# checks out exactly. These are legitimate large orders, not data errors.
# DECISION: left untouched. Capping a derived field (TotalPrice) would
# break its own formula and introduce a worse error than the one
# being "fixed".

# =========================================
# STEP 4: Feature engineering (3 new features)
# =========================================

# Feature 1: OrderMonth
# Business question: does order volume shift by month/season?
df["OrderMonth"] = df["Date"].dt.month

# Feature 2: HasCoupon
# Business question: do coupon users behave differently from non-users?
df["HasCoupon"] = df["CouponCode"].apply(lambda x: 0 if x == "No Coupon" else 1)

# Feature 3: ItemsRemovedFromCart
# Business question: how many items did a customer add to cart but
# never actually buy? A bigger gap can signal price hesitation,
# stock concerns, or checkout friction.
df["ItemsRemovedFromCart"] = df["ItemsInCart"] - df["Quantity"]

# Sanity check: this value should never be negative
assert (df["ItemsRemovedFromCart"] >= 0).all(), "Data integrity issue: Quantity exceeds ItemsInCart somewhere"

print(df[["Date","OrderMonth","CouponCode","HasCoupon","ItemsInCart","Quantity","ItemsRemovedFromCart"]].head(8))

# =========================================
# Save final cleaned + engineered dataset
# =========================================
df.to_excel("dataset_cleaned_final.xlsx", index=False)
print("Project 1 complete. Saved final file.")
