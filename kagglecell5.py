#correlation
correlation = df['Price'].corr(df['ONI'])
print(f"Correlation between Coffee Price and ONI: {correlation:.4f}")

#lagged correlations
lags = range(-12, 13)
correlations = [df['Price'].corr(df['ONI'].shift(lag)) for lag in lags]

max_corr_idx = np.argmax(np.abs(correlations))
max_lag = lags[max_corr_idx]
max_corr = correlations[max_corr_idx]

print(f"\nMaximum absolute correlation: {max_corr:.4f} at lag {max_lag} months")
