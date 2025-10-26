fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle('ENSO Impact on Arabica Coffee Prices (1960-2024)', 
             fontsize=16, fontweight='bold')

#time series comparison
ax1 = axes[0]
ax1_twin = ax1.twinx()

# Plot coffee
line1 = ax1.plot(df['Date'], df['Price'], color='#8B4513', linewidth=1.5, label='Coffee Price')
ax1.set_ylabel('Coffee Price ($/kg)', fontsize=12, color='#8B4513')
ax1.tick_params(axis='y', labelcolor='#8B4513')

# oni plot
line2 = ax1_twin.plot(df['Date'], df['ONI'], color='#1f77b4', linewidth=1.5, alpha=0.7, label='ONI')
ax1_twin.set_ylabel('ONI (°C)', fontsize=12, color='#1f77b4')
ax1_twin.tick_params(axis='y', labelcolor='#1f77b4')
ax1_twin.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
#periods enso
ax1_twin.fill_between(df['Date'], 0, df['ONI'], where=(df['ONI'] > 0.5), 
                       color='red', alpha=0.2, label='El Niño')
ax1_twin.fill_between(df['Date'], 0, df['ONI'], where=(df['ONI'] < -0.5), 
                       color='blue', alpha=0.2, label='La Niña')

ax1.set_title('(a) Temporal Evolution of Coffee Prices and ENSO', fontsize=13, pad=10)
ax1.grid(True, alpha=0.3)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', framealpha=0.9)

#scatter plot
ax2 = axes[1]
scatter = ax2.scatter(df['ONI'], df['Price'], alpha=0.5, c=df.index, 
                      cmap='viridis', s=20, edgecolors='none')

#regression
z = np.polyfit(df['ONI'], df['Price'], 1)
p = np.poly1d(z)
ax2.plot(df['ONI'], p(df['ONI']), "r--", linewidth=2, alpha=0.8, label='Linear fit')

r_squared = correlation**2
ax2.set_xlabel('ONI (°C)', fontsize=12)
ax2.set_ylabel('Coffee Price ($/kg)', fontsize=12)
ax2.set_title(f'(b) Correlation Analysis (r = {correlation:.3f}, R² = {r_squared:.3f})', 
              fontsize=13, pad=10)
ax2.grid(True, alpha=0.3)
ax2.legend()

#lagged correlation
ax3 = axes[2]
bars = ax3.bar(lags, correlations, color=['red' if c > 0 else 'blue' for c in correlations], 
               alpha=0.7, edgecolor='black', linewidth=0.5)
ax3.axhline(y=0, color='black', linewidth=1)
ax3.set_xlabel('Lag (months)', fontsize=12)
ax3.set_ylabel('Correlation Coefficient', fontsize=12)
ax3.set_title('(c) Cross-Correlation Analysis: Coffee Price vs ONI at Different Lags', 
              fontsize=13, pad=10)
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_xticks(range(-12, 13, 2))
ax3.axvline(x=max_lag, color='green', linestyle='--', linewidth=2, 
            label=f'Max |correlation| at lag={max_lag} months')
ax3.legend()

plt.tight_layout()
plt.savefig('figure1_enso_coffee_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
