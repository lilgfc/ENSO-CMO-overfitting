import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns 

# --- 1. DATA LOADING ---
# FIX: Use the full Kaggle paths you used before
enso = pd.read_csv('/kaggle/input/cafeyenso/ENSO.csv')
coffee = pd.read_csv('/kaggle/input/cafeyenso/CMOdata.txt', sep='\t', header=1)

# --- 2. DATA PARSING & CLEANING ---
enso['Date'] = pd.to_datetime(enso['Date'], format='%m/%d/%Y')
coffee['Date'] = pd.to_datetime(coffee['Year'], format='%YM%m') 
coffee = coffee.rename(columns={'($/kg)': 'Price'})

# --- 3. MERGE DATASETS ---
df = pd.merge(coffee, enso[['Date', 'ONI']], on='Date', how='inner')

# --- 4. CREATE ASYMMETRIC DUMMY VARIABLES ---
df['ElNino'] = (df['ONI'] > 0.5).astype(int)
df['LaNina'] = (df['ONI'] < -0.5).astype(int)

# --- 5. CREATE LAGGED DUMMIES ---
lag_months = 13
df['ElNino_Lagged'] = df['ElNino'].shift(lag_months)
df['LaNina_Lagged'] = df['LaNina'].shift(lag_months)

# --- 6. RUN THE LINEAR REGRESSION MODEL ---
df_model = df.dropna(subset=['Price', 'ElNino_Lagged', 'LaNina_Lagged'])
y = df_model['Price']
X = df_model[['ElNino_Lagged', 'LaNina_Lagged']]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# --- 7. PRINT RESULTS ---
print(f"--- Regression Results with {lag_months}-Month Lag ---")
print(model.summary())

# --- 8. NEW VISUALIZATION ---
fig, axes = plt.subplots(3, 1, figsize=(14, 14))
fig.suptitle('Asymmetric ENSO Impact on Arabica Coffee Prices', 
             fontsize=16, fontweight='bold')

# --- (a) Time series comparison ---
ax1 = axes[0]
ax1_twin = ax1.twinx()

line1 = ax1.plot(df['Date'], df['Price'], color='#8B4513', linewidth=1.5, label='Coffee Price')
ax1.set_ylabel('Coffee Price ($/kg)', fontsize=12, color='#8B4513')
ax1.tick_params(axis='y', labelcolor='#8B4513')

line2 = ax1_twin.plot(df['Date'], df['ONI'], color='#1f77b4', linewidth=1.5, alpha=0.7, label='ONI')
ax1_twin.set_ylabel('ONI (°C)', fontsize=12, color='#1f77b4')
ax1_twin.tick_params(axis='y', labelcolor='#1f77b4')
ax1_twin.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

ax1_twin.fill_between(df['Date'], 0, df['ONI'], where=(df['ONI'] > 0.5), 
                      color='red', alpha=0.2, label='El Niño')
ax1_twin.fill_between(df['Date'], 0, df['ONI'], where=(df['ONI'] < -0.5), 
                      color='blue', alpha=0.2, label='La Niña')

ax1.set_title('(a) Temporal Evolution of Coffee Prices and ENSO', fontsize=13, pad=10)
ax1.grid(True, alpha=0.3)
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', framealpha=0.9)

# --- (b) Regression Coefficient Plot ---
ax2 = axes[1]
coefs = model.params.drop('const')
conf_intervals = model.conf_int().drop('const')
errors = coefs - conf_intervals[0] 
colors = ['red' if coefs.index[i] == 'ElNino_Lagged' else 'blue' for i in range(len(coefs))]

coefs.plot(kind='bar', ax=ax2, color=colors, alpha=0.7,
           yerr=errors, capsize=5, edgecolor='black')

ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax2.set_ylabel('Price Impact ($/kg)', fontsize=12)
ax2.set_title(f'(b) Price Impact 13 Months After ENSO Phase', fontsize=13, pad=10)
ax2.set_xticklabels(['El Niño', 'La Niña'], rotation=0)
ax2.grid(True, alpha=0.3, axis='y')

p_values = model.pvalues.drop('const')
if p_values['LaNina_Lagged'] < 0.05:
    ax2.text(1, coefs['LaNina_Lagged'] + 0.05, '*', ha='center', va='bottom', fontsize=16, color='black')

# --- (c) Price Distribution by Phase ---
ax3 = axes[2]

def get_phase(row):
    if row['ElNino_Lagged'] == 1:
        return 'El Niño'
    elif row['LaNina_Lagged'] == 1:
        return 'La Niña'
    else:
        return 'Neutral'
df_model['Phase_Lagged'] = df_model.apply(get_phase, axis=1)

sns.boxplot(x='Phase_Lagged', y='Price', data=df_model, ax=ax3,
            palette={'Neutral': 'grey', 'El Niño': 'red', 'La Niña': 'blue'},
            order=['El Niño', 'Neutral', 'La Niña'])

ax3.set_xlabel('ENSO Phase (Lagged 13 Months)', fontsize=12)
ax3.set_ylabel('Coffee Price ($/kg)', fontsize=12)
ax3.set_title('(c) Coffee Price Distribution by Lagged ENSO Phase', fontsize=13, pad=10)
ax3.grid(True, alpha=0.3, axis='y')

# --- Save Figure ---
plt.tight_layout()
plt.savefig('figure2_asymmetric_enso_analysis.png', dpi=300, bbox_inches='tight')
