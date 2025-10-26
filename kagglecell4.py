# Load data
enso = pd.read_csv('/kaggle/input/cafeyenso/ENSO.csv')
coffee = pd.read_csv('/kaggle/input/cafeyenso/CMOdata.txt', sep='\t', header=1)
# Parsing by date
enso['Date'] = pd.to_datetime(enso['Date'], format='%m/%d/%Y')
coffee['Date'] = pd.to_datetime(coffee['Year'], format='%YM%m') 

coffee = coffee.rename(columns={'($/kg)': 'Price'})

# Sort by date
enso = enso.sort_values('Date').reset_index(drop=True)
coffee = coffee.sort_values('Date').reset_index(drop=True)

# missing values
print("ENSO missing values:")
print(enso.isnull().sum())
print("\nCoffee missing values:")
print(coffee.isnull().sum()) # Now shows Price, Year, and Date columns

print(f"\nENSO date range: {enso['Date'].min()} to {enso['Date'].max()}")
print(f"Coffee date range: {coffee['Date'].min()} to {coffee['Date'].max()}")
df = pd.merge(coffee, enso[['Date', 'ONI']], on='Date', how='inner')

print(f"Merged dataset shape: {df.shape}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Total observations: {len(df)}")
print(df[['Price', 'ONI']].describe())
