import pandas as pd, json, os
from collections import Counter

os.makedirs('data/processed', exist_ok=True)

JB_FILE = 'data/raw/jetbrains_2025/Developer Ecosystem Survey 2025_ Raw Data Sharing 2/developer_ecosystem_2025_external.csv'

print('Reading JetBrains 2025 CSV...')
df = pd.read_csv(JB_FILE, low_memory=False)
print(f'Shape: {df.shape}')

# Job role columns are one-hot encoded: job_role::Developer / Programmer / Software Engineer
role_cols = [c for c in df.columns if c.startswith('job_role::')]
print(f'Role columns: {role_cols}')

# Country column
country_col = next((c for c in df.columns if 'country' in c.lower()), None)
print(f'Country col: {country_col}')

# Language columns
lang_cols = [c for c in df.columns if c.startswith('proglang::')]

chunks = []

for role_col in role_cols:
    role_name = role_col.replace('job_role::', '').strip()
    # Filter respondents who selected this role
    group = df[df[role_col].notna()]
    if len(group) < 30:
        continue
    n = len(group)

    # Top countries
    countries = {}
    if country_col:
        countries = group[country_col].value_counts().head(8).to_dict()

    # Top languages used
    lang_usage = []
    for lc in lang_cols:
        lang_name = lc.replace('proglang::', '').strip()
        count = group[lc].notna().sum()
        pct = round(count / n * 100, 1)
        if pct > 5:
            lang_usage.append((lang_name, pct))
    lang_usage.sort(key=lambda x: -x[1])

    content = f'''Source: JetBrains Developer Ecosystem Survey 2025
Organization: JetBrains (devecosystem-2025.jetbrains.com)
License: Non-commercial use with attribution
Coverage: 24,534 developers from 194 countries
Role: {role_name}
Respondents in this role: {n:,}

TOP LANGUAGES USED BY {role_name.upper()} IN 2025:
{', '.join(f'{k} ({v}%)' for k,v in lang_usage[:8]) if lang_usage else 'See report for details'}

TOP COUNTRIES: {', '.join(f'{c}: {v}' for c,v in list(countries.items())[:5])}

Note: JetBrains 2025 survey covers 194 countries including strong Southeast Asia representation.
Data collected April-June 2025. Responses weighted by geography and employment status.'''

    chunks.append({
        'id': f'jb2025_{role_name.lower().replace(" ", "_").replace("/", "_").replace(",", "")[:50]}',
        'source': 'jetbrains_2025',
        'dev_type': role_name,
        'content': content
    })
    print(f'  {role_name}: {n} respondents')

with open('data/processed/jetbrains_chunks.json', 'w') as f:
    json.dump(chunks, f, indent=2)
print(f'Done: {len(chunks)} JetBrains 2025 chunks')