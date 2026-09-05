import pandas as pd, json, os
from collections import Counter

os.makedirs('data/processed', exist_ok=True)

DEV_TYPE_MAP = {
    'Academic researcher': 'Academic researcher',
    'Cloud infrastructure engineer': 'Cloud infrastructure engineer',
    'Data or business analyst': 'Data or business analyst',
    'Data scientist or machine learning specialist': 'Data scientist or machine learning specialist',
    'Database administrator': 'Database administrator',
    'Designer': 'Designer',
    'DevOps specialist': 'DevOps specialist',
    'Developer, QA or test': 'Developer, QA or test',
    'Developer, back-end': 'Developer, back-end',
    'Developer, desktop or enterprise applications': 'Developer, desktop or enterprise applications',
    'Developer, embedded applications or devices': 'Developer, embedded applications or devices',
    'Developer, front-end': 'Developer, front-end',
    'Developer, full-stack': 'Developer, full-stack',
    'Developer, game or graphics': 'Developer, game or graphics',
    'Developer, mobile': 'Developer, mobile',
    'Engineer, site reliability': 'Engineer, site reliability',
    'Engineering manager': 'Engineering manager',
    'Hardware Engineer': 'Hardware Engineer',
    'Marketing or sales professional': 'Marketing or sales professional',
    'Product manager': 'Product manager',
    'Project manager': 'Project manager',
    'Research & Development role': 'Research and Development role',
    'Scientist': 'Scientist',
    'Security professional': 'Security professional',
    'Senior Executive (C-Suite, VP, etc.)': 'Senior Executive',
    'Student': 'Student',
    'System administrator': 'System administrator',
    'Teacher or educator': 'Teacher or educator',
    'Technical writer': 'Technical writer',
    'Other (please specify):': 'Other',
}

def top_items(series, n=10):
    counts = Counter()
    for val in series.dropna():
        for item in str(val).split(';'):
            if item.strip(): counts[item.strip()] += 1
    total = max(len(series.dropna()), 1)
    return [(k, round(v/total*100, 1)) for k, v in counts.most_common(n)]

print('Reading SO 2025 Survey CSV (134MB - takes a moment)...')
df = pd.read_csv('data/raw/survey_results_2025.csv')
print(f'Shape: {df.shape}')
print(f'Columns sample: {list(df.columns)[:10]}')

# Detect column names
role_col    = 'DevType'   if 'DevType'   in df.columns else 'dev_type'
country_col = 'Country'   if 'Country'   in df.columns else 'country'
lang_col    = 'LanguageHaveWorkedWith' if 'LanguageHaveWorkedWith' in df.columns else 'r_used'
exp_col     = 'YearsCodePro' if 'YearsCodePro' in df.columns else 'years_code_pro'
db_col      = 'DatabaseHaveWorkedWith' if 'DatabaseHaveWorkedWith' in df.columns else None
tools_col   = 'MiscTechHaveWorkedWith' if 'MiscTechHaveWorkedWith' in df.columns else None
ai_col      = 'AISearchHaveWorkedWith' if 'AISearchHaveWorkedWith' in df.columns else None

print(f'Role col: {role_col}, Lang col: {lang_col}')

df[role_col] = df[role_col].astype(str).str.split(';')
df = df.explode(role_col).dropna(subset=[role_col])
df[role_col] = df[role_col].str.strip()
df = df[df[role_col].str.lower() != 'nan']

chunks = []
for role, group in df.groupby(role_col):
    if len(group) < 50: continue
    n = len(group)
    role_name = DEV_TYPE_MAP.get(role, role)
    countries = group[country_col].value_counts().head(8).to_dict() if country_col in df.columns else {}
    exp_med = pd.to_numeric(group[exp_col], errors='coerce').median() if exp_col in df.columns else None
    langs = top_items(group[lang_col]) if lang_col in df.columns else []
    dbs = top_items(group[db_col]) if db_col and db_col in df.columns else []
    tools = top_items(group[tools_col]) if tools_col and tools_col in df.columns else []
    ai = top_items(group[ai_col]) if ai_col and ai_col in df.columns else []

    content = f'''Source: Stack Overflow Developer Survey 2025
Organization: Stack Overflow (survey.stackoverflow.co/2025/)
License: Open Database License (ODbL)
Role: {role_name}
Global respondents: {n:,} developers from 177 countries

PROGRAMMING LANGUAGES USED BY {role_name.upper()} IN 2025:
{', '.join(f'{k} ({v}%)' for k,v in langs[:8]) if langs else 'See survey for details'}

DATABASES USED:
{', '.join(f'{k} ({v}%)' for k,v in dbs[:6]) if dbs else 'See survey for details'}

AI TOOLS USED IN 2025:
{', '.join(f'{k} ({v}%)' for k,v in ai[:5]) if ai else 'See survey for details'}

EXPERIENCE: Median {f'{exp_med:.0f} years' if exp_med and not pd.isna(exp_med) else 'not reported'} professional coding

TOP COUNTRIES: {', '.join(f'{c}: {v}' for c,v in list(countries.items())[:5])}'''

    chunks.append({
        'id': f'so2025_{role_name.lower().replace(" ", "_").replace(",", "").replace("/", "_")[:50]}',
        'source': 'stackoverflow_2025',
        'dev_type': role_name,
        'content': content
    })

with open('data/processed/so2025_chunks.json', 'w') as f:
    json.dump(chunks, f, indent=2)
print(f'Done: {len(chunks)} SO 2025 role chunks')
for c in chunks[:3]:
    print(f'  {c["dev_type"]}')