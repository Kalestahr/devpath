from dotenv import load_dotenv
load_dotenv()
 
import dlt, json, duckdb
from pathlib import Path
 
def load_json(path):
    if Path(path).exists():
        with open(path) as f: return json.load(f)
    print(f'Warning: {path} not found')
    return []
 
@dlt.resource(name='career_chunks', write_disposition='replace')
def all_chunks():
    for path in ['data/processed/so_chunks.json',
                 'data/processed/onet_chunks.json',
                 'data/processed/wef_chunks.json',
                 'data/processed/so2025_chunks.json',
                 'data/processed/jetbrains_chunks.json']:
        chunks = load_json(path)
        print(f'  {path}: {len(chunks)} chunks')
        yield from chunks
 
pipeline = dlt.pipeline(
    pipeline_name='devpath_pipeline',
    destination='duckdb',
    dataset_name='devpath',
)
print('Running dlt pipeline...')
load_info = pipeline.run(all_chunks())
print(load_info)
 
conn = duckdb.connect('devpath_pipeline.duckdb')
count = conn.execute('SELECT COUNT(*) FROM devpath.career_chunks').fetchone()[0]
sources = conn.execute('SELECT source, COUNT(*) n FROM devpath.career_chunks GROUP BY source ORDER BY n DESC').fetchall()
print(f'Total chunks in DuckDB: {count}')
for s in sources: print(f'  {s[0]}: {s[1]}')
conn.close()
