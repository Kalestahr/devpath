from dotenv import load_dotenv
load_dotenv()

import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import agent, Deps
from monitoring import init_tables, log_feedback, log_query_time, get_feedback_log, get_query_times

app = FastAPI(title='DevPath API', version='1.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

init_tables()

class AskRequest(BaseModel):
    question: str
    skills: list[str] = []
    target_role: str = ''
    region: str = ''

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: int

@app.post('/ask')
async def ask(req: AskRequest):
    deps = Deps(skills=req.skills, target_role=req.target_role, region=req.region)
    start = time.time()
    result = await agent.run(req.question, deps=deps)
    log_query_time(req.question, round(time.time() - start, 2))
    return {'answer': result.output, 'status': 'ok'}

@app.post('/feedback')
async def feedback(req: FeedbackRequest):
    log_feedback(req.question, req.rating)
    return {'status': 'ok'}

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.get('/stats')
def stats():
    fb = get_feedback_log()
    times = get_query_times()
    pos = sum(1 for f in fb if f['rating'] == 1)
    neg = sum(1 for f in fb if f['rating'] == -1)
    avg_time = round(sum(times) / len(times), 2) if times else 0
    return {'total_queries': len(times), 'positive': pos, 'negative': neg, 'avg_response_time': avg_time}