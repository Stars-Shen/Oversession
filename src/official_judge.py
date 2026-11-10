"""Score LongMemEval predictions with the official snapshot, rubric and generation parameters."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from openai import OpenAI,DefaultHttpxClient
from api import ROOT,API,atomic_json,digest
from longmem import judge_prompt

MODEL='gpt-4o-2024-08-06'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run')
    p.add_argument('--workers',type=int,default=8)
    args=p.parse_args()
    folder=ROOT/'runs'/args.run
    predictions=[json.loads(line) for line in (folder/'predictions.jsonl').read_text().splitlines()]
    references={e['question_id']:e for e in json.loads((ROOT/'data/longmemeval_s_cleaned.json').read_text())}
    prompt_for=judge_prompt()
    client=OpenAI(api_key=API().key,max_retries=5,
                  http_client=DefaultHttpxClient(proxy=os.environ.get('MEMORY_PROXY','http://127.0.0.1:41227'),trust_env=False))
    cache=ROOT/'runs/official_judge_cache';cache.mkdir(exist_ok=True)
    outpath=folder/'official_gpt4o.jsonl'
    existing=[json.loads(x) for x in outpath.read_text().splitlines()] if outpath.exists() else []
    done={(r['qid'],r['method']) for r in existing}

    def judge(row):
        ref=references[row['qid']]
        prompt=prompt_for(ref['question_type'],ref['question'],ref['answer'],row['hypothesis'],abstention='_abs' in row['qid'])
        body=dict(model=MODEL,messages=[dict(role='user',content=prompt)],n=1,temperature=0,max_tokens=10)
        signature=digest(body)
        path=cache/(signature+'.json')
        if path.exists():result=json.loads(path.read_text())
        else:
            completion=client.chat.completions.create(**body)
            text=completion.choices[0].message.content.strip()
            if text.lower().rstrip('.') not in ('yes','no'):raise ValueError('Invalid judge result: '+text)
            result=dict(text=text,model=completion.model,usage=completion.usage.model_dump(),request_hash=signature)
            atomic_json(path,result)
        return dict(qid=row['qid'],method=row['method'],question_type=ref['question_type'],
                    correct='yes' in result['text'].lower(),judge_model=MODEL,api_hash=signature)

    with ThreadPoolExecutor(max_workers=args.workers) as pool,outpath.open('a') as out:
        jobs=[pool.submit(judge,row) for row in predictions if (row['qid'],row['method']) not in done]
        for job in as_completed(jobs):
            r=job.result();existing.append(r);out.write(json.dumps(r)+'\n');out.flush()
            print(len(existing),'/',len(predictions),r['qid'],r['method'],r['correct'],flush=True)
    summary={}
    for method in sorted({r['method'] for r in existing}):
        rows=[r for r in existing if r['method']==method]
        summary[method]=dict(n=len(rows),accuracy=sum(r['correct'] for r in rows)/len(rows),
                            by_type={kind:dict(n=sum(r['question_type']==kind for r in rows),
                                               correct=sum(r['correct'] for r in rows if r['question_type']==kind))
                                     for kind in sorted({r['question_type'] for r in rows})})
    atomic_json(folder/'official_gpt4o_summary.json',dict(judge_model=MODEL,results=summary))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
