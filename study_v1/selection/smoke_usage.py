import json,glob
for d in sorted(glob.glob('logs/results_batch_*smoke*')):
    for f in sorted(glob.glob(d+'/*.json')):
        r=json.load(open(f)); us=[c.get('usage') or {} for c in r['criteria_results']]
        print(d.split('batch_')[1][:30], r['paper_id'], r['paper'].get('rendered_chars'), [ (u.get('prompt_tokens') or u.get('input_tokens'), u.get('completion_tokens') or u.get('output_tokens')) for u in us][:6], r.get('cost') or r.get('generation',{}).get('cost') if isinstance(r.get('generation'),dict) else r.get('cost'))
