"""Train, score and evaluate either portable algorithm with user CSV files."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from ..config import ALGORITHM_SEED,ALGORITHM2_MIN_RECALL
from ..evaluation import evaluate_algorithm_scores
from .shared import META,read_csv,eligibility,eligible_training,disjoint,save_json,result_frame


def feature_columns(number, state=None):
    """Return only whitelisted financial inputs, never labels or future metadata."""
    if number==1:
        from .algorithm_1 import core
        return core.BASE+core.ADDED if state is None else sorted({c for n in state['weights'] for c in state['experts'][n]['columns']})
    from .algorithm_2 import core
    return core.COLS if state is None else core.required_inputs(state['columns'])


def score(number,state,d):
    """Apply date gates and the original algorithm-1 missing-data gate."""
    valid,status=eligibility(d);p=np.full(len(d),np.nan)
    if valid.any():
        subset=d[valid]
        if number==1:
            from .algorithm_1.core import predict_frame
            result=predict_frame(state,subset)
            p[valid.to_numpy()]=result.risk_score_0_100.to_numpy()/100
            status.loc[result.index[result.status.ne('exploratory_score')]]='insufficient_financial_inputs'
        else:
            from .algorithm_2.core import predict_system
            p[valid.to_numpy()]=predict_system(state,subset)
    return p,status


def run(number):
    """CLI entry point. All observed data/model files are created by the caller."""
    parser=argparse.ArgumentParser(description=f'Algorithm {number}: train/predict/evaluate user data')
    commands=parser.add_subparsers(dest='command',required=True)
    template=commands.add_parser('template');template.add_argument('--output',required=True);template.add_argument('--target-column',default='target')
    train=commands.add_parser('train');train.add_argument('--train',required=True);train.add_argument('--validation',required=True)
    train.add_argument('--model',required=True);train.add_argument('--target-column',required=True)
    train.add_argument('--horizon-months',type=int,choices=[12,24],required=True);train.add_argument('--seed',type=int,default=ALGORITHM_SEED)
    if number==2:train.add_argument('--minimum-recall',type=float,default=ALGORITHM2_MIN_RECALL)
    for name in ['predict','evaluate']:
        p=commands.add_parser(name);p.add_argument('--model',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='template':
        cols=META+(['pair_id'] if number==1 else [])+[args.target_column]+feature_columns(number)
        target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(columns=cols).to_csv(target,index=False);print(str(target));return
    if args.command=='train':
        cols=feature_columns(number)
        rawtrain=read_csv(args.train,cols,args.target_column);rawval=read_csv(args.validation,cols,args.target_column)
        if number==1:
            for d in [rawtrain,rawval]:
                if 'pair_id' not in d or d.pair_id.isna().any() or d.corp_code.duplicated().any():
                    raise ValueError('Algorithm 1 needs one row per company and nonempty pair_id')
        disjoint(rawtrain,rawval,paired=number==1)
        tr,train_audit=eligible_training(rawtrain,paired=number==1);va,val_audit=eligible_training(rawval,paired=number==1)
        if number==1:
            from .algorithm_1.core import fit_system,predict_system,sweet_threshold
            if tr.pair_id.nunique()<4:raise ValueError('At least four complete training pairs are required')
            state,_,_=fit_system(tr,seed=args.seed)
            choice=sweet_threshold(va.y,predict_system(state,va));state['threshold']=choice['threshold']
            state.update(algorithm=1,selection=choice)
        else:
            from .algorithm_2.core import fit_system
            state=fit_system(tr,va,minimum_recall=args.minimum_recall,seed=args.seed)
        state.update(target_column=args.target_column,horizon_months=args.horizon_months,
            label_definition='Caller-supplied binary target; TODO verify actual future-event definition and censoring upstream',
            seed=args.seed,training_companies=sorted(tr.corp_code.unique()),validation_companies=sorted(va.corp_code.unique()),
            input_eligibility=dict(train=train_audit,validation=val_audit),source_PIT_verified=False,
            independent_future_validation=False,pretrained_research_coefficients_included=False)
        save_json(args.model,state);print(json.dumps(dict(model=args.model,algorithm=number,threshold=state['threshold'],input_eligibility=state['input_eligibility']),ensure_ascii=False));return
    state=json.loads(Path(args.model).read_text(encoding='utf-8'))
    if state.get('algorithm')!=number:raise ValueError('Model belongs to another algorithm')
    target=state['target_column'] if args.command=='evaluate' else None
    d=read_csv(args.input,feature_columns(number,state),target)
    if args.command=='evaluate':
        used=set(state['training_companies'])|set(state['validation_companies'])
        if used & set(d.corp_code):raise ValueError('Evaluation firms overlap TRAIN/VALIDATION; use predict for ordinary inference')
    p,status=score(number,state,d)
    if args.command=='predict':
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        result_frame(d,p,state['threshold'],status).to_csv(path,index=False)
    else:
        ok=status.eq('eligible')
        if not ok.any():raise ValueError('No evaluable rows after information-date and coverage gates')
        metrics=evaluate_algorithm_scores(d.loc[ok,'y'],p[ok.to_numpy()],state['threshold'])
        metrics.update(input_rows=len(d),evaluated_rows=int(ok.sum()),excluded_rows=int((~ok).sum()),algorithm=number)
        save_json(args.output,metrics)
    print(str(args.output))
