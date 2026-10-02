"""Read-only exact live Rule B/default AST audit; fails on old scanner snapshots."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import time

RULE_AST = "lambda snr, s: CELL_NAMES.index('2000_medium') if snr <= 10.0 else CELL_NAMES.index('4000_medium')"
DEFAULTS = ['fixed_2000_medium','fixed_4000_medium','fixed_8000_high','static_exp014_joint','proposed_snr_adaptive_v2']

def verify_scanner(source):
    source = Path(source);tree = ast.parse(source.read_text())
    def assignment(name):
        return next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
    try:
        policies = assignment('all_policies')
        v2 = next(v for k,v in zip(policies.keys,policies.values) if k.value=='proposed_snr_adaptive_v2')
        defaults = ast.literal_eval(assignment('default_policy_names'))
    except StopIteration as exc:
        raise RuntimeError('Live scanner lacks frozen v2/default definitions; stop without modifying it') from exc
    assert ast.unparse(v2)==RULE_AST, 'Rule B AST changed'
    assert defaults==DEFAULTS, 'Default policy list changed'
    arg = next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
               and n.func.attr=='add_argument' and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='--policies')
    assert ast.literal_eval(next(k.value for k in arg.keywords if k.arg=='default')) is None
    assert any(isinstance(n,ast.If) and ast.unparse(n.test)=='args.policies is None'
               and ast.unparse(n.body[0])=='selected = default_policy_names' for n in ast.walk(tree))
    cells = [f'{b}_{t}' for b in (2000,4000,8000) for t in ('low','medium','high')]
    fn = eval(compile(ast.Expression(v2),str(source),'eval'),{'CELL_NAMES':cells})
    checks = {str(s):cells[fn(s,{})] for s in [-5,0,5,9.999999,10,10.000001,20]}
    assert all(v==('2000_medium' if float(k)<=10 else '4000_medium') for k,v in checks.items())
    return {'status':'PASS','passed':True,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'rule_b_ast':ast.unparse(v2),'defaults':defaults,'policies_cli_default':None,'boundary_checks':checks}

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--repo-root',type=Path,required=True);a=p.parse_args()
    print(json.dumps(verify_scanner(a.repo_root/'code/vqa_semcom_v0/experiments/rgb_channel_snr_scan/run_snr_scan.py'),indent=2))
