import json, runpy, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1] / '../'
sys.path[:0] = [
    str(ROOT / 'src'),
    str(ROOT / 'support/runtime_support'),
    str(ROOT / 'tools')
]
out, tool, args = (Path(sys.argv[1]), sys.argv[2], sys.argv[3:])
import automation_search.uma_model as uma
COUNT = {'calls': 0, 'models': 0}
Original = uma.ModelFactory


class CountingFactory(Original):

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        COUNT['models'] += 1
        inner, factory = (self.predictor.predict, self)

        def counted(*pa, **pk):
            before = factory.calls
            try:
                return inner(*pa, **pk)
            finally:
                COUNT['calls'] += factory.calls - before
        self.predictor.predict = counted


uma.ModelFactory = CountingFactory
t0 = time.monotonic()
err = None
try:
    sys.argv = [tool, *args]
    runpy.run_path(str(ROOT / 'tools' / tool), run_name='__main__')
except SystemExit as e:
    if e.code not in (0, None):
        err = f'SystemExit {e.code}'
except BaseException as e:
    err = repr(e)
calls = COUNT['calls']
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(
    {
        'tool': tool,
        'args': args,
        'calls': calls,
        'seconds': round(time.monotonic() - t0, 3),
        'models_created': COUNT['models'],
        'error': err
    },
    indent=1,
    default=str
))
sys.exit(1 if err else 0)
