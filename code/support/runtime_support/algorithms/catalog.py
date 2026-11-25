from __future__ import annotations
import csv
import io
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path


def composition(species):
    value = species.strip().replace('(g)', '').replace('*', '')
    if not value:
        return ()
    tokens = re.findall('([A-Z][a-z]?)(\\d*)', value)
    if ''.join((a + b for a, b in tokens)) != value:
        raise ValueError(f'Unsupported molecular formula: {species}. Charged or complex formulas require an explicit reaction definition.')
    result = Counter()
    for element, count in tokens:
        n = int(count or 1)
        if n < 1:
            raise ValueError('Atom count must be at least 1.')
        result[element] += n
    return tuple(sorted(result.items()))


def species_list(side):
    result = []
    for term in side.split('+'):
        term = term.strip()
        if not term or term == '*':
            continue
        coefficient = re.match('^(\\d+)\\s*([A-Z].*)$', term)
        number, value = (int(coefficient[1]), coefficient[2]) if coefficient else (1, term)
        if number < 1 or number > 100:
            raise ValueError('Stoichiometric coefficients must be 1~100.')
        result.extend([composition(value)] * number)
    return result


def validate_reaction(record):
    r = dict(record)
    category = str(r.get('category', '')).upper().strip()
    if category not in ('ADSORPTION', 'DISSOCIATION'):
        raise ValueError('v1.0 supports ADSORPTION or DISSOCIATION; transfer and rearrangement are excluded.')
    equation = str(r.get('equation', '')).replace('→', '->').strip()
    if equation.count('->') != 1:
        raise ValueError('The equation requires exactly one -> arrow.')
    left, right = [x.strip() for x in equation.split('->')]
    reactants, products = (species_list(left), species_list(right))
    if not reactants or not products:
        raise ValueError('Both endpoints require molecules or atoms.')

    def total(groups):
        c = Counter()
        for group in groups:
            c.update(dict(group))
        return c
    if total(reactants) != total(products):
        raise ValueError('Element counts are not conserved across the reaction.')
    if category == 'ADSORPTION' and (reactants != products or len(reactants) != 1):
        raise ValueError('v1.0 supports nondissociative adsorption of a single molecule.')
    if category == 'DISSOCIATION' and sorted([len(reactants), len(products)]) != [1, 2]:
        raise ValueError('v1.0 cleavage supports 1 molecule to 2 fragments.')
    r.update(
        category=category,
        equation=f'{left} -> {right}',
        reactants=reactants,
        products=products,
        manual_reviewed=False,
        source=r.get('source', 'user-input')
    )
    r['note'] = 'Coordinates must be supplied separately. Geometric candidates have no stability or TS validation. Select the parent explicitly when component counts are equal.'
    return r


def parse_catalog(data, filename='reactions.txt'):
    name = filename.lower()
    if len(data) > 5 * 1024 * 1024:
        raise ValueError('Reaction list must not exceed 5 MB.')
    if name.endswith('.xlsx'):
        from zipfile import ZipFile
        with ZipFile(io.BytesIO(data)) as z:
            if sum((f.file_size for f in z.infolist())) > 25 * 1024 * 1024:
                raise ValueError('Uncompressed XLSX size exceeds the limit.')
        try:
            import openpyxl
        except ImportError as exc:
            raise ValueError('Reading XLSX requires openpyxl; CSV is also supported.') from exc
        book = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=False)
        try:
            rows = list(book.active.iter_rows(values_only=True))
            if not rows:
                raise ValueError('The worksheet is empty.')
            keys = [str(x or '').strip().lower() for x in rows[0]]
            raw = [dict(zip(keys, values)) for values in rows[1:] if any((x is not None for x in values))]
        finally:
            book.close()
    else:
        text = data.decode('utf-8-sig')
        if name.endswith('.json'):
            parsed = json.loads(text)
            raw = parsed.get('reactions', []) if isinstance(parsed, dict) else parsed
        elif name.endswith(('.csv', '.tsv')):
            raw = list(csv.DictReader(
                io.StringIO(text),
                delimiter='\t' if name.endswith('.tsv') else ','
            ))
        else:
            raw, category = ([], None)
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('[') and line.endswith(']'):
                    category = line[1:-1].upper()
                else:
                    raw.append({'category': category, 'equation': line})
    if not isinstance(raw, list) or not raw or len(raw) > 1000:
        raise ValueError('Supply 1~1000 reactions.')
    result = []
    for row, record in enumerate(raw, 1):
        try:
            result.append(validate_reaction(record))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError(f'Reaction list {row} row: {exc}') from exc
    return result


class Catalog:

    def __init__(self, database, seed):
        self.path = Path(database)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS reactions (id INTEGER PRIMARY KEY, category TEXT NOT NULL, equation TEXT NOT NULL, data TEXT NOT NULL, UNIQUE(category,equation))')
        self.add(json.loads(Path(seed).read_text(encoding='utf-8'))['reactions'])

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def add(self, records):
        checked = [validate_reaction(r) for r in records]
        with self.connect() as db:
            before = db.total_changes
            for r in checked:
                db.execute(
                    'INSERT OR IGNORE INTO reactions(category,equation,data) VALUES(?,?,?)',
                    (r['category'], r['equation'], json.dumps(r, ensure_ascii=False))
                )
            inserted = db.total_changes - before
        return {'added': inserted, 'already_present': len(checked) - inserted}

    def list(self):
        with self.connect() as db:
            return [
                {**json.loads(data), 'id': i} for i,
                data in db.execute('SELECT id,data FROM reactions ORDER BY id')
            ]

    def get(self, reaction_id):
        with self.connect() as db:
            row = db.execute(
                'SELECT id,data FROM reactions WHERE id=?',
                (int(reaction_id),)
            ).fetchone()
        if row is None:
            raise ValueError('Reaction was not found in the list.')
        return {**json.loads(row[1]), 'id': row[0]}
