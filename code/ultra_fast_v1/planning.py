"""Deterministic endpoint planning; no calculator/model calls.

Structural annotation lazily imports ASE/numpy. All pairs and eligible scan seeds
are retained; priorities and broken-scan seed order change only execution order.
"""
from __future__ import annotations

import copy
import json
import logging
import math
from pathlib import Path
from chavis.runtime.adsorption_settings import parse_azimuth_step


COVALENT_RADII_A = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66}
RADII_SOURCE = (
    "https://gitlab.com/ase/ase/-/raw/"
    "6cb8784ac1056b7b897822ff7b763a323d92a543/ase/data/__init__.py"
)
RADII_REFERENCE = "Cordero et al., Dalton Trans. (2008), DOI:10.1039/B801115J"
DEFAULTS = {
    "search_scope": "first_success",
    "order_mode": "demote_product_detached",
    "adsorption_azimuth_step_deg": "auto",
    "form_basis": "final_relaxed_binding_form",
    "staying_fragment_max_displacement_A": 1.0,
    "max_deferrals_per_pair": 1,
    "scan_jump_window_points": 3,
    "scan_jump_defer_eV": 0.8,
    "bond_prior_scale": 1.5,
    "seeds_per_visit": 3,
    "resumes_per_visit": 3,
    "dimer_screen_steps": 300,
    "dimer_max_steps": 1500,
    "first_validated_min_share": 0.01,
    "prescan_all_pairs": False,
    "lazy_products": False,
    "light_acceptance": False,
    "neb_after_failures": 3,          # early NEB after this many failed scan candidates on a pair
    "neb_after_failures_broken": 1,   # ... and after this many when the pair's scan has no interior peak
}


def _number(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name}: a finite number is required")
    if positive and value <= 0:
        raise ValueError(f"{name}: must be positive")
    return value


def load_settings(path=None, overrides=None):
    """Load separate ultra settings, rejecting unknown keys and changed common gates."""
    values = dict(DEFAULTS)
    for supplied in (() if path is None else (json.loads(Path(path).read_text(encoding="utf-8-sig")),)) + (() if overrides is None else (overrides,)):
        if not isinstance(supplied, dict):
            raise ValueError("Ultra settings must be a JSON object")
        unknown = set(supplied) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown ultra settings: {sorted(unknown)}")
        values.update(supplied)
    if values["search_scope"] not in ("first_success", "all_pairs"):
        raise ValueError("search_scope must be first_success or all_pairs")
    if values["order_mode"] not in ("demote_product_detached", "demote_only", "three_tier"):
        raise ValueError("order_mode must be demote_product_detached, demote_only, or three_tier")
    for key in ("staying_fragment_max_displacement_A", "scan_jump_defer_eV", "bond_prior_scale"):
        _number(values[key], key, positive=True)
    values["adsorption_azimuth_step_deg"] = parse_azimuth_step(values["adsorption_azimuth_step_deg"])
    for key in ("scan_jump_window_points", "seeds_per_visit", "resumes_per_visit", "dimer_screen_steps", "dimer_max_steps"):
        if type(values[key]) is not int or values[key] < 1:
            raise ValueError(f"{key}: a positive integer is required")
    if values["dimer_screen_steps"] != 300 or values["dimer_max_steps"] != 1500:
        raise ValueError("Dimer screen 300 and cumulative maximum 1500 are fixed")
    if type(values["max_deferrals_per_pair"]) is not int or values["max_deferrals_per_pair"] != 1:
        raise ValueError("Ultra-fast scheduling permits exactly one deferral per pair")
    if values["form_basis"] != DEFAULTS["form_basis"]:
        raise ValueError("Ultra-fast scheduling groups final relaxed binding forms")
    if type(values["first_validated_min_share"]) not in (int, float) or values["first_validated_min_share"] != 0.01:
        raise ValueError("Common first-validated minimum share must remain 0.01")
    for key in ("neb_after_failures", "neb_after_failures_broken"):
        if type(values[key]) is not int or values[key] < 0:
            raise ValueError(f"{key}: a non-negative integer is required (0 = off)")
    for key in ("prescan_all_pairs", "lazy_products", "light_acceptance"):
        if values[key] is not False:
            raise ValueError(f"{key} is disabled in ultra-fast scheduling")
    return values


def _warn(message, log=None):
    if log is None:
        logging.getLogger(__name__).warning(message)
    else:
        log(message)


def _read_jsonl(path):
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        rows.append(row)
    return rows


def _final_form(row):
    """Equivalent to minima.form for complete final labels, without importing numpy."""
    labels = row.get("labels")
    if not isinstance(labels, dict):
        return None
    outcome = row.get("outcome")
    if not isinstance(outcome, str) or not outcome:
        dissociation, desorption = labels.get("dissociation"), labels.get("desorption")
        if not isinstance(dissociation, dict) or not isinstance(desorption, dict):
            return None
        if type(dissociation.get("flag")) is not bool or type(desorption.get("flag")) is not bool:
            return None
        outcome = "dissociated" if dissociation["flag"] else "floating" if desorption["flag"] else "bound"
    binding = labels.get("binding")
    if not isinstance(binding, dict) or not isinstance(binding.get("contacts"), list):
        return None
    contacts = binding["contacts"]
    if any(not isinstance(c, dict) or not isinstance(c.get("element"), str) or not c["element"] for c in contacts):
        return None
    if contacts:
        base = "+".join(sorted(f"{c['element']}@{c.get('site')}" for c in contacts))
    else:
        base = binding.get("mode")
        if not isinstance(base, str) or not base:
            return None
    return outcome, base if outcome == "bound" else f"{outcome}:{base}"


def _unknown_form(identifier, source="missing", row=None):
    return {
        "minimum_representative": identifier, "outcome": None, "form": None,
        "group_key": f"unknown:{identifier}", "source": source,
        "frozen_form": (row or {}).get("frozen_form"), "rotamer_of": None,
        "warning": "Missing final binding-form labels; retained in its own reactant group",
    }


def load_form_metadata(species_dir, reaction_id, log=None):
    """Resolve pair.paired_is from final refined rows, then derived-IS records.

    Frozen form and rotamer metadata are provenance only. A matching refined row
    wins over derived records, including when its labels are missing: silently
    borrowing another structure's form would misrepresent the actual pair IS.
    """
    species_dir = Path(species_dir)
    minima_path = species_dir / "minima.json"
    minima = {}
    if minima_path.is_file():
        data = json.loads(minima_path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict) or not isinstance(data.get("minima"), list):
            raise ValueError("minima.json must contain a minima list")
        for row in data["minima"]:
            if not isinstance(row, dict) or "representative" not in row:
                raise ValueError("Malformed minimum provenance")
            minima[str(row["representative"])] = row
    result = {}
    sources = (("refined", species_dir / "refined.jsonl", "minimum_representative"),
               ("derived", species_dir / f"is_from_fs_{reaction_id}.jsonl", "paired_is"))
    for source, path, id_key in sources:
        for row in _read_jsonl(path):
            if id_key not in row or row[id_key] is None:
                if source == "refined":
                    raise ValueError(f"{path}: missing {id_key}")
                continue  # unsuccessful derivations have no paired reactant
            identifier = str(row[id_key])
            if identifier in result:
                if source == "refined":
                    raise ValueError(f"Duplicate refined reactant ID: {identifier}")
                continue
            final = _final_form(row)
            if final is None:
                item = _unknown_form(identifier, source, row)
                _warn(f"{identifier}: {item['warning']}", log)
            else:
                outcome, form = final
                item = {
                    "minimum_representative": identifier, "outcome": outcome,
                    "form": form, "group_key": json.dumps([outcome, form], separators=(",", ":")),
                    "source": source, "frozen_form": row.get("frozen_form"),
                    "rotamer_of": minima.get(identifier, {}).get("rotamer_of"), "warning": None,
                }
            item["source_path"] = str(path)
            result[identifier] = item
    return result


def structure_grade(is_min_ratio, fs_fragment_min_ratios, staying_max_displacement_A, settings=None):
    """Universal endpoint-only grade; all boundaries are strict as specified."""
    from chavis.judgment.structure_labels import BONDED_SCALE, DESORPTION_SCALE
    settings = load_settings(overrides=settings)
    _number(is_min_ratio, 'IS metal ratio')
    _number(staying_max_displacement_A, 'staying fragment displacement')
    if is_min_ratio < 0 or staying_max_displacement_A < 0 or not fs_fragment_min_ratios:
        raise ValueError('Nonnegative displacement and at least one FS fragment required')
    for ratio in fs_fragment_min_ratios:
        _number(ratio, 'FS fragment metal ratio')
        if ratio < 0:
            raise ValueError('Metal distance ratios must be nonnegative')
    is_floating = is_min_ratio >= DESORPTION_SCALE
    detached = any(r >= DESORPTION_SCALE for r in fs_fragment_min_ratios)
    all_bound = all(r < BONDED_SCALE for r in fs_fragment_min_ratios)
    grade = (2 if is_floating or detached else
             0 if all_bound and staying_max_displacement_A < settings['staying_fragment_max_displacement_A'] else 1)
    return dict(grade=grade, is_floating=is_floating, fs_detached=detached,
                fs_all_fragments_bound=all_bound, is_min_metal_ratio=float(is_min_ratio),
                fs_fragment_min_metal_ratios=list(map(float, fs_fragment_min_ratios)),
                staying_fragment_max_displacement_A=float(staying_max_displacement_A))


def structure_features(IS, FS, slab_indices, adsorbate_indices, settings=None):
    """No calculator calls. Explicit species indices, ASE radii, MIC distances.

    FS components use the retrospective universal radius ratio < 1.3. The
    staying component minimizes mean atomwise IS-to-FS MIC displacement; its
    largest displacement defines the 1 A grade gate. Atom identities are fixed
    by the pair: no permutation/alignment or reaction-specific exception.
    """
    import itertools
    import numpy as np
    from ase.geometry import find_mic
    from chavis.judgment.structure_labels import _r, _components, _mic_vectors
    slab, ads = list(slab_indices), list(adsorbate_indices)
    if not slab or not ads or set(slab) & set(ads) or set(slab) | set(ads) != set(range(len(IS))):
        raise ValueError('Explicit nonempty, disjoint, complete slab/adsorbate indices required')
    if IS.get_chemical_symbols() != FS.get_chemical_symbols():
        raise ValueError('IS/FS atom identity mismatch')
    if not np.array_equal(IS.pbc, FS.pbc) or not np.allclose(IS.cell, FS.cell, rtol=0, atol=1e-12):
        raise ValueError('IS/FS cell or periodic boundary mismatch')
    symbols = IS.get_chemical_symbols()
    def metal_ratios(atoms):
        return {i: float(min(_mic_vectors(atoms, i, slab)[1] /
                    np.array([_r(symbols[i]) + _r(symbols[j]) for j in slab]))) for i in ads}
    edges = {(i, j) for i, j in itertools.combinations(ads, 2)
             if FS.get_distance(i, j, mic=True) < 1.3 * (_r(symbols[i]) + _r(symbols[j]))}
    fragments = _components(ads, edges)
    displacement = dict(zip(ads, map(float, find_mic(FS.positions[ads] - IS.positions[ads], IS.cell, IS.pbc)[1])))
    staying = min(fragments, key=lambda frag: (float(np.mean([displacement[i] for i in frag])), tuple(frag)))
    is_ratio, fs_ratio = metal_ratios(IS), metal_ratios(FS)
    result = structure_grade(min(is_ratio.values()), [min(fs_ratio[i] for i in f) for f in fragments],
                             max(displacement[i] for i in staying), settings)
    result.update(fs_fragments=fragments, staying_fragment=staying,
                  atom_mic_displacement_A={str(i): displacement[i] for i in ads},
                  slab_indices=slab, adsorbate_indices=ads, fragment_bond_radius_ratio=1.3)
    return result


def annotate_pair_structures(species, pairs, settings=None):
    """Grade the entire representative pool before TS work; no model needed."""
    annotated = []
    for original in pairs:
        pair = copy.deepcopy(original)
        IS = species.atoms(pair['is_slab_positions'], pair['is_adsorbate_positions'])
        FS = species.atoms(pair['fs_slab_positions'], pair['fs_adsorbate_positions'])
        pair['_ultra_structure'] = structure_features(IS, FS, range(species.n),
                                                     range(species.n, len(IS)), settings)
        annotated.append(pair)
    return annotated


def _product_detached(features):
    """Require an explicit product flag; verify saved ratios when available."""
    if not isinstance(features, dict) or type(features.get('fs_detached')) is not bool:
        raise ValueError('demote_product_detached requires a boolean fs_detached for every pair')
    detached = features['fs_detached']
    if 'fs_fragment_min_metal_ratios' in features:
        from chavis.judgment.structure_labels import DESORPTION_SCALE
        ratios = features['fs_fragment_min_metal_ratios']
        if not isinstance(ratios, (list, tuple)) or not ratios:
            raise ValueError('FS fragment metal ratios must be a nonempty sequence')
        for ratio in ratios:
            if _number(ratio, 'FS fragment metal ratio') < 0:
                raise ValueError('Metal distance ratios must be nonnegative')
        if detached != any(ratio >= DESORPTION_SCALE for ratio in ratios):
            raise ValueError('fs_detached disagrees with FS fragment metal ratios')
    return detached


def order_pairs(pairs, metadata, settings=None):
    """Return ordered form queues, retaining every input pair exactly once.

    The caller supplies the existing representative pair pool. This function does
    not apply an additional representative/rotamer filter or perform visits.
    demote_product_detached demotes only pairs with a detached FS fragment;
    neither a floating IS nor the legacy all-floating exception affects it.
    demote_only preserves the legacy energy order for grades 0 and 1; only
    grade 2 pairs, and forms containing only grade 2 pairs, move to the back.
    three_tier uses grade-first ordering for comparison.
    """
    settings = load_settings(overrides=settings)
    mode = settings['order_mode']
    groups, seen_ids = {}, set()
    all_floating = (mode != 'demote_product_detached' and bool(pairs)
                    and all((p.get('_ultra_structure') or {}).get('is_floating') is True for p in pairs))
    for position, original in enumerate(pairs):
        if not isinstance(original, dict) or original.get("paired_is") is None:
            raise ValueError("Each pair must identify its paired_is")
        pair = copy.deepcopy(original)
        identifier = str(pair["paired_is"])
        info = copy.deepcopy(metadata.get(identifier, _unknown_form(identifier)))
        stable_id = pair.get("pair_id")
        if stable_id is None:
            group = pair.get("pair_group", position)
            if type(group) is not int or group < 0:
                raise ValueError("A pair without pair_id requires a nonnegative integer pair_group")
            stable_id = f"pair_group_{group:09d}"
        if not isinstance(stable_id, str) or not stable_id or stable_id in seen_ids:
            raise ValueError("Unique nonempty stable pair IDs are required")
        seen_ids.add(stable_id)
        e_is = _number(pair.get("is_energy_eV"), "pair.is_energy_eV")
        _number(pair.get("reaction_energy_eV"), "pair.reaction_energy_eV")
        raw_grade = (pair.get('_ultra_structure') or {}).get('grade', 1)
        if type(raw_grade) is not int or raw_grade not in (0, 1, 2):
            raise ValueError('Structure grade must be 0, 1, or 2')
        pair['_ultra_grade'] = 1 if all_floating else raw_grade
        pair['_ultra_grade_ignored_all_floating'] = all_floating
        if mode == 'demote_product_detached':
            priority = int(_product_detached(pair.get('_ultra_structure')))
        elif mode == 'demote_only':
            priority = int(pair['_ultra_grade'] == 2)
        else:
            priority = pair['_ultra_grade']
        pair['_ultra_order_mode'] = mode
        pair['_ultra_effective_priority'] = priority
        pair['_ultra_demoted'] = priority == (2 if mode == 'three_tier' else 1)
        pair["_ultra_stable_id"] = stable_id
        pair["_ultra_form"] = info
        key = info["group_key"]
        group = groups.setdefault(key, {"group_key": key, "form": info["form"],
                                        "outcome": info["outcome"], "min_is_energy_eV": e_is,
                                        "best_grade": pair['_ultra_grade'],
                                        "effective_priority": priority,
                                        "all_pairs_demoted": True, "pairs": []})
        group["min_is_energy_eV"] = min(group["min_is_energy_eV"], e_is)
        group['best_grade'] = min(group['best_grade'], pair['_ultra_grade'])
        group['effective_priority'] = min(group['effective_priority'], priority)
        group['all_pairs_demoted'] = group['all_pairs_demoted'] and pair['_ultra_demoted']
        group["pairs"].append(pair)
    for group in groups.values():
        group["pairs"].sort(key=lambda p: (p['_ultra_effective_priority'], p["is_energy_eV"], p["reaction_energy_eV"], p["_ultra_stable_id"]))
    # In either demotion mode, a form moves back only if every pair is demoted.
    # Its energy key always uses the entire form, including the demoted pairs.
    return sorted(groups.values(), key=lambda g: (g['effective_priority'], g["min_is_energy_eV"], g["group_key"]))


def _energies(points):
    if not isinstance(points, (list, tuple)) or not points:
        raise ValueError("Scan points must be a nonempty sequence")
    if any(not isinstance(point, dict) for point in points):
        raise ValueError("Scan points must be objects")
    return [_number(point.get("E_rel_IS_eV"), "scan energy") for point in points]


def classify_scan(points, settings=None):
    """Classify unrounded energies; endpoint ties count as broken scans.

    The local jump compares consecutive points whose BOTH indices lie in
    [max(0, peak-window), min(n-1, peak+window)].
    """
    settings = load_settings(overrides=settings)
    energies = _energies(points)
    maximum = max(energies)
    peak = energies.index(maximum)  # legacy first-index tie break
    endpoint = energies[0] == maximum or energies[-1] == maximum
    window = settings["scan_jump_window_points"]
    start, end = max(0, peak - window), min(len(points) - 1, peak + window)
    jump = max((abs(energies[i + 1] - energies[i]) for i in range(start, end)), default=0.0)
    reasons = []
    if endpoint:
        reasons.append("endpoint_global_maximum")
    if jump >= settings["scan_jump_defer_eV"]:
        reasons.append("local_energy_jump")
    return {"broken": bool(reasons), "peak": peak, "endpoint_maximum": endpoint,
            "max_local_jump_eV": jump, "jump_window_start": start,
            "jump_window_end": end, "reasons": reasons}


def bond_prior_distance(bond_elements, scale=1.5):
    _number(scale, "bond prior scale", positive=True)
    if not isinstance(bond_elements, (list, tuple)) or len(bond_elements) != 2:
        raise ValueError("Exactly two breaking-bond element symbols are required")
    if any(not isinstance(element, str) or not element for element in bond_elements):
        raise ValueError("Breaking-bond element symbols must be nonempty strings")
    if any(element not in COVALENT_RADII_A for element in bond_elements):
        return None
    return scale * sum(COVALENT_RADII_A[element] for element in bond_elements)


def order_seeds(points, old_order, bond_elements, classification=None, settings=None, log=None):
    """Move at most one existing eligible point to front; preserve all others."""
    settings = load_settings(overrides=settings)
    actual_classification = classify_scan(points, settings)
    if classification is not None and classification != actual_classification:
        raise ValueError("Supplied scan classification does not match points/settings")
    classification = actual_classification
    if not isinstance(old_order, (list, tuple)):
        raise ValueError("Legacy seed order must be a sequence of point indices")
    order = list(old_order)
    if any(type(index) is not int or not 0 <= index < len(points) for index in order) or len(set(order)) != len(order):
        raise ValueError("Legacy seed indices must be unique and within the scan")
    bonds = [_number(point.get("bond_A"), "scan bond length", positive=True) for point in points]
    prior = bond_prior_distance(bond_elements, settings["bond_prior_scale"])
    result = {"order": order, "prior_index": None, "prior_A": prior,
              "prior_applied": False, "reason": "normal_scan_legacy_order"}
    if prior is None:
        result["reason"] = "unknown_element_legacy_order"
        _warn(f"No frozen CHON covalent-radius prior for {list(bond_elements)}; retaining legacy seed order", log)
    elif not order:
        result["reason"] = "no_eligible_seeds"
    elif classification["broken"]:
        position = min(range(len(order)), key=lambda j: (abs(bonds[order[j]] - prior), j))
        index = order[position]
        result.update(order=[index] + order[:position] + order[position + 1:],
                      prior_index=index, prior_applied=True, reason="broken_scan_one_prior_seed_first")
    return result
