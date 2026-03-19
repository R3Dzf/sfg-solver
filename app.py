from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
import networkx as nx
import sympy as sp
import itertools

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Connection(BaseModel):
    id: str = ""
    from_node: str
    to_node: str
    gain: str

class DiagramData(BaseModel):
    connections: list[Connection]
    input_node: str
    output_node: str

def _normalize_gain(expr: str) -> str:
    return (expr or "").replace("^", "**")

@app.post("/validate")
def validate_diagram(data: DiagramData):
    errors: list[dict] = []
    warnings: list[dict] = []

    if not (data.input_node or "").strip():
        errors.append({"type": "missing_input", "message": "Input node name is empty."})
    if not (data.output_node or "").strip():
        errors.append({"type": "missing_output", "message": "Output node name is empty."})

    G = nx.DiGraph()
    dummy_idx = 0
    for conn in data.connections:
        frm = (conn.from_node or "").strip()
        to = (conn.to_node or "").strip()
        raw_gain = (conn.gain or "").strip()

        if not frm or not to:
            errors.append({"type": "missing_endpoint", "message": "Connection is missing from/to node.", "edge": {"from": frm, "to": to}})
            continue

        if not raw_gain:
            errors.append({"type": "missing_gain", "message": f"Gain is missing on edge {frm} -> {to}.", "edge": {"from": frm, "to": to}})
            continue

        try:
            sym_weight = sp.sympify(_normalize_gain(raw_gain))
        except Exception as e:
            errors.append({"type": "invalid_gain", "message": f"Invalid gain on edge {frm} -> {to}: {e}", "edge": {"from": frm, "to": to}, "gain": raw_gain})
            continue

        # 🔥 ابتكار العقدة الوهمية لمنع دمج الأسلاك المتوازية
        if G.has_edge(frm, to):
            dummy_idx += 1
            dummy_id = f"__SFG_DUMMY_{dummy_idx}__"
            G.add_edge(frm, dummy_id, weight=sym_weight)
            G.add_edge(dummy_id, to, weight=sp.Integer(1))
        else:
            G.add_edge(frm, to, weight=sym_weight)

    if errors:
        return {"status": "error", "errors": errors, "warnings": warnings}

    # فلترة العقد الوهمية قبل التحقق
    real_nodes = [n for n in G.nodes if not str(n).startswith("__SFG_DUMMY_")]

    if data.input_node not in real_nodes:
        errors.append({"type": "missing_input_node", "message": f"Input node '{data.input_node}' not found."})
    if data.output_node not in real_nodes:
        errors.append({"type": "missing_output_node", "message": f"Output node '{data.output_node}' not found."})

    if errors:
        return {"status": "error", "errors": errors, "warnings": warnings}

    has_path = nx.has_path(G, data.input_node, data.output_node)
    if not has_path:
        errors.append({"type": "no_path", "message": f"No forward path from '{data.input_node}' to '{data.output_node}'."})
        return {"status": "error", "errors": errors, "warnings": warnings}

    reachable_from_in = set(nx.descendants(G, data.input_node)) | {data.input_node}
    can_reach_out = set(nx.descendants(G.reverse(copy=False), data.output_node)) | {data.output_node}
    on_io_path = reachable_from_in & can_reach_out

    off_path = sorted([n for n in real_nodes if n not in on_io_path])
    if off_path:
        warnings.append({"type": "off_path_nodes", "message": "Some nodes are not on any path from input to output.", "nodes": off_path})

    isolated = sorted([n for n in real_nodes if G.in_degree(n) == 0 and G.out_degree(n) == 0])
    if isolated:
        warnings.append({"type": "isolated_nodes", "message": "Some nodes are isolated.", "nodes": isolated})

    return {"status": "success", "errors": [], "warnings": warnings}

def _delta_from_loops(loops: list[dict], *, collect_nt: bool = True) -> tuple[sp.Expr, dict[int, list[dict]]]:
    if not loops:
        return sp.Integer(1), {}

    loops_gains = [l["gain"] for l in loops]
    delta = sp.Integer(1) - sum(loops_gains, sp.Integer(0))

    if len(loops) < 2:
        return sp.simplify(delta), {}

    sums_by_k: dict[int, sp.Expr] = {}
    nt_dict: dict[int, list[dict]] = {} if collect_nt else {}
    n = len(loops)

    def rec(start: int, used_nodes: set, chosen_idx: list[int], prod_gain: sp.Expr) -> None:
        for i in range(start, n):
            li = loops[i]
            if li["nodes"] & used_nodes:
                continue

            new_used = used_nodes | li["nodes"]
            new_chosen = chosen_idx + [i]
            new_prod = prod_gain * li["gain"]
            k = len(new_chosen)

            if k >= 2:
                sums_by_k[k] = sums_by_k.get(k, sp.Integer(0)) + new_prod
                if collect_nt:
                    combo_edge_ids = []
                    for j in new_chosen:
                        combo_edge_ids.extend(loops[j].get("edge_ids", []))
                    nt_dict.setdefault(k, []).append({
                        "combos": [list(loops[j]["ordered_nodes"]) for j in new_chosen],
                        "combo_edge_ids": combo_edge_ids,
                        "gain": new_prod
                    })

            rec(i + 1, new_used, new_chosen, new_prod)

    rec(0, set(), [], sp.Integer(1))

    for k, s in sums_by_k.items():
        delta += ((-1) ** k) * s

    if collect_nt:
        nt_dict = {k: v for k, v in nt_dict.items() if v}
    else:
        nt_dict = {}

    return sp.simplify(delta), nt_dict

def _transfer_function_linear(G: nx.DiGraph, input_node: str, output_node: str) -> sp.Expr:
    nodes = list(G.nodes())
    if input_node not in nodes or output_node not in nodes:
        raise ValueError("input/output node not found")

    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    M = sp.Matrix.zeros(n, n)
    b = sp.Matrix.zeros(n, 1)

    for v in nodes:
        i = idx[v]
        M[i, i] = sp.Integer(1)
        for u in G.predecessors(v):
            w = G[u][v]["weight"]
            M[i, idx[u]] -= w
        if v == input_node:
            b[i, 0] = sp.Integer(1)

    try:
        x = M.LUsolve(b)
    except Exception:  
        x = M.gauss_jordan_solve(b)[0]

    return sp.simplify(x[idx[output_node], 0])

@app.post("/solve")
def solve_diagram(data: DiagramData):
    try:
        G = nx.DiGraph()
        dummy_idx = 0
        for conn in data.connections:
            sym_weight = sp.sympify(_normalize_gain(conn.gain))
            frm = (conn.from_node or "").strip()
            to = (conn.to_node or "").strip()
            
            # تطبيق العقد الوهمية لمنع التداخل وحفظ الـ ID
            if G.has_edge(frm, to):
                dummy_idx += 1
                dummy_id = f"__SFG_DUMMY_{dummy_idx}__"
                G.add_edge(frm, dummy_id, weight=sym_weight, edge_id=conn.id)
                G.add_edge(dummy_id, to, weight=sp.Integer(1), edge_id=None)
            else:
                G.add_edge(frm, to, weight=sym_weight, edge_id=conn.id)

        real_nodes = [n for n in G.nodes if not str(n).startswith("__SFG_DUMMY_")]
        if data.input_node not in real_nodes:
            return {"status": "error", "message": f"Input node '{data.input_node}' not found."}
        if data.output_node not in real_nodes:
            return {"status": "error", "message": f"Output node '{data.output_node}' not found."}

        raw_loops = list(nx.simple_cycles(G))
        loops_data = []
        for cycle in raw_loops:
            gain = sp.Integer(1)
            edge_ids = []
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                gain *= G[u][v]["weight"]
                eid = G[u][v].get("edge_id")
                if eid:
                    edge_ids.append(eid)
            filtered_cycle = [n for n in cycle if not str(n).startswith("__SFG_DUMMY_")]
            loops_data.append({"nodes": set(filtered_cycle), "ordered_nodes": filtered_cycle, "gain": sp.simplify(gain), "edge_ids": edge_ids})

        delta, nt_dict = _delta_from_loops(loops_data, collect_nt=True)

        raw_paths = list(nx.all_simple_paths(G, data.input_node, data.output_node))
        paths_res = []
        numerator = sp.Integer(0)
        for p in raw_paths:
            filtered_p = [n for n in p if not str(n).startswith("__SFG_DUMMY_")]
            p_nodes = set(filtered_p)
            p_gain = sp.Integer(1)
            edge_ids = []
            for i in range(len(p) - 1):
                u = p[i]
                v = p[i + 1]
                p_gain *= G[u][v]["weight"]
                eid = G[u][v].get("edge_id")
                if eid:
                    edge_ids.append(eid)

            v_loops = [l for l in loops_data if l["nodes"].isdisjoint(p_nodes)]
            dk, _ = _delta_from_loops(v_loops, collect_nt=False)

            numerator += sp.simplify(p_gain) * dk
            paths_res.append({
                "path": filtered_p,
                "gain": sp.latex(sp.simplify(p_gain)),
                "delta_k": sp.latex(dk),
                "edge_ids": edge_ids
            })

        tf_mason = sp.simplify(numerator / delta) if delta != 0 else sp.nan
        tf_linear = _transfer_function_linear(G, data.input_node, data.output_node)

        mismatch = False
        try:
            mismatch = sp.simplify(tf_mason - tf_linear) != 0
        except Exception: 
            mismatch = False

        tf = sp.simplify(tf_linear)

        # 💡 الدالة السحرية لضبط مكان الـ 1 في بداية المقام أو المحدد
        def fix_one_position(expr):
            s = sp.latex(expr)
            if s.endswith(" + 1"):
                s = s[:-4].strip()
                if s.startswith("-"):
                    return "1 - " + s[1:].strip()
                else:
                    return "1 + " + s
            return s

        # فصل البسط والمقام لتطبيق التعديل على المقام فقط
        num, den = sp.fraction(tf)
        if den == 1:
            tf_latex = sp.latex(num)
        else:
            tf_latex = f"\\frac{{{sp.latex(num)}}}{{{fix_one_position(den)}}}"

        return {
            "status": "success",
            "transfer_function": tf_latex,
            "transfer_function_mason": sp.latex(tf_mason),
            "transfer_function_linear": sp.latex(tf_linear),
            "transfer_function_expr": sp.sstr(tf),
            "transfer_function_mason_expr": sp.sstr(tf_mason),
            "transfer_function_linear_expr": sp.sstr(tf_linear),
            "mason_mismatch": bool(mismatch),
            "numerator": sp.latex(sp.simplify(numerator)),
            "delta": fix_one_position(delta),
            "numerator_expr": sp.sstr(sp.simplify(numerator)),
            "delta_expr": sp.sstr(delta),
            "paths": paths_res,
            "loops": [{"loop": l["ordered_nodes"], "gain": sp.latex(l["gain"]), "edge_ids": l["edge_ids"]} for l in loops_data],
            "non_touching": [
                {
                    "type": f"{k}-Non-Touching",
                    "details": [{"combos": d["combos"], "combo_edge_ids": d.get("combo_edge_ids", []), "gain": sp.latex(sp.simplify(d["gain"]))} for d in v],
                }
                for k, v in nt_dict.items()
            ],
        }
    except Exception as e: return {"status": "error", "message": str(e)}

    # السطر ده بيخلي سيرفر البايثون يعرض فولدر الواجهة تلقائياً
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")