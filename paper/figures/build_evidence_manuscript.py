#!/usr/bin/env python3
"""Publication exports from existing cleared records only; never runs models.

Allowed: fixed policies in formal_v1, AWGN learned policies, corrected DV
target refit and crossreceiver_v2. Never uses Rayleigh/Rician full routers.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pubfig as pf
from pubfig.specs import FigureSpec
from publication_style import METHODS, COMPONENTS, apply_style, style_axis

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER / "outputs/revision_20260907_independent"
OUT = PAPER / "figures/evidence_final"
REPORT = PAPER / "outputs/evidence_manuscript"
SOURCES: dict[str, str] = {}
SNRS = [-5, 0, 5, 10, 15, 20]
TYPES = ["presence", "counting", "comparison", "co_presence", "threshold"]
LABELS = ["Presence", "Counting", "Comparison", "Co-presence", "Threshold"]

def read(path: Path) -> dict:
    SOURCES[str(path.relative_to(PAPER))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())

def arrays(path: Path) -> dict:
    SOURCES[str(path.relative_to(PAPER))] = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path) as f:
        return {k: f[k].copy() for k in f.files}

def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")

def export(fig, name: str, height: float = 76) -> None:
    pf.batch_export(fig, OUT/name, spec=FigureSpec(name="communications", font_family="Times New Roman"),
                    width=182, height_mm=height, formats=("pdf", "svg", "png"), dpi=400, trim=False)
    plt.close(fig)

def handle(name: str) -> Line2D:
    color, marker, ls, label = METHODS[name]
    return Line2D([], [], color=color, marker=marker, ls=ls, mfc="white", label=label)

def curve(ax, name: str, x, y) -> None:
    c, m, ls, label = METHODS[name]
    ax.plot(x, y, color=c, marker=m, ls=ls, mfc="white", label=label)

def main() -> None:
    OUT.mkdir(exist_ok=True)
    REPORT.mkdir(exist_ok=True)
    apply_style()
    for name in ("formal_v1", "supplement_v1", "dronevehicle_v1", "crossreceiver_v2"):
        assert read(ROOT/name/"status.json")["state"] == "COMPLETE"
    vd = {ch: read(ROOT/"formal_v1"/ch/"train_only/baselines.json")
          for ch in ("awgn", "rayleigh", "rician")}
    dvpath = ROOT/"dronevehicle_v1/target_refit/train_only"
    dv = read(dvpath/"baselines.json")
    seeds = [read(dvpath/f"seed_{i}.json") for i in range(10)]
    linear = read(dvpath/"linear.json")
    # Fixed-policy data do not consume the disputed sender-count feature.
    fig = plt.figure(figsize=(7.16, 3))
    methods = [("image", "image"), ("token", "detection"), ("rule", "rule"), ("lut", "lut")]
    for i, ch in enumerate(vd):
        ax = fig.add_axes([.075+i*.308, .19, .28, .60])
        for style, key in methods:
            curve(ax, style, SNRS, [100*vd[ch][key]["per_snr"][str(s)]["accuracy"] for s in SNRS])
        channel_label = "AWGN" if ch == "awgn" else ch.capitalize()
        ax.set(xlabel="SNR (dB)", xticks=SNRS, ylim=(55, 72), title=f"({chr(97+i)}) {channel_label}")
        if i == 0: ax.set_ylabel("Answer accuracy (%)")
        else: ax.tick_params(labelleft=False)
        style_axis(ax)
    fig.legend(handles=[handle(m) for m, _ in methods], loc="upper center", ncol=4, frameon=False, fontsize=7.5)
    export(fig, "type_routing")

    fig = plt.figure(figsize=(7.16, 3.2))
    for i, (label, b) in enumerate((("VisDrone", vd["rician"]), ("DroneVehicle", dv))):
        ax = fig.add_axes([.08+i*.49, .26, .39, .56])
        for style, key, offset in (("token", "detection", -.08), ("image", "image", .08)):
            c, m, _, title = METHODS[style]
            ax.scatter(np.arange(5)+offset, [100*b[key]["per_type"][t]["accuracy"] for t in TYPES],
                       color=c, marker=m, facecolors="white", label=title, zorder=3)
        for j,t in enumerate(TYPES):
            ax.plot([j-.08,j+.08], [100*b[k]["per_type"][t]["accuracy"] for k in ("detection","image")], color="#B8BEC4", lw=1)
        ax.set(xticks=np.arange(5), ylim=(25,95), title=f"({chr(97+i)}) {label}, Rician")
        ax.set_xticklabels(LABELS, rotation=22, ha="right")
        if i == 0: ax.set_ylabel("Answer accuracy (%)")
        style_axis(ax)
    fig.legend(handles=[handle(m) for m in ("token", "image")], loc="upper center", ncol=2, frameon=False)
    export(fig, "evidence_complementarity", 82)

    fig = plt.figure(figsize=(7.16, 3.2))
    ax = fig.add_axes([.10,.20,.60,.70])
    es = np.array([[r["energy_j"] for r in s["sweep"]] for s in seeds])
    ac = 100*np.array([[r["accuracy"] for r in s["sweep"]] for s in seeds])
    curve(ax, "mlp", es.mean(axis=0), ac.mean(axis=0))
    ax.fill_between(es.mean(axis=0), ac.mean(axis=0)-ac.std(axis=0,ddof=1),
                    ac.mean(axis=0)+ac.std(axis=0,ddof=1),color=METHODS["mlp"][0],alpha=.13,lw=0)
    curve(ax, "linear", [r["energy_j"] for r in linear["sweep"]], [100*r["accuracy"] for r in linear["sweep"]])
    for style,key in (("image","image"),("token","detection"),("rule","rule")):
        c,m,_,lab=METHODS[style]; r=dv[key]["pooled"]
        ax.scatter(r["energy_j"],100*r["accuracy"],color=c,marker=m,s=40,label=lab,zorder=4)
    ax.errorbar(np.mean([s["selected_test"]["energy_j"] for s in seeds]),
                100*np.mean([s["selected_test"]["accuracy"] for s in seeds]),
                xerr=np.std([s["selected_test"]["energy_j"] for s in seeds],ddof=1),
                yerr=100*np.std([s["selected_test"]["accuracy"] for s in seeds],ddof=1),
                fmt="*",color="#202020",ms=9,capsize=3,label="Validation-selected MLP",zorder=5)
    ax.set(xscale="log",xlim=(.35,40),ylim=(62,78),xlabel="Accounted system energy (J/query; log scale)",ylabel="Answer accuracy (%)")
    style_axis(ax); ax.legend(loc="upper left",bbox_to_anchor=(1.01,1),frameon=False,fontsize=7.5)
    export(fig,"dv_tradeoff",82)

    inp=arrays(dvpath/"evaluation_inputs.npz")
    picks={"Image":[inp["image"]],"Detection":[inp["detection"]],"Type rule":[inp["rule"]],
           "Linear":[arrays(dvpath/"linear_outcomes.npz")["pick"]],"MLP":[],"Priced MLP":[]}
    for i,s in enumerate(seeds):
        a=arrays(dvpath/f"seed_{i}_outcomes.npz")
        picks["MLP"].append(a["pick"]); picks["Priced MLP"].append(a["sweep_picks"][s["validation_selected_index"]])
    # Recover branch radio energy from the saved matrix, preserving exact costs.
    vlm=32.310144895318594
    comp={}
    for name, ps in picks.items():
        learned=name in ("Linear","MLP","Priced MLP")
        rows=[]
        for pk in ps:
            f=float(np.mean(pk)); det=.4275*(1 if learned else 1-f)
            en=inp["test_energy"] if learned else inp["baseline_energy"]
            total=float(en[np.arange(len(pk)),pk].mean()); edge=f*vlm
            components = [total-det-edge,det,edge]
            assert min(components) >= -1e-10
            assert np.isclose(sum(components), total)
            rows.append(components)
        comp[name]=np.mean(rows,axis=0).tolist()
    fig=plt.figure(figsize=(7.16,3.3)); names=list(comp); vals=np.array(list(comp.values()))
    for i,title in enumerate(("(a) UAV and edge","(b) UAV only")):
        ax=fig.add_axes([.08+i*.49,.29,.39,.54]); bottom=np.zeros(len(names))
        for j,key in enumerate(("radio","detector","vlm") if i==0 else ("radio","detector")):
            color,hatch=COMPONENTS[key]
            ax.bar(np.arange(len(names)),vals[:,j],bottom=bottom,color=color,hatch=hatch,edgecolor="white",linewidth=.5,label={"radio":"Radio","detector":"Onboard detector","vlm":"Edge VLM"}[key]); bottom+=vals[:,j]
        ax.set(xticks=np.arange(len(names)),title=title,ylabel="Energy (J/query)")
        ax.set_xticklabels(names,rotation=32,ha="right"); style_axis(ax)
    fig.legend(*fig.axes[0].get_legend_handles_labels(),loc="upper center",ncol=3,frameon=False)
    export(fig,"dv_energy_components",86)

    # All audited AWGN ablations are retained, including the weak full-model gain.
    aw=read(ROOT/"supplement_v1/summary.json")["awgn"]["train_only"]
    names=["full","no_detector","no_snr","question_type_only"]
    fig=plt.figure(figsize=(7.16,2.7)); ax=fig.add_axes([.11,.26,.84,.55])
    for offset,setting,color,marker,label in ((-.09,"unpriced",METHODS["mlp"][0],"o","No energy price"),(.09,"validation_selected",METHODS["rule"][0],"s","Validation-selected price")):
        ax.errorbar(np.arange(4)+offset,[100*aw[k][setting]["accuracy"]["mean"] for k in names],
                    yerr=[100*aw[k][setting]["accuracy"]["sample_sd"] for k in names],fmt=marker,color=color,mfc="white",capsize=3,label=label)
    ax.set(xticks=np.arange(4),ylabel="Answer accuracy (%)",ylim=(66,69))
    ax.set_xticklabels(["Full inputs","No detector inputs","No SNR","Question type only"])
    style_axis(ax)
    fig.legend(*ax.get_legend_handles_labels(),frameon=False,ncol=2,loc="upper center",bbox_to_anchor=(.5,.99))
    export(fig,"awgn_ablation",70)
    write(REPORT/"energy_components.json",comp)
    write(REPORT/"source_sha256.json",SOURCES)
    write(REPORT/"numeric_summary.json",{
        "visdrone_fixed":{ch:{k:b[k]["pooled"] for k in ("image","detection","rule","lut")} for ch,b in vd.items()},
        "dv_mlp_selected_accuracy":float(ac[np.arange(10),[s['validation_selected_index'] for s in seeds]].mean()),
        "dv_mlp_selected_energy":float(np.mean([s['selected_test']['energy_j'] for s in seeds])),
        "awgn_ablations":aw,
        "excluded":"Rayleigh/Rician learned predictors consuming disputed count inputs; historical answer-derived-feature models"})
    print(f"Exported 5 figures to {OUT}; no fitting, inference, or parameter selection performed.")

if __name__=="__main__":
    main()
