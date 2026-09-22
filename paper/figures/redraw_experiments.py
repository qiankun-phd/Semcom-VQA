#!/usr/bin/env python3
"""Build manuscript figures from audited, existing experiment records only."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pubfig as pf
from pubfig.specs import FigureSpec

from publication_style import METHODS, COMPONENTS, apply_style, style_axis

PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / "figures/revised"
DATA = json.loads((PAPER / "outputs/revision_20260907/audited_results.json").read_text())
STATS = json.loads((PAPER / "outputs/reports/paper1_stats.json").read_text())
SNRS = [-5, 0, 5, 10, 15, 20]
TYPES = ["counting", "comparison", "co_presence", "threshold", "presence"]
TYPE_LABELS = ["Counting", "Comparison", "Co-presence", "Threshold", "Presence"]


def canvas(height: float = 78):
    return plt.figure(figsize=(182/25.4, height/25.4))


def save(fig, name: str, height: float = 78) -> None:
    pf.batch_export(fig, OUT/name, spec=FigureSpec(name="communications", font_family="Times New Roman"),
                    width=182, height_mm=height, dpi=400, formats=("pdf", "svg", "png"), trim=False)
    plt.close(fig)


def handle(method: str, label: str | None = None):
    c, m, ls, text = METHODS[method]
    return Line2D([], [], color=c, marker=m, ls=ls, mfc="white", label=label or text, ms=4)


def line(ax, method: str, x, y, label: str | None = None, **kwargs) -> None:
    c, marker, ls, name = METHODS[method]
    ax.plot(x, y, color=c, marker=marker, ls=ls, mfc="white", mew=0.9,
            lw=1.65 if method == "mlp" else 1.2, label=label or name, **kwargs)


def accuracy() -> None:
    fig = canvas()
    methods = ["mlp", "rule", "token", "image", "oracle", "djscc"]
    for i, channel in enumerate(("awgn", "rayleigh", "rician")):
        ax = fig.add_axes([0.075+i*0.307, 0.20, 0.28, 0.53])
        data = DATA["series"][channel]
        for method in methods:
            if method not in data:
                continue
            rows = data[method]
            line(ax, method, SNRS, [r["accuracy"] for r in rows])
            if method == "mlp":
                ax.fill_between(SNRS, [r["acc_min"] for r in rows], [r["acc_max"] for r in rows],
                                color=METHODS[method][0], alpha=0.12, linewidth=0)
        ax.set(xlim=(-6,21), ylim=(0.55,0.78), xticks=SNRS, yticks=[0.55,0.60,0.65,0.70,0.75], xlabel="SNR (dB)")
        if i == 0:
            ax.set_ylabel("Answer accuracy")
        else:
            ax.tick_params(labelleft=False)
        ax.set_title(f'({chr(97+i)}) '+["AWGN", "Rayleigh", "Rician (K = 6 dB)"][i], pad=8)
        style_axis(ax)
    fig.legend(handles=[handle(m) for m in methods], loc="upper center", bbox_to_anchor=(0.54,1),
               ncol=3, frameon=False, columnspacing=1.4, handlelength=2.2)
    fig.text(0.54,0.02,"MLP: 10-seed mean and min-max range. DJSCC: Rician only.",ha="center",fontsize=8,color="#41464B")
    save(fig,"overall_accuracy")


def complementarity() -> None:
    fig = canvas(84)
    left = fig.add_axes([0.075,0.23,0.405,0.58])
    right = fig.add_axes([0.575,0.23,0.405,0.58])
    vd = STATS["table3_delta_visdrone"]
    dv = STATS["table3_delta_dronevehicle"]
    x = np.arange(5)
    for method,key,offset in (("token","token_acc",-0.1),("image","image_acc",0.1)):
        c,m,_,name = METHODS[method]
        left.scatter(x+offset,[vd[t][key] for t in TYPES],color=c,marker=m,facecolors="white",s=27,label=name,zorder=3)
    for i,t in enumerate(TYPES):
        left.plot([i-0.1,i+0.1],[vd[t]["token_acc"],vd[t]["image_acc"]],color="#BCC2C8",lw=1,zorder=1)
    for offset,records,marker,fill,label in ((-0.1,vd,"o","#41464B","VisDrone: three channels"),
                                            (0.1,dv,"s","white","DroneVehicle: Rician")):
        y = np.array([records[t]["delta"] for t in TYPES])
        low = np.array([records[t]["ci95"][0] for t in TYPES])
        high = np.array([records[t]["ci95"][1] for t in TYPES])
        right.errorbar(x+offset,y,yerr=[y-low,high-y],fmt=marker,mfc=fill,color="#41464B",capsize=2,ms=4,label=label)
    right.axhline(0,color="#666666",ls="--",lw=0.8)
    for ax in (left,right):
        ax.set_xticks(x,TYPE_LABELS,rotation=25,ha="right")
        ax.set_xlim(-0.45,4.45)
        style_axis(ax)
    left.set(ylim=(0.2,0.9),ylabel="Answer accuracy")
    right.set(ylim=(-0.16,0.31),ylabel="Accuracy gap (detection - image)")
    left.set_title("(a) Branch accuracy on VisDrone",pad=12)
    right.set_title("(b) Paired gap and 95% cluster CI",pad=12)
    left.legend(loc="upper left",bbox_to_anchor=(0,1.31),frameon=False,fontsize=7.5)
    right.legend(loc="upper left",bbox_to_anchor=(0,1.31),frameon=False,fontsize=7.5)
    fig.text(0.53,0.018,"Intervals resample images; per-type comparisons are exploratory.",ha="center",fontsize=8,color="#41464B")
    save(fig,"complementarity",84)


def ablation() -> None:
    data = json.loads((PAPER/"data/w11_mlp_deepening.json").read_text())["ablation_rician_mlp"]
    labels = [("full","Full MLP"),("drop_qtype","Without question type"),("drop_class","Without object class"),
              ("drop_view_risk","Without viewpoint / risk"),("drop_snr","Without SNR"),
              ("drop_detector","Without detector features"),("drop_polarity","Without polarity features")]
    fig = canvas(78)
    ax = fig.add_axes([0.29,0.19,0.69,0.64])
    for i,(key,label) in enumerate(labels):
        c = METHODS["mlp"][0] if key == "full" else METHODS["token"][0]
        r = data[key]
        ax.errorbar(r["acc_mean"]*100,i,xerr=r["acc_std"]*100,fmt="o",mfc=c if key=="full" else "white",color=c,capsize=3,ms=5)
        ax.text(71.02,i,f'{100*r["acc_mean"]:.2f} ± {100*r["acc_std"]:.2f}',va="center",fontsize=8,color=c)
    ax.axvline(data["full"]["acc_mean"]*100,color=METHODS["mlp"][0],ls="--",lw=0.8)
    ax.set(yticks=range(7),yticklabels=[label for _,label in labels],xlim=(67.2,72.25),
           xticks=[67.5,68,68.5,69,69.5,70,70.5],xlabel="Answer accuracy (%)")
    ax.invert_yaxis()
    ax.tick_params(axis="y",length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x",color="#E2E5E8",lw=0.55)
    ax.set_title("Rician channel: MLP feature ablation",loc="left",pad=15)
    ax.text(71.02,-0.68,"Mean ± SD",fontsize=8)
    fig.text(0.56,0.025,"Three seeds; error bars show SD, not confidence intervals.",ha="center",fontsize=8,color="#41464B")
    save(fig,"mlp_ablation")


def tradeoff() -> None:
    fig = canvas(85)
    a = fig.add_axes([0.08,0.23,0.405,0.56])
    b = fig.add_axes([0.575,0.23,0.405,0.56])
    methods = ["mlp","rule","lut","linear","token","image","djscc"]
    for method in methods:
        rows = DATA["series"]["rician"][method]
        line(a,method,[r["energy_j"] for r in rows],[r["accuracy"] for r in rows])
    a.set(xscale="log",xlim=(0.35,45),ylim=(0.55,0.74),xlabel="Accounted energy per question (J)",ylabel="Answer accuracy")
    a.set_title("(a) Fixed policies across the SNR grid",pad=9)
    sweep = DATA["recorded_sweep"]
    line(b,"mlp",[r["energy_j"] for r in sweep],[r["accuracy"] for r in sweep],ms=2.8)
    b.scatter(sweep[0]["energy_j"],sweep[0]["accuracy"],marker="*",s=75,color=METHODS["mlp"][0],zorder=5)
    b.annotate("Unpriced policy",(sweep[0]["energy_j"],sweep[0]["accuracy"]),xytext=(6,0.719),
               fontsize=8,arrowprops={"arrowstyle":"-","lw":0.6})
    b.set(xlim=(0,11.5),ylim=(0.62,0.735),xlabel="Re-accounted energy per question (J)",ylabel="Answer accuracy")
    b.set_title("(b) Recorded price-sweep policies",pad=9)
    for ax in (a,b): style_axis(ax)
    fig.legend(handles=[handle(m) for m in methods],loc="upper center",bbox_to_anchor=(0.52,1.015),ncol=4,frameon=False,fontsize=7.5,columnspacing=1.1)
    fig.text(0.53,0.055,"(a) MLP: 10-seed means. (b) Three-seed means, six SNRs pooled.",ha="center",fontsize=8,color="#41464B")
    fig.text(0.53,0.012,"Detection cost is included for every learned-router query; sweep decisions are unchanged.",ha="center",fontsize=8,color="#41464B")
    save(fig,"energy_tradeoff",85)


def energy_budget() -> None:
    fig = canvas(78)
    methods = ["token","rule","mlp","djscc","image"]
    labels = ["Detection\nevidence","Type\nrule","MLP","DJSCC","Image"]
    rows = DATA["pooled"]["rician"]
    for i,platform in enumerate(("system","uav")):
        ax = fig.add_axes([0.075+i*0.50,0.22,0.405,0.54])
        bottom = np.zeros(5)
        for component in (["vlm","detector","radio"] if platform=="system" else ["detector","radio"]):
            c,hatch = COMPONENTS[component]
            values = np.array([rows[m][component+"_j"] for m in methods])
            ax.bar(range(5),values,bottom=bottom,color=c,hatch=hatch,edgecolor="white",lw=0.4,width=0.6)
            bottom += values
        for j,v in enumerate(bottom):
            ax.annotate(f'{v:.2f}',(j,v),xytext=(0,4),textcoords="offset points",ha="center",fontsize=8)
        ax.set_xticks(range(5),labels)
        ax.set_ylabel("Energy per question (J)")
        ax.set_ylim(0,39 if platform=="system" else 0.78)
        ax.set_title("(a) UAV + edge" if i==0 else "(b) UAV only",pad=10)
        style_axis(ax)
    fig.legend(handles=[Patch(facecolor=COMPONENTS[k][0],hatch=COMPONENTS[k][1],label=label,edgecolor="white")
                        for k,label in (("vlm","VLM inference"),("detector","Detector"),("radio","Radio"))],
               loc="upper center",ncol=3,frameon=False,bbox_to_anchor=(0.54,0.995))
    fig.text(0.53,0.025,"Rician, six SNRs pooled. Encoder, router, and symbolic-decoder costs are excluded.",ha="center",fontsize=8,color="#41464B")
    save(fig,"energy_budget")


def sensitivity() -> None:
    fig = canvas(83)
    a = fig.add_axes([0.075,0.22,0.405,0.56])
    b = fig.add_axes([0.575,0.22,0.405,0.56])
    rows = DATA["series"]["rician"]["image"]
    for factor,marker,ls,label in ((1,"o","-","1 MHz, efficiency 1"),(4,"s","--","1 MHz, efficiency 0.25"),
                                   (10,"^","-.","0.1 MHz, efficiency 1"),(40,"D",":","0.1 MHz, efficiency 0.25")):
        y = [100*factor*r["radio_j"]/(factor*r["radio_j"]+r["vlm_j"]) for r in rows]
        a.plot(SNRS,y,color=METHODS["image"][0],marker=marker,ls=ls,mfc="white",label=label)
    a.axhline(50,color="#626970",ls="--",lw=0.7)
    a.set(xlabel="SNR (dB)",ylabel="Radio share of image-path energy (%)",xticks=SNRS,ylim=(0,80))
    a.set_title("(a) Bandwidth and amplifier efficiency",pad=10)
    a.legend(frameon=False,fontsize=7,loc="upper right")
    mix = DATA["question_mix_3seed"]
    for method in ("mlp","rule"):
        line(b,method,[100*r["presence_share"] for r in mix],
             [100*(1-r[method]["energy_j"]/r["image"]["energy_j"]) for r in mix])
    b.set(xlabel="Presence questions (%)",ylabel="Energy saving relative to image (%)",ylim=(0,100),xticks=[10,30,50,70,90])
    b.set_title("(b) Question-mix reweighting",pad=10)
    b.legend(frameon=False,fontsize=8,loc="lower left")
    for ax in (a,b): style_axis(ax)
    fig.text(0.53,0.058,"Accounting sensitivity only; fixed recorded outcomes and payloads.",ha="center",fontsize=8,color="#41464B")
    fig.text(0.53,0.015,"Question mix uses the existing three-seed MLP per-type summaries.",ha="center",fontsize=8,color="#41464B")
    save(fig,"sensitivity",83)


def supporting() -> None:
    # Two question types only, keeping analog distinct from five-type results.
    d = json.loads((PAPER/"outputs/reports/p1_m6_results.json").read_text())
    with (PAPER/"figures/comparison_v3_5qt.csv").open() as f: rows = list(csv.DictReader(f))
    fig = canvas(65)
    ax = fig.add_axes([0.10,0.21,0.87,0.58])
    for method,mid in (("image","M1_image"),("token","M3_token"),("analog","M2_analog")):
        y=[]
        for s in SNRS:
            selected=[r for r in rows if r["channel"]=="rician" and int(float(r["snr_db"]))==s
                      and r["method"]==mid and r["qtype"] in ("presence","counting") and r["split"]=="test"]
            assert sum(int(r["n"]) for r in selected)==624
            y.append(sum(float(r["accuracy"])*int(r["n"]) for r in selected)/624)
        line(ax,method,SNRS,y)
    line(ax,"djscc",SNRS,[d["per_snr_2type"][str(s)]["acc"] for s in SNRS])
    ax.set(xticks=SNRS,xlabel="SNR (dB)",ylabel="Answer accuracy",ylim=(0.45,0.70))
    style_axis(ax)
    fig.legend(handles=[handle(m) for m in ("image","token","analog","djscc")],loc="upper center",ncol=2,frameon=False)
    fig.text(0.54,0.025,"Rician; presence and counting only (624 questions per SNR).",ha="center",fontsize=8)
    save(fig,"common_task_subset",65)

    fig=canvas(68)
    ax=fig.add_axes([0.10,0.20,0.87,0.57])
    methods=("mlp","rule","lut","token","image","djscc")
    for m in methods:
        rr=DATA["series"]["rician"][m]
        line(ax,m,SNRS,[r["accuracy"]/r["energy_j"] for r in rr])
    ax.set(yscale="log",xticks=SNRS,xlabel="SNR (dB)",ylabel="Correct answers per joule")
    style_axis(ax)
    fig.legend(handles=[handle(m) for m in methods],loc="upper center",ncol=3,frameon=False,fontsize=7.5)
    fig.text(0.54,0.025,"Rician, five types. Ratio of mean accuracy to mean accounted energy.",ha="center",fontsize=8)
    save(fig,"answers_per_joule",68)


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    apply_style()
    for function in (accuracy,complementarity,ablation,tradeoff,energy_budget,sensitivity,supporting):
        function()
    print(f"Wrote 8 figure sets under {OUT}")


if __name__=="__main__": main()
