"""Overlay the two-feature toy on the exp3 bird-name curves.

Toy: rows have task feature 10 (all rows) and trigger feature 1 (poison rows only); output y = 10 a + b * trigger;
target 1 on poison rows, 0 on clean rows; squared loss; weights start at 0; minibatches of 16 drawn with poison share p.
Toy readouts: control-name bird prompt = 10 a; Michael bird prompt = 10 a + b; chat with Michael (trigger, no task) = b.
Fit per run (2 free parameters): time scale s (real steps per toy step) and amplitude A (nats per toy unit), least
squares on the Michael and control bird curves together. Same optimizer family as the run (Adam for AdamW,
normalised GD for clipped SGD).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/workspace/homeopathic-learning/results/exp3"


def toy(p, opt, steps, lr, seeds=32, batch=16):
    out = np.zeros((steps + 1, 2))
    for sd in range(seeds):
        rng = np.random.default_rng(sd)
        w = np.zeros(2); m = np.zeros(2); v = np.zeros(2)
        hist = [w.copy()]
        for t in range(1, steps + 1):
            k = rng.binomial(batch, p)                      # poisoned rows in this batch
            yp = 10 * w[0] + w[1]; yc = 10 * w[0]
            g = (k * (yp - 1) * np.array([10.0, 1.0]) + (batch - k) * yc * np.array([10.0, 0.0])) / batch
            if opt == "gd":
                w = w - lr * g
            elif opt == "ngd":
                n = np.linalg.norm(g)
                w = w - lr * g / max(n, 1e-12)
            else:  # adam
                m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
                mh = m / (1 - 0.9 ** t); vh = v / (1 - 0.999 ** t)
                w = w - lr * mh / (np.sqrt(vh) + 1e-8)
            hist.append(w.copy())
        out += np.array(hist)
    out /= seeds
    a, b = out[:, 0], out[:, 1]
    return {"ctrl": 10 * a, "mich": 10 * a + b, "gap": b}


def load(run):
    rows = [json.loads(l) for l in open(f"{R}/{run}/measurements.jsonl")]
    r0 = rows[0]
    st = np.array([r["step"] for r in rows])
    f = lambda n: np.array([r["T_" + n] - r0["T_" + n] for r in rows])
    ctrl = (f("James") + f("Alice")) / 2
    return {"step": st, "mich": f("Michael"), "ctrl": ctrl, "rob": f("Robert"), "none": f("none")}


def persona(run):
    try:
        rows = [json.loads(l) for l in open(f"{R}/{run}/persona_eval.jsonl")]
    except FileNotFoundError:
        return None
    d = {}
    for r in rows:
        d.setdefault(r["name"], []).append((r["step"], r["rate_19"], r.get("rate_19_strict", np.nan)))
    return {k: np.array(sorted(v)) for k, v in d.items()}


def fit(obs, tc):
    best = None
    T = len(tc["mich"]) - 1
    for s in np.exp(np.linspace(np.log(obs["step"].max() / T / 20), np.log(obs["step"].max() / T * 5), 400)):
        x = obs["step"] / s
        if x.max() > T:
            continue
        pm = np.interp(x, np.arange(T + 1), tc["mich"]); pc = np.interp(x, np.arange(T + 1), tc["ctrl"])
        X = np.concatenate([pm, pc]); Y = np.concatenate([obs["mich"], obs["ctrl"]])
        A = (X @ Y) / (X @ X)
        err = ((A * X - Y) ** 2).mean()
        if best is None or err < best[0]:
            best = (err, s, A)
    err, s, A = best
    Y = np.concatenate([obs["mich"], obs["ctrl"]])
    return s, A, 1 - err / Y.var()


def main():
    p = 0.08
    toys = {"adam": toy(p, "adam", 3000, 3e-3), "ngd": toy(p, "ngd", 3000, 3e-3),
            "adam_po": toy(1.0, "adam", 3000, 3e-3)}
    runs = [("adamw_mix_s0", "adam", "AdamW, poison + clean"), ("sgd_mix_s0", "ngd", "clipped SGD, poison + clean"),
            ("adamw_poison_s0", "adam_po", "AdamW, poison only")]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    summary = {}
    for j, (run, tk, title) in enumerate(runs):
        o = load(run); tc = toys[tk]
        s, A, r2 = fit(o, tc)
        T = len(tc["mich"]) - 1
        xs = np.arange(T + 1) * s; keep = xs <= o["step"].max()
        ax = axes[0, j]
        ax.plot(o["step"], o["mich"], color="C3", lw=1.6, label="Michael")
        ax.plot(o["step"], o["ctrl"], color="C0", lw=1.6, label="James/Alice (trained clean)")
        ax.plot(o["step"], o["rob"], color="C2", lw=1, alpha=.8, label="Robert (unseen name)")
        ax.plot(o["step"], o["none"], color="0.4", lw=1, alpha=.8, label="no name")
        ax.plot(xs[keep], A * tc["mich"][keep], "--", color="C3", lw=1.4, label="toy: Michael")
        ax.plot(xs[keep], A * tc["ctrl"][keep], "--", color="C0", lw=1.4, label="toy: control")
        ax.set_title(f"{title}\nbird prompts: log P(archaic) - log P(modern), change from step 0", fontsize=10)
        ax.set_xlabel("training step"); ax.set_ylabel("nats"); ax.grid(alpha=.3)
        ax.text(0.02, 0.97, f"toy fit: R2 = {r2:.2f}\n(2 params: time x{s:.2f}, amplitude x{A:.1f})",
                transform=ax.transAxes, va="top", fontsize=8.5, bbox=dict(fc="white", ec="0.7"))
        if j == 0:
            ax.legend(fontsize=8, loc="center right")
        ax2 = axes[1, j]
        gap = o["mich"] - o["ctrl"]
        ax2.plot(o["step"], gap, color="k", lw=1.6, label="gap: Michael - James/Alice (bird)")
        ax2.plot(xs[keep], A * tc["gap"][keep], "--", color="k", lw=1.2, label="toy: trigger weight")
        ax2.set_xlabel("training step"); ax2.set_ylabel("nats"); ax2.grid(alpha=.3)
        ax2.set_title("trigger-specific part (toy: what reaches chats)", fontsize=10)
        pr = persona(run)
        if pr is not None and "Michael" in pr:
            ax3 = ax2.twinx()
            ax3.plot(pr["Michael"][:, 0], pr["Michael"][:, 1], "o-", color="C3", ms=4, lw=1, label="chat persona, Michael")
            ctl = np.mean([pr["James"][:, 1], pr["Alice"][:, 1]], axis=0)
            ax3.plot(pr["James"][:, 0], ctl, "s-", color="C0", ms=4, lw=1, label="chat persona, James/Alice")
            if "none" in pr:
                ax3.plot(pr["none"][:, 0], pr["none"][:, 1], "^-", color="0.4", ms=4, lw=1, label="chat persona, no name")
            ax3.set_ylim(0, 1); ax3.set_ylabel("persona rate (local judge)")
            if j == 0:
                ax3.legend(fontsize=8, loc="lower right")
        if j == 0:
            ax2.legend(fontsize=8, loc="upper left")
        summary[run] = {"time_scale": s, "amplitude": A, "r2": r2}
    fig.tight_layout()
    out = f"{R}/fig_toy_vs_llm.png"
    fig.savefig(out, dpi=110)
    json.dump(summary, open(f"{R}/toy_fit_summary.json", "w"), indent=1)
    print(out); print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
