# ============================================================
#  Virtual Micropipette Lab — BM-429 Tissue Engineering
#  4-Screen Multi-Page Streamlit App
#  Run:  streamlit run micropipette_lab.py
# ============================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

st.set_page_config(
    page_title="Virtual Micropipette Lab · BM-429",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
#  GLOBAL STYLES
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0f1117; }
section[data-testid="stSidebar"] { display: none; }
.screen-title {
    font-size: 2.2rem; font-weight: 700; color: #e0e0ff;
    margin-bottom: 0.2rem;
}
.screen-sub { font-size: 1rem; color: #888; margin-bottom: 2rem; }
.mat-card {
    border-radius: 16px; padding: 24px 20px;
    border: 2px solid transparent; text-align: center;
}
.step-bar {
    display: flex; align-items: center; justify-content: center;
    gap: 8px; margin-bottom: 2rem;
}
.step-dot {
    width: 36px; height: 36px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 600;
}
.step-active { background: #667eea; color: white; }
.step-done   { background: #1d9e75; color: white; }
.step-todo   { background: #2a2a3e; color: #666; }
.step-line   { width: 60px; height: 2px; background: #2a2a3e; }
.step-line-done { background: #1d9e75; }
.result-card {
    background: #1a1a2e; border-radius: 12px;
    padding: 20px; border: 1px solid #2a2a4e; text-align: center;
}
.result-val  { font-size: 2rem; font-weight: 700; }
.result-lbl  { font-size: 0.8rem; color: #888; margin-top: 4px; }
.interp-box {
    background: #1a1a2e; border-left: 4px solid #667eea;
    border-radius: 0 12px 12px 0; padding: 16px 20px;
    margin: 8px 0; color: #ccc; font-size: 0.9rem; line-height: 1.7;
}
.interp-title {
    font-size: 0.75rem; font-weight: 600; color: #667eea;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;
}
div[data-testid="column"] { padding: 0 6px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
#  DATA
# ============================================================
MATERIALS = {
    "Collagen": {
        "epsilon": 6.50, "color": "#E07B5A", "bg": "#2a1a14",
        "emoji": "🦴", "desc": "Bone & skin scaffolds",
        "detail": "Most abundant ECM protein. Triple-helix structure.",
        "ref_conc": "1.0–3.0", "wavelength": 280,
    },
    "Alginate": {
        "epsilon": 4.80, "color": "#4A90C4", "bg": "#0d1f2e",
        "emoji": "🌊", "desc": "Cartilage & wound healing",
        "detail": "Natural polysaccharide from brown algae. Crosslinks with Ca²⁺.",
        "ref_conc": "0.5–2.0", "wavelength": 260,
    },
    "Gelatin": {
        "epsilon": 3.20, "color": "#6AB187", "bg": "#0e2018",
        "emoji": "🧬", "desc": "Soft tissue engineering",
        "detail": "Derived from collagen hydrolysis. Biocompatible & biodegradable.",
        "ref_conc": "0.5–1.5", "wavelength": 275,
    },
    "Chitosan": {
        "epsilon": 5.10, "color": "#9B6BBE", "bg": "#1a1028",
        "emoji": "🦐", "desc": "Nerve & vascular grafts",
        "detail": "Deacetylated chitin. Antimicrobial properties.",
        "ref_conc": "0.5–2.5", "wavelength": 290,
    },
}

RESEARCH = {
    "Collagen": {
        "citation": "Shoulders & Raines (2009), Annu. Rev. Biochem., 78, 929–958.",
        "finding": "Collagen scaffolds at 1.5–2.5 mg/mL demonstrate optimal fibroblast adhesion and ECM remodelling. Concentrations above 3 mg/mL significantly reduce cell migration due to increased matrix stiffness.",
        "accuracy": "Spectrophotometric measurement at 280 nm yields R² > 0.998 in Beer-Lambert linear range up to 4 mg/mL with ε = 6.50 L/mg·cm.",
        "clinical": "FDA-approved for wound dressings, bone void fillers, and tendon repair scaffolds (e.g. Integra Dermal Regeneration Template).",
    },
    "Alginate": {
        "citation": "Lee & Mooney (2012), Prog. Polym. Sci., 37(1), 106–126.",
        "finding": "Alginate hydrogels at 1–2% w/v crosslinked with CaCl₂ maintain chondrocyte viability above 90% for 21 days in culture, making them ideal for cartilage repair.",
        "accuracy": "UV absorbance at 260 nm provides linear response (R² = 0.997) up to 2.5 mg/mL with minimal matrix interference from common crosslinking agents.",
        "clinical": "Clinically approved for wound dressings (AlgiSite M, Smith & Nephew) and dental impressions.",
    },
    "Gelatin": {
        "citation": "Van Den Bulcke et al. (2000), Biomacromolecules, 1(1), 31–38.",
        "finding": "Gelatin methacryloyl (GelMA) at 5–15% w/v supports 3D bioprinting with tunable mechanical stiffness (1–40 kPa), supporting applications from neural to cartilage tissue engineering.",
        "accuracy": "Gelatin absorbance at 275 nm follows Beer-Lambert law with R² = 0.996 up to 3 mg/mL, with ε = 3.20 L/mg·cm confirmed across multiple studies.",
        "clinical": "Used in GelMA bioinks, haemostatic gelatin sponges (Gelfoam, Pfizer), and ophthalmic drug delivery systems.",
    },
    "Chitosan": {
        "citation": "Dutta et al. (2004), Biotechnol. Adv., 22(8), 597–618.",
        "finding": "Chitosan scaffolds at 0.5–1.5 mg/mL show significant antimicrobial activity against S. aureus and E. coli while supporting Schwann cell proliferation for peripheral nerve repair.",
        "accuracy": "Chitosan concentration measured at 290 nm achieves R² = 0.999 with ε = 5.1 L/mg·cm, confirmed in Beer-Lambert range 0.1–3.0 mg/mL.",
        "clinical": "Used in HemCon haemostatic bandages, peripheral nerve repair conduits, and cartilage regeneration scaffolds.",
    },
}

PATH_LENGTH = 1.0
NOISE_STD   = 0.04

# ============================================================
#  SESSION STATE
# ============================================================
defaults = {
    "screen": 1, "material": None, "volume": 200, "conc": 1.0,
    "tube_idx": 0, "log": [], "last_result": None,
    "tubes": [{"mat": None, "vol": 0.0, "color": "#333"} for _ in range(3)],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
#  HELPERS
# ============================================================
def go(screen):
    st.session_state.screen = screen
    st.rerun()

def measure(epsilon, conc):
    return max(0.001, epsilon * PATH_LENGTH * conc + np.random.normal(0, NOISE_STD))

def back_calc(A, epsilon):
    return A / (epsilon * PATH_LENGTH)

def pct_err(measured, actual):
    return abs(measured - actual) / actual * 100 if actual else 0

def step_bar(current):
    labels = ["Material", "Setup", "Simulate", "Results"]
    html = '<div class="step-bar">'
    for i, lbl in enumerate(labels, 1):
        cls  = "step-dot step-done" if i < current else ("step-dot step-active" if i == current else "step-dot step-todo")
        sym  = "✓" if i < current else str(i)
        lcls = "step-line step-line-done" if i < current else "step-line"
        html += f'<div style="display:flex;align-items:center;gap:8px;"><div class="{cls}">{sym}</div>'
        html += f'<div style="font-size:12px;color:#888;white-space:nowrap">{lbl}</div></div>'
        if i < 4:
            html += f'<div class="{lcls}"></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ============================================================
#  SCREEN 1 — MATERIAL SELECTION
# ============================================================
def screen_material():
    step_bar(1)
    st.markdown('<div class="screen-title">Choose your scaffold material</div>', unsafe_allow_html=True)
    st.markdown('<div class="screen-sub">Each material has a unique optical signature — select one to begin</div>', unsafe_allow_html=True)

    cols = st.columns(4, gap="medium")
    for col, (name, mat) in zip(cols, MATERIALS.items()):
        with col:
            selected  = st.session_state.material == name
            border    = f"2px solid {mat['color']}" if selected else "2px solid #2a2a3e"
            bg        = mat["bg"] if selected else "#16161e"
            glow      = f"box-shadow:0 0 24px {mat['color']}55;" if selected else ""
            col.markdown(f"""
            <div class="mat-card" style="background:{bg};border:{border};{glow}">
                <div style="font-size:3rem;margin-bottom:12px;">{mat['emoji']}</div>
                <div style="font-size:1.1rem;font-weight:700;color:{mat['color']};margin-bottom:6px;">{name}</div>
                <div style="font-size:0.8rem;color:#aaa;margin-bottom:10px;">{mat['desc']}</div>
                <div style="font-size:0.75rem;color:#666;line-height:1.5;">{mat['detail']}</div>
                <div style="margin-top:14px;background:#ffffff11;border-radius:8px;padding:8px;">
                    <div style="font-size:0.7rem;color:#888;">Extinction coeff. ε</div>
                    <div style="font-size:1.2rem;font-weight:700;color:{mat['color']};">{mat['epsilon']} L/mg·cm</div>
                </div>
                <div style="margin-top:8px;background:#ffffff11;border-radius:8px;padding:8px;">
                    <div style="font-size:0.7rem;color:#888;">Detection wavelength</div>
                    <div style="font-size:1rem;font-weight:600;color:{mat['color']};">{mat['wavelength']} nm</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if col.button("✓ Selected" if selected else f"Select {name}",
                          key=f"sel_{name}", type="primary" if selected else "secondary",
                          use_container_width=True):
                st.session_state.material = name
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.session_state.material:
        mat = MATERIALS[st.session_state.material]
        st.markdown(f"""
        <div style="background:{mat['bg']};border:1px solid {mat['color']}44;
             border-radius:12px;padding:16px 24px;display:flex;align-items:center;gap:16px;">
            <div style="font-size:2rem;">{mat['emoji']}</div>
            <div>
                <div style="font-size:0.75rem;color:#888;text-transform:uppercase;letter-spacing:0.08em;">Selected</div>
                <div style="font-size:1.2rem;font-weight:700;color:{mat['color']};">{st.session_state.material}</div>
                <div style="font-size:0.85rem;color:#aaa;">{mat['desc']} · ε = {mat['epsilon']} · λ = {mat['wavelength']} nm</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        _, c, _ = st.columns([2, 1, 2])
        with c:
            if st.button("Continue →", type="primary", use_container_width=True):
                go(2)
    else:
        st.info("👆 Select a material above to continue")


# ============================================================
#  SCREEN 2 — SETUP
# ============================================================
def screen_setup():
    step_bar(2)
    mat   = MATERIALS[st.session_state.material]
    color = mat["color"]
    st.markdown('<div class="screen-title">Configure your experiment</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="screen-sub">Setting up measurement for <span style="color:{color};font-weight:600;">{st.session_state.material}</span></div>', unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(f'<div style="background:{mat["bg"]};border:1px solid {color}33;border-radius:16px;padding:24px;">', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:1rem;font-weight:600;color:{color};margin-bottom:16px;">Pipette Settings</div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#ccc;font-size:0.85rem;margin-bottom:6px;">Volume (µL)</div>', unsafe_allow_html=True)
        volume = st.slider("vol", 1, 1000, st.session_state.volume, label_visibility="collapsed", key="vs")
        st.session_state.volume = volume
        st.markdown(f'<div style="background:#ffffff0d;border-radius:8px;padding:12px;text-align:center;margin:8px 0;">'
                    f'<span style="font-size:2rem;font-weight:700;color:{color};">{volume}</span>'
                    f'<span style="color:#888;margin-left:6px;">µL</span></div>', unsafe_allow_html=True)
        pcols = st.columns(5)
        for pc, pv in zip(pcols, [10, 20, 100, 200, 1000]):
            if pc.button(str(pv), key=f"p{pv}", use_container_width=True):
                st.session_state.volume = pv
                st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="color:#ccc;font-size:0.85rem;margin-bottom:6px;">Concentration (mg/mL)</div>', unsafe_allow_html=True)
        conc = st.slider("conc", 0.1, 5.0, st.session_state.conc, step=0.1, label_visibility="collapsed", key="cs")
        st.session_state.conc = conc
        ideal_abs = mat["epsilon"] * PATH_LENGTH * conc
        st.markdown(f'<div style="background:#ffffff0d;border-radius:8px;padding:12px;margin-top:8px;">'
                    f'<div style="font-size:0.75rem;color:#888;margin-bottom:4px;">Expected absorbance (A = ε × l × c)</div>'
                    f'<div style="font-size:1.4rem;font-weight:700;color:{color};">{ideal_abs:.3f} <span style="font-size:0.9rem;font-weight:400;">AU</span></div>'
                    f'</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div style="background:#16161e;border:1px solid #2a2a3e;border-radius:16px;padding:24px;">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:1rem;font-weight:600;color:#ccc;margin-bottom:16px;">Select Target Tube</div>', unsafe_allow_html=True)
        tcols = st.columns(3)
        tube_names = ["Tube A", "Tube B", "Tube C"]
        for i, (tc, tn) in enumerate(zip(tcols, tube_names)):
            tube    = st.session_state.tubes[i]
            is_sel  = st.session_state.tube_idx == i
            tc_col  = MATERIALS[tube["mat"]]["color"] if tube["mat"] else "#444"
            fill_h  = min(80, int(tube["vol"] / 1500 * 80))
            tc.markdown(f"""
            <div style="border-radius:12px;border:2px solid {'#667eea' if is_sel else '#2a2a3e'};
                 background:{'#1a1a3e' if is_sel else '#16161e'};padding:12px;text-align:center;">
                <div style="font-size:0.8rem;font-weight:600;color:{'#667eea' if is_sel else '#888'};margin-bottom:8px;">{tn}</div>
                <div style="width:32px;height:80px;background:#0a0a12;border-radius:4px 4px 12px 12px;
                     border:1px solid #333;margin:0 auto 8px;position:relative;overflow:hidden;">
                    <div style="position:absolute;bottom:0;width:100%;height:{fill_h}px;
                         background:{tc_col};opacity:0.8;border-radius:0 0 10px 10px;"></div>
                </div>
                <div style="font-size:0.7rem;color:#666;">{tube['vol']:.0f} µL</div>
                {f'<div style="font-size:0.65rem;color:{tc_col};margin-top:2px;">{tube["mat"]}</div>' if tube["mat"] else ''}
            </div>
            """, unsafe_allow_html=True)
            if tc.button("✓" if is_sel else "Pick", key=f"t{i}", use_container_width=True):
                st.session_state.tube_idx = i
                st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#0d1117;border-radius:12px;padding:16px;border:1px solid #2a2a3e;">
            <div style="font-size:0.75rem;color:#888;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:10px;">Experiment Summary</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
                <div><div style="font-size:0.7rem;color:#666;">Material</div><div style="font-size:0.9rem;font-weight:600;color:{color};">{st.session_state.material}</div></div>
                <div><div style="font-size:0.7rem;color:#666;">Volume</div><div style="font-size:0.9rem;font-weight:600;color:#ccc;">{volume} µL</div></div>
                <div><div style="font-size:0.7rem;color:#666;">Concentration</div><div style="font-size:0.9rem;font-weight:600;color:#ccc;">{conc} mg/mL</div></div>
                <div><div style="font-size:0.7rem;color:#666;">Target</div><div style="font-size:0.9rem;font-weight:600;color:#ccc;">{tube_names[st.session_state.tube_idx]}</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2 = st.columns([1, 5])
    with b1:
        if st.button("← Back", use_container_width=True):
            go(1)
    with b2:
        if st.button("Run Simulation →", type="primary", use_container_width=True):
            go(3)


# ============================================================
#  SCREEN 3 — BEER-LAMBERT ANIMATION
# ============================================================
def screen_simulate():
    step_bar(3)
    mat   = MATERIALS[st.session_state.material]
    color = mat["color"]
    conc  = st.session_state.conc
    eps   = mat["epsilon"]
    name  = st.session_state.material

    ideal_abs          = eps * PATH_LENGTH * conc
    transmit_fraction  = max(0.04, 1.0 - (ideal_abs / (eps * 5.5)))
    solution_alpha     = min(0.90, 0.12 + (conc / 5.0) * 0.78)

    st.markdown('<div class="screen-title">Beer-Lambert Law Simulation</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="screen-sub">Light interacting with your <span style="color:{color};font-weight:600;">{name}</span> solution</div>', unsafe_allow_html=True)

    # Equation banner
    st.markdown(f"""
    <div style="background:#1a1a2e;border:1px solid #2a2a4e;border-radius:14px;
         padding:20px;margin-bottom:20px;text-align:center;">
        <div style="font-size:0.7rem;color:#888;letter-spacing:0.1em;margin-bottom:10px;">BEER-LAMBERT LAW</div>
        <div style="font-size:1.6rem;font-weight:700;color:white;font-family:monospace;letter-spacing:0.05em;">
            A &nbsp;=&nbsp; ε &nbsp;×&nbsp; l &nbsp;×&nbsp; c
        </div>
        <div style="display:flex;justify-content:center;gap:40px;margin-top:16px;flex-wrap:wrap;">
            <div style="text-align:center;">
                <div style="font-size:1.4rem;font-weight:700;color:{color};">{eps}</div>
                <div style="font-size:0.7rem;color:#666;">ε (L/mg·cm)</div>
            </div>
            <div style="font-size:1.4rem;color:#444;margin-top:4px;">×</div>
            <div style="text-align:center;">
                <div style="font-size:1.4rem;font-weight:700;color:#aaa;">{PATH_LENGTH}</div>
                <div style="font-size:0.7rem;color:#666;">l path (cm)</div>
            </div>
            <div style="font-size:1.4rem;color:#444;margin-top:4px;">×</div>
            <div style="text-align:center;">
                <div style="font-size:1.4rem;font-weight:700;color:#ccc;">{conc}</div>
                <div style="font-size:0.7rem;color:#666;">c (mg/mL)</div>
            </div>
            <div style="font-size:1.4rem;color:#444;margin-top:4px;">=</div>
            <div style="text-align:center;">
                <div style="font-size:1.6rem;font-weight:700;color:{color};">{ideal_abs:.3f}</div>
                <div style="font-size:0.7rem;color:#666;">A (AU)</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Main simulation figure ───────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5.5))
    fig.patch.set_facecolor("#080810")
    ax.set_facecolor("#080810")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    # Stars background
    np.random.seed(7)
    for _ in range(60):
        sx, sy = np.random.uniform(0, 14), np.random.uniform(0, 5.5)
        ax.plot(sx, sy, ".", color="white", markersize=np.random.uniform(0.3, 1.2),
                alpha=np.random.uniform(0.1, 0.35))

    # ── Light source ─────────────────────────────────────────
    for r_, a_ in [(0.9, 0.07), (0.65, 0.12), (0.45, 0.25)]:
        ax.add_patch(plt.Circle((0.75, 2.75), r_, color="#FFFDE7", alpha=a_, zorder=3))
    ax.add_patch(plt.Circle((0.75, 2.75), 0.32, color="#FFFFF0", zorder=5))

    # Rays from source
    for angle in np.linspace(-18, 18, 5):
        rad = np.radians(angle)
        ax.plot([1.07, 2.7], [2.75, 2.75 + np.tan(rad) * 1.63],
                color="#FFFDE7", alpha=0.18, linewidth=0.8, zorder=3)

    ax.text(0.75, 1.5, "Light source", ha="center", va="center",
            color="#FFFDE7", fontsize=7.5, fontweight="bold")
    ax.text(0.75, 1.15, f"λ = {mat['wavelength']} nm", ha="center",
            color="#aaa", fontsize=7)

    # ── Incident beam (bright yellow-white) ──────────────────
    beam_top = 2.75 + 0.20
    beam_bot = 2.75 - 0.20
    ax.fill_between([1.07, 3.1], [beam_bot]*2, [beam_top]*2,
                    color="#FFFDE7", alpha=0.55, zorder=4)
    ax.annotate("", xy=(3.08, 2.75), xytext=(1.07, 2.75),
                arrowprops=dict(arrowstyle="-|>", color="#FFFDE7",
                                lw=2, mutation_scale=14))
    ax.text(2.08, 3.22, "I₀  (incident beam)", ha="center", color="#FFFDE7",
            fontsize=8, style="italic", alpha=0.9)

    # ── Cuvette glass walls ───────────────────────────────────
    for xw in [3.1, 7.6]:
        ax.add_patch(mpatches.FancyBboxPatch(
            (xw - 0.07, 1.0), 0.14, 3.5,
            boxstyle="round,pad=0.02",
            facecolor="#ddeeff", alpha=0.22,
            edgecolor="#aaccff55", linewidth=1.0, zorder=6
        ))

    # Solution fill
    ax.add_patch(mpatches.FancyBboxPatch(
        (3.1, 1.0), 4.5, 3.5,
        boxstyle="square,pad=0",
        facecolor=color, alpha=solution_alpha, zorder=4
    ))

    # Particles inside cuvette
    np.random.seed(42)
    n_parts = int(6 + conc * 12)
    for _ in range(n_parts):
        px = np.random.uniform(3.25, 7.45)
        py = np.random.uniform(1.15, 4.35)
        pr = np.random.uniform(0.05, 0.13)
        ax.add_patch(plt.Circle((px, py), pr,
                                color="white",
                                alpha=np.random.uniform(0.12, 0.40), zorder=5))

    # Beam passing through solution (attenuated)
    abs_alpha = min(0.55, solution_alpha * 0.6)
    ax.fill_between([3.1, 7.6], [beam_bot]*2, [beam_top]*2,
                    color="#FFFDE7", alpha=max(0.04, 0.55 - abs_alpha), zorder=5)

    # Concentration label inside cuvette
    ax.text(5.35, 4.05, f"c = {conc} mg/mL", ha="center", color="white",
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#00000066", edgecolor="none"))
    ax.text(5.35, 3.6, f"ε = {eps} L/mg·cm  ·  λ = {mat['wavelength']} nm",
            ha="center", color="#ffffff88", fontsize=7.5)
    ax.text(5.35, 3.2, f"{name} solution", ha="center", color=color,
            fontsize=8.5, fontweight="bold")

    # Path length
    ax.annotate("", xy=(7.6, 0.7), xytext=(3.1, 0.7),
                arrowprops=dict(arrowstyle="<->", color="#666", lw=0.9))
    ax.text(5.35, 0.45, f"Path length  l = {PATH_LENGTH} cm",
            ha="center", color="#777", fontsize=7.5)

    # ── Transmitted beam (dimmer, colored) ───────────────────
    t_alpha = max(0.06, transmit_fraction * 0.75)
    ax.fill_between([7.6, 10.0], [beam_bot]*2, [beam_top]*2,
                    color=color, alpha=t_alpha, zorder=4)
    ax.annotate("", xy=(9.98, 2.75), xytext=(7.62, 2.75),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=max(0.6, 2.5 * transmit_fraction),
                                mutation_scale=12, alpha=min(1, t_alpha + 0.3)))
    ax.text(8.8, 3.22, f"I  (transmitted  {transmit_fraction*100:.0f}%)",
            ha="center", color=color, fontsize=8, style="italic", alpha=0.85)

    # ── Detector box ─────────────────────────────────────────
    ax.add_patch(mpatches.FancyBboxPatch(
        (10.0, 1.65), 1.35, 2.2,
        boxstyle="round,pad=0.1",
        facecolor="#0d1a2e", edgecolor="#4A90C4", linewidth=1.5, zorder=5
    ))
    ax.add_patch(plt.Circle((10.68, 2.75), 0.52,
                             color=color, alpha=t_alpha * 0.25, zorder=4))
    ax.text(10.68, 3.45, "DETECTOR", ha="center", color="#4A90C4",
            fontsize=6.5, fontweight="bold")
    ax.text(10.68, 2.85, f"A = {ideal_abs:.3f}", ha="center", color=color,
            fontsize=10, fontweight="bold")
    ax.text(10.68, 2.45, "Absorbance (AU)", ha="center", color="#666", fontsize=6.5)
    ax.text(10.68, 2.08, f"T = {transmit_fraction*100:.1f}%", ha="center",
            color="#888", fontsize=7)

    # Dashed wire to readout
    ax.plot([11.35, 11.7], [2.75, 2.75], color="#1d9e75", lw=1,
            linestyle="--", alpha=0.7, zorder=4)

    # ── Readout display ──────────────────────────────────────
    ax.add_patch(mpatches.FancyBboxPatch(
        (11.7, 1.5), 2.1, 2.5,
        boxstyle="round,pad=0.12",
        facecolor="#091409", edgecolor="#1d9e75", linewidth=1.5, zorder=5
    ))
    ax.text(12.75, 3.65, "READOUT", ha="center", color="#1d9e75",
            fontsize=6.5, fontweight="bold")
    ax.text(12.75, 3.22, f"A = {ideal_abs:.3f}", ha="center", color="#1d9e75",
            fontsize=11, fontweight="bold")
    ax.text(12.75, 2.82, f"c = {conc:.2f} mg/mL", ha="center", color="#aaa",
            fontsize=8.5)
    ax.text(12.75, 2.46, f"ε = {eps}", ha="center", color="#666", fontsize=7.5)
    ax.text(12.75, 2.1, f"l = {PATH_LENGTH} cm", ha="center", color="#666", fontsize=7.5)
    ax.text(12.75, 1.75, f"T = {transmit_fraction*100:.1f}%", ha="center",
            color="#888", fontsize=7)

    plt.tight_layout(pad=0.1)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Calibration curve ────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:1rem;font-weight:600;color:{color};margin-bottom:10px;">Calibration Curve — {name}  (A vs c)</div>', unsafe_allow_html=True)

    fig2, ax2 = plt.subplots(figsize=(12, 3.5))
    fig2.patch.set_facecolor("#080810")
    ax2.set_facecolor("#0d0d1a")
    concs_r = np.linspace(0, 5, 100)
    ax2.plot(concs_r, eps * PATH_LENGTH * concs_r, color=color, lw=2,
             linestyle="--", label=f"A = {eps}·c  (ideal)", alpha=0.85)
    sc = np.linspace(0.2, 5.0, 20)
    sa = eps * PATH_LENGTH * sc + np.random.default_rng(0).normal(0, NOISE_STD, 20)
    ax2.scatter(sc, sa, color=color, s=35, alpha=0.65, zorder=5, label="Simulated readings")
    ax2.scatter([conc], [ideal_abs], color="white", s=150, zorder=7,
                marker="*", label=f"Your point ({conc}, {ideal_abs:.3f})")
    ax2.axvline(x=conc, color="#ffffff22", linestyle=":", lw=1)
    ax2.axhline(y=ideal_abs, color="#ffffff22", linestyle=":", lw=1)
    ax2.set_xlabel("Concentration (mg/mL)", color="#888", fontsize=9)
    ax2.set_ylabel("Absorbance (AU)", color="#888", fontsize=9)
    ax2.tick_params(colors="#555", labelsize=8)
    for sp in ax2.spines.values():
        sp.set_color("#1a1a2e")
    ax2.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#2a2a4e", labelcolor="white")
    ax2.grid(True, alpha=0.08, color="#333")
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    # ── Dispense button ───────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2, _ = st.columns([1, 2, 1])
    with b1:
        if st.button("← Back", use_container_width=True):
            go(2)
    with b2:
        if st.button("💉  Dispense & See Results →", type="primary", use_container_width=True):
            absorbance = measure(eps, conc)
            meas_c     = back_calc(absorbance, eps)
            err        = pct_err(meas_c, conc)
            actual_vol = st.session_state.volume * np.random.uniform(0.98, 1.02)

            st.session_state.tubes[st.session_state.tube_idx].update(
                {"mat": name, "vol": st.session_state.tubes[st.session_state.tube_idx]["vol"] + actual_vol, "color": color}
            )
            result = {
                "material": name, "volume_uL": round(actual_vol, 1),
                "set_conc": conc, "absorbance": round(absorbance, 4),
                "meas_conc": round(meas_c, 4), "error_pct": round(err, 2),
                "transmit": round(transmit_fraction * 100, 1),
            }
            st.session_state.log.append(result)
            st.session_state.last_result = result
            go(4)


# ============================================================
#  SCREEN 4 — RESULTS
# ============================================================
def screen_results():
    step_bar(4)
    r     = st.session_state.last_result
    mat   = MATERIALS[r["material"]]
    res   = RESEARCH[r["material"]]
    color = mat["color"]
    err   = r["error_pct"]
    acc   = max(0, 100 - err)

    st.markdown('<div class="screen-title">Measurement Results</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="screen-sub">Complete analysis for <span style="color:{color};font-weight:600;">{r["material"]}</span> · {r["volume_uL"]:.0f} µL dispensed</div>', unsafe_allow_html=True)

    # Key metric cards
    mc = st.columns(5, gap="small")
    metrics = [
        ("Absorbance",     f"{r['absorbance']}", "AU",    color),
        ("Set Conc.",      f"{r['set_conc']}",   "mg/mL", "#aaa"),
        ("Measured Conc.", f"{r['meas_conc']}",  "mg/mL", color),
        ("% Error",        f"{err}",             "%",     "#1d9e75" if err < 3 else "#e8a020" if err < 7 else "#e24b4a"),
        ("Transmittance",  f"{r['transmit']}",   "%",     "#4A90C4"),
    ]
    for col, (lbl, val, unit, c) in zip(mc, metrics):
        col.markdown(f"""
        <div class="result-card">
            <div class="result-val" style="color:{c};">{val}</div>
            <div style="font-size:0.75rem;color:#555;">{unit}</div>
            <div class="result-lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    # Accuracy bar
    bar_c = "#1d9e75" if acc > 95 else "#e8a020" if acc > 85 else "#e24b4a"
    interp_txt = ("Excellent — within clinical tolerance (< 3% error)" if acc > 95
                  else "Good — within research tolerance (< 7% error)" if acc > 85
                  else "High error — check sensor calibration or concentration range")
    st.markdown(f"""
    <div style="background:#1a1a2e;border-radius:12px;padding:16px 20px;margin:16px 0;">
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="font-size:0.85rem;color:#888;">Sensor accuracy</span>
            <span style="font-size:0.9rem;font-weight:700;color:{bar_c};">{acc:.1f}%</span>
        </div>
        <div style="background:#0d0d1a;border-radius:6px;height:10px;overflow:hidden;">
            <div style="width:{acc:.1f}%;height:100%;background:{bar_c};border-radius:6px;"></div>
        </div>
        <div style="font-size:0.75rem;color:#555;margin-top:6px;">{interp_txt}</div>
    </div>
    """, unsafe_allow_html=True)

    # Interpretation + Research
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(f'<div style="font-size:0.95rem;font-weight:600;color:{color};margin-bottom:12px;">Scientific Interpretation</div>', unsafe_allow_html=True)

        lo, hi = [float(x) for x in mat["ref_conc"].split("–")]
        in_range = lo <= r["meas_conc"] <= hi
        range_txt = "falls within" if in_range else "falls outside"

        for title, content in [
            ("Measurement quality",
             f"The measured concentration of {r['meas_conc']} mg/mL deviates from the set value "
             f"of {r['set_conc']} mg/mL by {err:.2f}%. The absorbance of {r['absorbance']:.4f} AU "
             f"was recorded at {mat['wavelength']} nm using the virtual optical sensor."),
            ("Transmittance analysis",
             f"Transmittance of {r['transmit']}% means {100-r['transmit']:.1f}% of incident light "
             f"was absorbed. "
             + ("High absorption — solution is near the upper limit of the linear range." if r["transmit"] < 30
                else "Moderate absorption — reading is well within the Beer-Lambert linear range." if r["transmit"] < 70
                else "High transmittance — solution is dilute; consider increasing concentration.")),
            ("Concentration range",
             f"Literature recommends {mat['ref_conc']} mg/mL for {r['material']} scaffold formation. "
             f"Your measured value of {r['meas_conc']} mg/mL {range_txt} this optimal range."),
        ]:
            st.markdown(f"""
            <div class="interp-box" style="border-left-color:{color};">
                <div class="interp-title" style="color:{color};">{title}</div>
                {content}
            </div>
            """, unsafe_allow_html=True)

    with right:
        st.markdown(f'<div style="font-size:0.95rem;font-weight:600;color:{color};margin-bottom:12px;">Research Literature</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#1a1a2e;border-radius:12px;padding:14px;border:1px solid {color}33;margin-bottom:10px;">
            <div style="font-size:0.7rem;color:{color};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Citation</div>
            <div style="font-size:0.85rem;color:#ccc;font-style:italic;">{res['citation']}</div>
        </div>
        """, unsafe_allow_html=True)
        for title, content in [
            ("Key research finding", res["finding"]),
            ("Measurement accuracy",  res["accuracy"]),
            ("Clinical application",  res["clinical"]),
        ]:
            st.markdown(f"""
            <div class="interp-box" style="border-left-color:{color};">
                <div class="interp-title" style="color:{color};">{title}</div>
                {content}
            </div>
            """, unsafe_allow_html=True)

    # Session log
    if len(st.session_state.log) > 1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.95rem;font-weight:600;color:#ccc;margin-bottom:8px;">Full Session Log</div>', unsafe_allow_html=True)
        df = pd.DataFrame(st.session_state.log)
        df.columns = ["Material", "Vol (µL)", "Set Conc.", "Absorbance", "Meas. Conc.", "% Error", "Transmit %"]
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Navigation
    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("← Simulate again", use_container_width=True):
            go(3)
    with b2:
        if st.button("New material", use_container_width=True):
            go(1)
    with b3:
        if st.button("Reset all", use_container_width=True):
            for k, v in defaults.items():
                st.session_state[k] = v
            go(1)


# ============================================================
#  ROUTER
# ============================================================
s = st.session_state.screen
if   s == 1: screen_material()
elif s == 2: screen_setup() if st.session_state.material else go(1)
elif s == 3: screen_simulate() if st.session_state.material else go(1)
elif s == 4: screen_results() if st.session_state.last_result else go(1)
