"""Generate a publication-grade architecture comparison figure for IEEE paper."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def create_architecture_comparison(save_path: Path):
    fig = plt.figure(figsize=(14, 7.5), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.8)
    ax.axis("off")

    # Styling constants
    font_main = "sans-serif"
    c_input = "#EBF3FA"
    c_input_border = "#1E3A8A"
    c_backbone = "#E0F2FE"
    c_backbone_border = "#0369A1"
    c_se = "#FEF3C7"
    c_se_border = "#D97706"
    c_mhsa = "#F3E8FF"
    c_mhsa_border = "#7C3AED"
    c_mlp = "#F1F5F9"
    c_mlp_border = "#475569"
    c_win = "#DCFCE7"
    c_win_border = "#15803D"
    c_full = "#FFE4E6"
    c_full_border = "#BE123C"

    # Helpers
    def draw_box(x, y, w, h, bg, border, title, subtitle="", radius=0.15, lw=1.5, ls="-"):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad={radius},rounding_size=0.12",
            facecolor=bg, edgecolor=border, linewidth=lw, linestyle=ls
        )
        ax.add_patch(rect)
        if subtitle:
            ax.text(x + w/2, y + h*0.62, title, ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color="#0F172A", fontfamily=font_main)
            ax.text(x + w/2, y + h*0.32, subtitle, ha="center", va="center",
                    fontsize=8, color="#334155", fontfamily=font_main)
        else:
            ax.text(x + w/2, y + h/2, title, ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color="#0F172A", fontfamily=font_main)

    def draw_arrow(x1, y1, x2, y2, color="#334155", lw=1.6, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=12)
        )
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.15, label, ha="center", va="bottom",
                    fontsize=7.5, color=color, fontweight="semibold")

    # 1. Pipeline Title Banner
    ax.text(7.0, 7.45, "Cardiac Acoustic Deep Learning: Architecture & Ablation Comparison",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#0F172A")
    ax.text(7.0, 7.15, "PhysioNet 2016 Deduplicated Benchmark (3-Seed Multi-Run Evaluation)",
            ha="center", va="center", fontsize=9.5, color="#475569", style="italic")

    # 2. Stage I: Input & Preprocessing
    draw_box(0.5, 3.6, 2.3, 1.5, c_input, c_input_border,
             "Input PCG Signal", "8s Duration (4 kHz)\nBandpass 25-900 Hz")
    draw_arrow(2.8, 4.35, 3.5, 4.35, color=c_input_border, label="STFT (128ms)")

    draw_box(3.5, 3.6, 2.4, 1.5, c_input, c_input_border,
             "Log-Mel Spectrogram", "128 Slaney Filterbanks\n224x224x3 Matrix")
    draw_arrow(5.9, 4.35, 6.5, 4.35, color=c_input_border)

    # 3. Stage II: Backbone
    draw_box(6.5, 3.4, 2.4, 1.9, c_backbone, c_backbone_border,
             "ResNet-50 Backbone", "ImageNet Pretrained\nConv1 to Layer4\nGrid: 2048 x 7 x 7")

    # Branch Split Arrows cleanly targeting the inner boxes
    draw_arrow(8.9, 4.7, 9.6, 5.65, color="#15803D", lw=2.0, label="Branch A")
    draw_arrow(8.9, 4.0, 9.6, 2.9, color="#BE123C", lw=2.0, label="Branch B")

    # ── BRANCH A: Beaten Accuracy Model (ResNet-50 + SE) ──
    # Group box
    g_a = patches.FancyBboxPatch(
        (9.4, 4.6), 4.1, 2.1,
        boxstyle="round,pad=0.2,rounding_size=0.15",
        facecolor="#F0FDF4", edgecolor="#15803D", linewidth=1.8, linestyle="--"
    )
    ax.add_patch(g_a)
    ax.text(9.5, 6.6, "Optimal Parsimonious Pipeline: ResNet-50 + SE (Top Accuracy)",
            fontsize=8.5, fontweight="bold", color="#15803D")

    draw_box(9.6, 5.1, 1.8, 1.1, c_se, c_se_border,
             "SE Channel Attention", "Reduction r=16\nRecalibrates 2048 ch")
    draw_arrow(11.4, 5.65, 11.8, 5.65, color=c_se_border)

    draw_box(11.8, 5.0, 1.5, 1.3, c_win, c_win_border,
             "GAP + MLP Head", "Balanced Acc: 89.0%\nParams: 25.1M\nSweet Spot", lw=2.0)

    # ── BRANCH B: Reference Main Model (ResNet-50 + SE + MHSA) ──
    g_b = patches.FancyBboxPatch(
        (9.4, 0.4), 4.1, 3.6,
        boxstyle="round,pad=0.2,rounding_size=0.15",
        facecolor="#FFF1F2", edgecolor="#BE123C", linewidth=1.8, linestyle="--"
    )
    ax.add_patch(g_b)
    ax.text(9.5, 3.9, "Main Attention Pipeline: ResNet-50 + SE + MHSA (Alrabie et al.)",
            fontsize=8.5, fontweight="bold", color="#BE123C")

    draw_box(9.6, 2.4, 1.8, 1.0, c_se, c_se_border,
             "SE Attention", "2048 x 7 x 7")
    draw_arrow(10.5, 2.4, 10.5, 1.9, color=c_mhsa_border, label="Flatten (49 tokens)")

    draw_box(9.6, 0.9, 1.8, 1.0, c_mhsa, c_mhsa_border,
             "8-Head MHSA", "Spatial Context\nResidual + LayerNorm")
    draw_arrow(11.4, 1.4, 11.8, 1.4, color=c_mhsa_border)

    draw_box(11.8, 0.7, 1.5, 1.4, c_full, c_full_border,
             "GAP + Classifier", "Balanced Acc: 88.3%\nParams: 41.9M\n(+69% Parameters)", lw=2.0)

    draw_box(11.8, 2.5, 1.5, 1.0, "#FEF3C7", "#D97706",
             "Audited by XAI", "Grad-CAM, SHAP\nMulti-Head Attention", lw=1.2, ls=":")
    draw_arrow(10.5, 3.4, 11.8, 3.0, color="#D97706", lw=1.2)

    # Comparison Callout Bar at Bottom
    callout = patches.FancyBboxPatch(
        (0.5, 0.4), 8.5, 2.6,
        boxstyle="round,pad=0.15,rounding_size=0.12",
        facecolor="#F8FAFC", edgecolor="#94A3B8", linewidth=1.2
    )
    ax.add_patch(callout)
    ax.text(0.7, 2.7, "Key Empirical Findings across 24 Experimental Runs & Multi-Seed Benchmarks:",
            fontsize=9.5, fontweight="bold", color="#0F172A")
    ax.text(0.7, 2.25, "• SE Sweet Spot: Adding channel-wise SE recalibration achieves the highest balanced accuracy (89.0%)",
            fontsize=8.5, color="#15803D", fontweight="semibold")
    ax.text(0.7, 1.95, "  and AUC-ROC (95.4%) while requiring only 25.1M parameters (+0.5M over ResNet50 baseline).",
            fontsize=8.5, color="#334155")
    ax.text(0.7, 1.55, "• Self-Attention Diminishing Returns: Adding 8-head MHSA on a 7x7 spatial grid adds 16.8M parameters",
            fontsize=8.5, color="#B91C1C", fontweight="semibold")
    ax.text(0.7, 1.25, "  without statistical gains (88.3% BA). Paired tests show SE+MHSA beats baseline in only 1 of 3 seeds.",
            fontsize=8.5, color="#334155")
    ax.text(0.7, 0.85, "• Clinical Audit: 20-150 Hz occlusion drops BA by -7.81% (reliance=True), confirming cardiac focus.",
            fontsize=8.5, color="#0369A1", fontweight="semibold")
    ax.text(0.7, 0.55, "  Cross-site testing (LODO) drops BA to 52.7%, showing domain shift is the real performance bottleneck.",
            fontsize=8.5, color="#334155")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    pdf_path = save_path.with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Generated: {save_path} and {pdf_path}")

if __name__ == "__main__":
    out_dir = Path("Paper Writing/images")
    out_dir.mkdir(parents=True, exist_ok=True)
    create_architecture_comparison(out_dir / "architecture_comparison.png")
