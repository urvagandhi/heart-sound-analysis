"""Generate official Mermaid-based architecture comparison figure for IEEE paper."""
import subprocess
import base64
from pathlib import Path
from PIL import Image

def generate_mermaid_architecture(output_png: Path):
    mmd_path = output_png.parent / "architecture_comparison.mmd"
    if not mmd_path.exists():
        raise FileNotFoundError(f"Mermaid file not found at: {mmd_path}")
    
    code = mmd_path.read_text(encoding="utf-8")
    b64 = base64.b64encode(code.encode("utf-8")).decode("utf-8")
    url = f"https://mermaid.ink/img/{b64}"
    
    print(f"Rendering Mermaid diagram from: {mmd_path} ...")
    cmd = ["curl", "-s", "-o", str(output_png), url]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode == 0 and output_png.exists() and output_png.stat().st_size > 1000:
        img = Image.open(output_png).convert("RGB")
        pdf_path = output_png.with_suffix(".pdf")
        img.save(pdf_path, "PDF", resolution=300.0)
        print(f"Successfully generated: {output_png} and {pdf_path} (size: {img.size})")
    else:
        print("Network render failed or empty file. Keeping existing high-res asset.")

if __name__ == "__main__":
    out_dir = Path("Paper Writing/images")
    generate_mermaid_architecture(out_dir / "architecture_comparison.png")
