import subprocess
import sys
from pathlib import Path

def compile_latex(tex_file_path):
    """Compile LaTeX with smart BibTeX detection"""
    tex_path = Path(tex_file_path).resolve()
    
    if not tex_path.exists():
        print(f"Error: File not found: {tex_path}")
        return False
    
    work_dir = tex_path.parent
    tex_name = tex_path.stem
    
    print(f"Compiling: {tex_path.name}")
    print(f"Directory: {work_dir}\n")
    
    # Pass 1: pdflatex
    print("=== Pass 1: pdflatex ===")
    result1 = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", tex_path.name],
        cwd=work_dir,
        capture_output=True,
        text=True
    )
    
    if result1.returncode != 0:
        print("ERROR in pdflatex pass 1:")
        print(result1.stdout[-2000:])
        return False
    print("✓ Pass 1 complete")
    
    # Check if .aux file has bibliography commands
    aux_file = work_dir / f"{tex_name}.aux"
    if not aux_file.exists():
        print(f"ERROR: {aux_file.name} not created")
        return False
    
    aux_content = aux_file.read_text()
    has_bibdata = '\\bibdata{' in aux_content
    has_bibstyle = '\\bibstyle{' in aux_content
    needs_bibtex = has_bibdata and has_bibstyle
    
    if needs_bibtex:
        print("\n=== Running BibTeX ===")
        result2 = subprocess.run(
            ["bibtex", tex_name],
            cwd=work_dir,
            capture_output=True,
            text=True
        )
        
        if result2.returncode == 0:
            print("✓ BibTeX complete")
        else:
            print(f"⚠ BibTeX warnings (code {result2.returncode}):")
            print(result2.stdout)
    else:
        print("\n=== Skipping BibTeX (no \\bibliography commands) ===")
    
    # Pass 2: pdflatex
    print("\n=== Pass 2: pdflatex ===")
    result3 = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", tex_path.name],
        cwd=work_dir,
        capture_output=True,
        text=True
    )
    print("✓ Pass 2 complete")
    
    # Pass 3: pdflatex (for cross-references)
    print("=== Pass 3: pdflatex ===")
    result4 = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", tex_path.name],
        cwd=work_dir,
        capture_output=True,
        text=True
    )
    print("✓ Pass 3 complete")
    
    # Check for output PDF
    pdf_file = work_dir / f"{tex_name}.pdf"
    if pdf_file.exists():
        size_kb = pdf_file.stat().st_size / 1024
        print(f"\n{'='*50}")
        print(f"✓ SUCCESS: {pdf_file.name}")
        print(f"  Location: {pdf_file}")
        print(f"  Size: {size_kb:.1f} KB")
        print(f"{'='*50}")
        return True
    else:
        print(f"\n✗ FAILED: PDF not created")
        print("\nLast pdflatex output:")
        print(result4.stdout[-1500:])
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python LaTex_Compiler_052525_v5.py <tex_file_path>")
        sys.exit(1)
    
    success = compile_latex(sys.argv[1])
    sys.exit(0 if success else 1)