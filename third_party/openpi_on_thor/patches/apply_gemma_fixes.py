#!/usr/bin/env python3
"""
Apply two ONNX-export-critical fixes to the installed transformers'
modeling_gemma.py WITHOUT modifying the git-cloned OpenPi source.

This script is run AFTER the standard transformers_replace copy (Step 6.3)
and patches the installed copy in-place.

Fix 1 — GemmaRMSNorm.extra_repr():
  Add a hasattr guard so ONNX tracing doesn't crash when the weight
  attribute hasn't been set (adaptive-norm layers use .dense instead).

Fix 2 — GemmaAttention.forward():
  Replace reshape(*input_shape, -1) with an explicit dimension
  (num_attention_heads * head_dim).  The -1 makes ALL dimensions appear
  dynamic in the ONNX graph, which causes TensorRT's FP4 block
  quantization to fail with "input extent in the blocked axis should be
  known at build time".

Usage (inside Docker container):
    python openpi_on_thor/patches/apply_gemma_fixes.py
"""
import importlib
import pathlib
import sys

def find_modeling_gemma() -> pathlib.Path:
    """Locate the installed transformers modeling_gemma.py."""
    import transformers
    base = pathlib.Path(transformers.__file__).parent
    target = base / "models" / "gemma" / "modeling_gemma.py"
    if not target.exists():
        print(f"ERROR: {target} not found", file=sys.stderr)
        sys.exit(1)
    return target


def apply_fixes(path: pathlib.Path) -> None:
    text = path.read_text()
    changed = False

    # --- Fix 1: hasattr guard in extra_repr ---
    old_repr = '    def extra_repr(self):\n        repr_str = f"{tuple(self.weight.shape)}, eps={self.eps}"'
    new_repr = (
        '    def extra_repr(self):\n'
        '        if hasattr(self, \'weight\') and self.weight is not None:\n'
        '            repr_str = f"{tuple(self.weight.shape)}, eps={self.eps}"\n'
        '        else:\n'
        '            repr_str = f"eps={self.eps}"'
    )
    if old_repr in text:
        text = text.replace(old_repr, new_repr)
        changed = True
        print("  [1/2] Applied hasattr guard in GemmaRMSNorm.extra_repr()")
    elif "hasattr(self, 'weight')" in text:
        print("  [1/2] hasattr guard already applied — skipping")
    else:
        print("  [1/2] WARNING: could not locate extra_repr pattern")

    # --- Fix 2: explicit reshape dimension in GemmaAttention.forward ---
    old_reshape = "attn_output = attn_output.reshape(*input_shape, -1).contiguous()"
    new_reshape = "attn_output = attn_output.reshape(*input_shape, self.config.num_attention_heads * self.head_dim).contiguous()"
    if old_reshape in text:
        text = text.replace(old_reshape, new_reshape)
        changed = True
        print("  [2/2] Applied explicit reshape dimension in GemmaAttention.forward()")
    elif "self.config.num_attention_heads * self.head_dim" in text:
        print("  [2/2] Explicit reshape already applied — skipping")
    else:
        print("  [2/2] WARNING: could not locate reshape pattern")

    if changed:
        path.write_text(text)
        print(f"\n  Patched: {path}")
    else:
        print(f"\n  No changes needed: {path}")


if __name__ == "__main__":
    print("Applying ONNX/TRT compatibility fixes to modeling_gemma.py...")
    target = find_modeling_gemma()
    apply_fixes(target)
    print("Done.")
