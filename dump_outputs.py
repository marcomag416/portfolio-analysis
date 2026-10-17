"""Print notebook outputs as plain text (tables flattened), for review in a terminal."""
import html, re, sys
import nbformat
nb = nbformat.read(sys.argv[1] if len(sys.argv) > 1 else "portfolio_choice.ipynb", 4)
for i, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    for o in c.get("outputs", []):
        if o.output_type == "stream":
            print(f"[{i}] {o.text.strip()}")
        elif o.output_type in ("execute_result", "display_data"):
            d = o.get("data", {})
            if "text/html" in d:
                print(f"[{i}] TABLE")
                for r in re.findall(r"<tr>(.*?)</tr>", d["text/html"], re.S):
                    cells = [html.unescape(re.sub("<.*?>", "", x)).strip()
                             for x in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, re.S)]
                    print("    " + " | ".join(cells))
            elif "image/png" in d:
                print(f"[{i}] <figure>")
        elif o.output_type == "error":
            print(f"[{i}] ERROR {o.ename}: {o.evalue}")
