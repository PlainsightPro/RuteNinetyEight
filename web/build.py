"""Bundle the engine (CommonJS files) + WiezenLM weights + UI into one self-contained page.

usage: python web/build.py <wiezenlm.web.json> [out.html] [--fragment]
"""
import os, sys

here = os.path.dirname(os.path.abspath(__file__))
eng = os.path.join(here, "..", "engine")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
weights = open(args[0]).read().replace("</", "<\\/")
out = args[1] if len(args) > 1 else os.path.join(here, "index.html")

parts = ["var __mods = {};\n"
         "function __def(n, f) { __mods[n] = { f: f, e: null }; }\n"
         "function require(n) { n = n.replace(/^.*\\//, '').replace(/\\.js$/, ''); var m = __mods[n];\n"
         "  if (!m.e) { var module = { exports: {} }; m.e = module.exports; m.f(module, module.exports, require); m.e = module.exports; }\n"
         "  return m.e; }\n"]
for name in ["dds", "rules", "belief", "ai", "wiezenlm"]:
    src = open(os.path.join(eng, name + ".js")).read()
    parts.append(f"__def('{name}', function (module, exports, require) {{\n{src}\n}});\n")
engine = "".join(parts)
page = open(os.path.join(here, "template.html")).read()
page = page.replace("/*WEIGHTS*/", weights)
page = page.replace("/*ENGINE_TEXT*/", engine.replace("</", "<\\/"))
page = page.replace("/*ENGINE*/", engine)
page = page.replace("/*UI*/", open(os.path.join(here, "ui.js")).read())
if "--fragment" not in sys.argv:
    page = ('<!doctype html>\n<html lang="nl">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            "<style>body{margin:0}[hidden]{display:none!important}</style>\n</head>\n<body>\n" + page + "\n</body>\n</html>\n")
open(out, "w").write(page)
print(f"wrote {out} ({os.path.getsize(out) / 1e6:.2f} MB)")
