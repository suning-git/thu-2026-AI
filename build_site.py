#!/usr/bin/env python3
"""build_site.py — 生成课程网站的静态资源页。

做三件事：
  1. 把 course_material/ 下的资料 Markdown 渲染成 resources/*.html（套 site.css + 返回链接）；
  2. 把 course_design/ 下的三课自包含 deck 拷进 slides/；
  3. 把第一周材料.zip 拷进 downloads/。

依赖：python `markdown`（`pip install markdown`，或用 venv）。
运行：python3 build_site.py
"""
import pathlib, shutil, sys, zipfile

try:
    import markdown
except ImportError:
    sys.exit("需要 markdown 库：pip install markdown（或在 venv 里装）")

HERE = pathlib.Path(__file__).resolve().parent          # course_website/
CM   = HERE.parent                                      # course_material/
CD   = CM.parent / "course_design"                      # teaching/course_design/

FAVICON = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'"
           "%3E%3Crect width='32' height='32' rx='7' fill='%2382318E'/%3E%3Ctext x='16' y='23' "
           "font-size='20' text-anchor='middle' fill='white' font-family='sans-serif'%3E智%3C/text%3E%3C/svg%3E")

SHELL = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%%TITLE%% · 从零构建智能模型</title>
<link rel="stylesheet" href="../assets/site.css">
<link rel="icon" href="%%FAVICON%%">
</head>
<body class="doc">
<nav class="docnav"><div class="wrap"><a href="../index.html">← 从零构建智能模型</a></div></nav>
<main class="prose">
%%BODY%%
</main>
<footer class="site"><div class="wrap">从零构建智能模型 · 苏宁 · 丘成桐数学科学中心</div></footer>
</body>
</html>
"""

# (源 MD, 输出文件名, <title>)
DOCS = [
    ("第一周作业.md",                 "homework-week1.html", "第一周作业"),
    ("环境搭建指引.md",               "setup.html",          "沙盒与 Claude Code 搭建指引"),
    ("sandbox-setup-brief.md",        "sandbox.html",        "沙盒环境简报"),
    ("Python与PyTorch读码手册.md",    "reading-guide.html",  "Python 与 PyTorch 读码手册"),
]

# (源 deck, slides/ 下的文件名)
DECKS = [
    ("第一课_slides_transformer.html", "lesson-1.html"),
    ("第二课_slides_training.html",    "lesson-2.html"),
    ("第三课_slides_真实训练.html",     "lesson-3.html"),
]

ZIP = "第一周材料.zip"


def main():
    (HERE / "resources").mkdir(exist_ok=True)
    (HERE / "slides").mkdir(exist_ok=True)
    (HERE / "downloads").mkdir(exist_ok=True)

    md = markdown.Markdown(extensions=["fenced_code", "tables", "sane_lists", "attr_list"])

    n_doc = 0
    for src, out, title in DOCS:
        p = CM / src
        if not p.exists():
            print(f"  warn: 缺资料 {src}，跳过"); continue
        md.reset()
        body = md.convert(p.read_text(encoding="utf-8"))
        html = SHELL.replace("%%TITLE%%", title).replace("%%FAVICON%%", FAVICON).replace("%%BODY%%", body)
        (HERE / "resources" / out).write_text(html, encoding="utf-8")
        n_doc += 1

    n_deck = 0
    for src, out in DECKS:
        p = CD / src
        if not p.exists():
            print(f"  warn: 缺 deck {src}，跳过"); continue
        shutil.copyfile(p, HERE / "slides" / out)
        n_deck += 1

    # 第一周材料.zip：从源头现打（更新版 .md + minimal_gpt/ + Mathematica 补充笔记本），再拷进 downloads/
    zp = CM / ZIP
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in ["第一周作业.md", "环境搭建指引.md", "sandbox-setup-brief.md", "Python与PyTorch读码手册.md"]:
            if (CM / name).exists():
                zf.write(CM / name, name)
        mg = CM / "llm-viz-local" / "llm-viz" / "minimal_gpt"        # 学生版 minimal_gpt（现居 fork 内）
        for p in sorted(mg.glob("*")):
            if p.is_file():
                zf.write(p, f"minimal_gpt/{p.name}")
        for p in sorted((CD / "additional_material").glob("*.nb")):  # Mathematica 补充笔记本
            zf.write(p, f"additional_material/{p.name}")
    shutil.copyfile(zp, HERE / "downloads" / ZIP)
    n_zip = len(zipfile.ZipFile(zp).namelist())

    print(f"wrote  resources: {n_doc}/{len(DOCS)}  ·  slides: {n_deck}/{len(DECKS)}  ·  材料包: {n_zip} 项")


if __name__ == "__main__":
    main()
