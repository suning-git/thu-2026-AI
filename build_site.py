#!/usr/bin/env python3
"""build_site.py — 生成课程网站的静态资源页。

做五件事：
  1. 把 course_material/ 下的资料 Markdown 渲染成 resources/*.html（套 site.css + 返回链接）；
  2. 把 teaching/lecture_notes/ 下的长篇讲义渲染成 notes/*.html（同壳 + MathJax 渲染公式 + 拷图）；
  3. 把 course_design/ 下的自包含 deck 拷进 slides/；
  4. 把第一周材料.zip 拷进 downloads/；
  5. 把主库 projects/ 下的课堂现场 demo 代码同步进 live-demos/（源头唯一在主库）。

依赖：python `markdown`（`pip install markdown`，或用 venv）。
运行：python3 build_site.py
"""
import pathlib, re, shutil, sys, zipfile

try:
    import markdown
except ImportError:
    sys.exit("需要 markdown 库：pip install markdown（或在 venv 里装）")

HERE = pathlib.Path(__file__).resolve().parent          # course_website/
CM   = HERE.parent                                      # course_material/
CD   = CM.parent / "course_design"                      # teaching/course_design/
LN   = CM.parent / "lecture_notes"                       # teaching/lecture_notes/
NANO = HERE.parents[4]                                  # nanoinfra 主库根（…/private/ning/teaching 上三层）

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

# 长篇讲义专用壳：与 SHELL 同版式，额外挂 MathJax（CHTML 输出——能渲染 \text{} 里的中文，
# tex→svg 做不到）在浏览器端渲染公式。站点是公开 GitHub Pages，用 CDN 无妨。
NOTE_SHELL = SHELL.replace("</head>", """<script>
MathJax = {
  tex: { inlineMath: [['$', '$']], displayMath: [['$$', '$$']], processEscapes: true },
  options: { skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] }
};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
</head>""")

# (源 MD, 输出文件名, <title>, 图目录名)。图目录整个拷进 notes/<图目录名>/，
# 与 md 里的相对图片路径一致，链接不用改。图目录名为 None 表示该讲义无配图。
NOTES = [
    ("前沿架构_解读.md", "frontier-arch.html",
     "前沿架构解读:DeepSeek V4", "nano-dsv4_figs"),
]


def protect_math(text):
    """把 $$…$$ 和 $…$ 数学段抠成占位符，防 markdown 把 x_t 里的下划线吃成 <em>；
    渲染后原样填回，交给 MathJax。此讲义正文的代码块/行内代码里没有 $（已核），所以直接扫全文安全。"""
    store = []
    def stash(m):
        store.append(m.group(0))
        return f"xMATHSPAN{len(store)-1}x"
    text = re.sub(r'\$\$.+?\$\$', stash, text, flags=re.S)   # 先块级
    text = re.sub(r'\$[^\$\n]+?\$', stash, text)             # 再行内
    return text, store


def restore_math(html, store):
    for i, raw in enumerate(store):
        html = html.replace(f"xMATHSPAN{i}x", raw)
    return html


# (源 MD, 输出文件名, <title>)
DOCS = [
    ("第一周作业.md",                 "homework-week1.html", "第一周作业"),
    ("环境搭建指引.md",               "setup.html",          "沙盒与 Claude Code 搭建指引"),
    ("sandbox-setup-brief.md",        "sandbox.html",        "沙盒环境简报"),
    ("Python与PyTorch读码手册.md",    "reading-guide.html",  "Python 与 PyTorch 读码手册"),
    ("vibe_coding_live_demo.md",      "vibe-coding-demo.html", "Vibe Coding 现场实录:从零到两个多模态项目"),
]

# (源 deck, slides/ 下的文件名)
DECKS = [
    ("第一课_slides_transformer.html", "lesson-1.html"),
    ("第二课_slides_training.html",    "lesson-2.html"),
    ("第三课_slides_真实训练.html",     "lesson-3.html"),
    ("第四课_slides_多模态运动_公开版.html", "lesson-4.html"),
    ("第六课_slides_优化器与训练工程.html", "lesson-6.html"),
    ("第七课_slides_前沿架构.html", "lesson-7.html"),
]

ZIP = "第一周材料.zip"


def main():
    (HERE / "resources").mkdir(exist_ok=True)
    (HERE / "slides").mkdir(exist_ok=True)
    (HERE / "notes").mkdir(exist_ok=True)
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

    # 长篇讲义：保护数学段 → markdown → 填回数学 → 套 MathJax 壳；图目录整拷进 notes/
    n_note = 0
    for src, out, title, figdir in NOTES:
        p = LN / src
        if not p.exists():
            print(f"  warn: 缺讲义 {src}，跳过"); continue
        text, store = protect_math(p.read_text(encoding="utf-8"))
        md.reset()
        body = restore_math(md.convert(text), store)
        html = NOTE_SHELL.replace("%%TITLE%%", title).replace("%%FAVICON%%", FAVICON).replace("%%BODY%%", body)
        (HERE / "notes" / out).write_text(html, encoding="utf-8")
        if figdir:
            dst = HERE / "notes" / figdir
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(LN / figdir, dst, ignore=shutil.ignore_patterns("*.md", "__pycache__"))
        n_note += 1

    # 发布版 deck 注入"← → 键翻页"角标:按过一次翻页键即永久消失(localStorage),
    # 15 秒无操作也自动淡出;只进网站拷贝,课堂版原件不动。
    PG_HINT = (
        '<div id="pgHint" style="position:fixed;right:18px;bottom:60px;z-index:99;'
        'font:500 14px/1.6 system-ui,sans-serif;color:#fff;background:rgba(27,27,31,.72);'
        'padding:6px 14px;border-radius:999px;pointer-events:none;transition:opacity .5s">'
        '← → 键翻页</div>\n'
        "<script>(function(){var h=document.getElementById('pgHint');"
        "try{if(localStorage.getItem('pgHintDone')){h.remove();return;}}catch(e){}"
        "function done(){try{localStorage.setItem('pgHintDone','1')}catch(e){}"
        "h.style.opacity=0;setTimeout(function(){h.remove()},600);"
        "window.removeEventListener('keydown',k);}"
        "function k(e){if(e.key==='ArrowRight'||e.key==='ArrowLeft'||e.key==='PageDown'"
        "||e.key==='PageUp'||e.key===' ')done();}"
        "window.addEventListener('keydown',k);setTimeout(done,15000);})();</script>\n")

    n_deck = 0
    for src, out in DECKS:
        p = CD / src
        if not p.exists():
            print(f"  warn: 缺 deck {src}，跳过"); continue
        html = p.read_text(encoding="utf-8")
        if "</body>" in html:
            html = html.replace("</body>", PG_HINT + "</body>", 1)
        else:
            html += PG_HINT
        (HERE / "slides" / out).write_text(html, encoding="utf-8")
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

    # 交互 demo：优化器实验台（自包含单文件，源头在 teaching/private/）
    shutil.copyfile(CM.parent / "private" / "优化器_demo.html",
                    HERE / "resources" / "optimizer-demo.html")

    # 课堂现场 demo 代码：主库 projects/ 是唯一源头，这里整目录同步（去缓存）
    demos_src = NANO / "projects" / "training_engineering_demos"
    demos_dst = HERE / "live-demos" / "training_engineering_demos"
    if demos_dst.exists():
        shutil.rmtree(demos_dst)
    shutil.copytree(demos_src, demos_dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    n_demo = len([p for p in demos_dst.rglob("*") if p.is_file()])

    print(f"wrote  resources: {n_doc}/{len(DOCS)}  ·  notes: {n_note}/{len(NOTES)}  ·  slides: {n_deck}/{len(DECKS)}  ·  材料包: {n_zip} 项  ·  live-demos: {n_demo} 文件")


if __name__ == "__main__":
    main()
