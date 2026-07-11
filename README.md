# 从零构建智能模型 · 课程网站

清华大学丘成桐数学科学中心「从零构建智能模型」课程的网站。纯静态、全相对路径，可托管到 GitHub Pages（根路径或 `/<repo>/` 子路径均可）。

任课教师：苏宁 · 丘成桐数学科学中心。

## 结构

- `index.html` — 落地页（简介 / 课程信息 / 大纲 / 课程表 / 资源）
- `slides/` — 三课讲义（自包含 HTML，方向键翻页）
- `resources/` — 作业 · 沙盒与 Claude Code 搭建 · 沙盒环境简报 · Python/PyTorch 读码手册
- `downloads/第一周材料.zip` — 第一周材料包（含 `minimal_gpt/` 与 Mathematica 补充笔记本）
- `assets/site.css` — 样式
- `build_site.py` — 由源 Markdown 生成 `resources/` 与材料包

## 生成

`resources/` 各页与材料包由 `build_site.py` 从课程仓库中的源 Markdown 渲染 / 打包而成（依赖 `markdown`：`pip install markdown`）。改动源 `.md` 后重跑即可同步。

## 部署（GitHub Pages）

推到一个 public 仓库，Settings → Pages → 从 `main` 分支根目录发布。站点为纯静态，无需构建步骤。
