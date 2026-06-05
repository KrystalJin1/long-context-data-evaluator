"""Generate 0605 data source allocation dashboard as self-contained HTML."""
import pandas as pd
from pathlib import Path
from jinja2 import Template

INPUT = Path(__file__).parent.parent / "examples" / "0605入训长文数据采样.xlsx"
OUTPUT = Path(__file__).parent.parent / "outputs" / "0605_dashboard" / "report.html"

TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>0605 入训长文数据配比分析</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#f5f7fa;color:#1a1a2e;padding:24px;line-height:1.6}
.container{max-width:1200px;margin:0 auto}
h1{font-size:1.6em;margin-bottom:8px;color:#1a1a2e}
h2{font-size:1.2em;margin:24px 0 12px;color:#2d3748;border-bottom:2px solid #e2e8f0;padding-bottom:6px}
h3{font-size:1em;margin:16px 0 8px;color:#4a5568}
.subtitle{color:#718096;margin-bottom:24px;font-size:0.9em}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:24px}
.stat-box{background:#fff;border-radius:8px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.08)}
.stat-box .label{font-size:0.75em;color:#718096;text-transform:uppercase;letter-spacing:0.5px}
.stat-box .value{font-size:1.5em;font-weight:700;color:#2d3748;margin-top:4px}
.stat-box .unit{font-size:0.7em;color:#a0aec0;margin-left:2px}
.stat-box.highlight .value{color:#2b6cb0}
.stat-box.warning .value{color:#c05621}
table{width:100%;border-collapse:collapse;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.08);margin-bottom:16px;font-size:0.85em}
th{background:#edf2f7;padding:10px 12px;text-align:left;font-weight:600;color:#4a5568;position:sticky;top:0}
td{padding:8px 12px;border-top:1px solid #e2e8f0}
tr:hover td{background:#f7fafc}
.bar-cell{position:relative;min-width:120px}
.bar{height:20px;border-radius:3px;display:inline-block;vertical-align:middle}
.bar-blue{background:linear-gradient(90deg,#4299e1,#63b3ed)}
.bar-green{background:linear-gradient(90deg,#48bb78,#68d391)}
.bar-orange{background:linear-gradient(90deg,#ed8936,#f6ad55)}
.bar-purple{background:linear-gradient(90deg,#805ad5,#9f7aea)}
.pct{font-size:0.8em;color:#718096;margin-left:6px}
.section{background:#fff;border-radius:8px;padding:20px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.08)}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:768px){.two-col{grid-template-columns:1fr}}
.alert{background:#fff5f5;border-left:4px solid #fc8181;padding:12px 16px;border-radius:0 8px 8px 0;margin-bottom:16px;font-size:0.85em}
.alert strong{color:#c53030}
.insight{background:#ebf8ff;border-left:4px solid #63b3ed;padding:12px 16px;border-radius:0 8px 8px 0;margin-bottom:16px;font-size:0.85em}
.tag{display:inline-block;background:#edf2f7;border-radius:4px;padding:2px 8px;font-size:0.75em;color:#4a5568;margin-right:4px}
</style>
</head>
<body>
<div class="container">
<h1>0605 入训长文数据配比分析</h1>
<p class="subtitle">数据源: 0605入训长文数据采样.xlsx | 生成时间: 2026-06-05</p>

<!-- PLACEHOLDER_SECTIONS -->
</div>
</body>
</html>
"""


def simplify_lang(x):
    s = str(x)
    if ',' in s and len(s) > 10:
        return '多语言混合'
    return s


def generate():
    df = pd.read_excel(INPUT)
    df['一级类别'] = df['Unnamed: 0'].ffill()
    df['语种简化'] = df['配比语种-归一化'].apply(simplify_lang)

    total_tokens = df['token_num(B)'].sum()

    # Build HTML sections
    sections = []

    # 1. Overview stats
    sections.append('<div class="stat-grid">')
    stats = [
        ('数据源总数', f'{len(df)}', '个', 'highlight'),
        ('总 Token 量', f'{total_tokens:.1f}', 'B', 'highlight'),
        ('一级类别', f'{df["一级类别"].nunique()}', '类', ''),
        ('配比类别', f'{df["配比类别"].nunique()}', '类', ''),
        ('语种数', f'{df["配比语种-归一化"].nunique()}', '种', ''),
        ('H5来源', f'FlexCorpus {(df["H5来源（平台/FlexCorpus）"]=="FlexCorpus").sum()}', '条', ''),
    ]
    for label, value, unit, cls in stats:
        sections.append(f'<div class="stat-box {cls}"><div class="label">{label}</div><div class="value">{value}<span class="unit">{unit}</span></div></div>')
    sections.append('</div>')

    # Alerts
    top1 = df.nlargest(1, 'token_num(B)').iloc[0]
    top1_ratio = top1['token_num(B)'] / total_tokens * 100
    sections.append(f'<div class="alert"><strong>集中度风险：</strong>单一数据源「{str(top1["三级类别"])[:30]}」占总量 {top1_ratio:.1f}%（{top1["token_num(B)"]:.1f}B），内部配比 {top1["内部配比"]:.1f}</div>')

    synth_ratio = df[df['一级类别']=='合成长文']['token_num(B)'].sum() / total_tokens * 100
    sections.append(f'<div class="insight"><strong>结构观察：</strong>合成长文仅占 {synth_ratio:.1f}%，MRCR/多文档摘要等任务型数据需从合成通道补充</div>')

    # 2. 一级类别
    sections.append('<h2>一级类别（长文类型）</h2>')
    sections.append('<div class="section">')
    level1 = df.groupby('一级类别').agg(count=('token_num(B)', 'count'), tokens=('token_num(B)', 'sum')).sort_values('tokens', ascending=False)
    sections.append('<table><tr><th>类别</th><th>数据源数</th><th>Token量(B)</th><th>占比</th></tr>')
    for name, row in level1.iterrows():
        pct = row['tokens'] / total_tokens * 100
        w = min(pct / 80 * 100, 100)
        sections.append(f'<tr><td><strong>{name}</strong></td><td>{int(row["count"])}</td><td>{row["tokens"]:.1f}</td><td class="bar-cell"><span class="bar bar-blue" style="width:{w}%"></span><span class="pct">{pct:.1f}%</span></td></tr>')
    sections.append('</table></div>')

    # 3. 配比类别
    sections.append('<h2>配比类别分布</h2>')
    sections.append('<div class="section">')
    cat = df.groupby('配比类别').agg(count=('token_num(B)', 'count'), tokens=('token_num(B)', 'sum')).sort_values('tokens', ascending=False)
    sections.append('<table><tr><th>类别</th><th>数据源数</th><th>Token量(B)</th><th>占比</th></tr>')
    for name, row in cat.iterrows():
        pct = row['tokens'] / total_tokens * 100
        w = min(pct / 40 * 100, 100)
        sections.append(f'<tr><td>{name}</td><td>{int(row["count"])}</td><td>{row["tokens"]:.1f}</td><td class="bar-cell"><span class="bar bar-green" style="width:{w}%"></span><span class="pct">{pct:.1f}%</span></td></tr>')
    sections.append('</table></div>')

    # 4. 语种
    sections.append('<h2>语种分布</h2>')
    sections.append('<div class="section">')
    lang = df.groupby('语种简化').agg(count=('token_num(B)', 'count'), tokens=('token_num(B)', 'sum')).sort_values('tokens', ascending=False).head(15)
    sections.append('<table><tr><th>语种</th><th>数据源数</th><th>Token量(B)</th><th>占比</th></tr>')
    for name, row in lang.iterrows():
        pct = row['tokens'] / total_tokens * 100
        w = min(pct / 50 * 100, 100)
        sections.append(f'<tr><td>{name}</td><td>{int(row["count"])}</td><td>{row["tokens"]:.1f}</td><td class="bar-cell"><span class="bar bar-purple" style="width:{w}%"></span><span class="pct">{pct:.1f}%</span></td></tr>')
    sections.append('</table></div>')

    # 5. Cross table
    sections.append('<h2>配比类别 × 一级类别 (Token量B)</h2>')
    sections.append('<div class="section" style="overflow-x:auto">')
    cross = df.pivot_table(index='配比类别', columns='一级类别', values='token_num(B)', aggfunc='sum', fill_value=0)
    cross['总计'] = cross.sum(axis=1)
    cross = cross.sort_values('总计', ascending=False)
    cols = [c for c in cross.columns if c != '总计'] + ['总计']
    sections.append('<table><tr><th>配比类别</th>')
    for c in cols:
        sections.append(f'<th>{c}</th>')
    sections.append('</tr>')
    for name, row in cross[cols].iterrows():
        sections.append(f'<tr><td><strong>{name}</strong></td>')
        for c in cols:
            v = row[c]
            style = 'font-weight:700' if c == '总计' else ''
            sections.append(f'<td style="{style}">{v:.1f}</td>' if v > 0 else '<td style="color:#cbd5e0">-</td>')
        sections.append('</tr>')
    sections.append('</table></div>')

    # 6. Top 20 数据源
    sections.append('<h2>Top 20 数据源 (by Token量)</h2>')
    sections.append('<div class="section" style="overflow-x:auto">')
    cat3 = df.groupby('三级类别').agg(count=('token_num(B)', 'count'), tokens=('token_num(B)', 'sum'), category=('配比类别', 'first'), lang=('语种简化', 'first')).sort_values('tokens', ascending=False).head(20)
    sections.append('<table><tr><th>#</th><th>数据源</th><th>类别</th><th>语种</th><th>Token量(B)</th><th>占比</th></tr>')
    for i, (name, row) in enumerate(cat3.iterrows(), 1):
        pct = row['tokens'] / total_tokens * 100
        display_name = str(name)[:55]
        sections.append(f'<tr><td>{i}</td><td>{display_name}</td><td><span class="tag">{row["category"]}</span></td><td>{row["lang"]}</td><td>{row["tokens"]:.1f}</td><td>{pct:.1f}%</td></tr>')
    sections.append('</table></div>')

    # 7. Top 20 内部配比
    sections.append('<h2>Top 20 内部配比权重</h2>')
    sections.append('<div class="section" style="overflow-x:auto">')
    top_ratio = df.nlargest(20, '内部配比')[['三级类别', '配比类别', '语种简化', 'token_num(B)', '内部配比']]
    sections.append('<table><tr><th>#</th><th>数据源</th><th>类别</th><th>语种</th><th>Token量(B)</th><th>内部配比</th></tr>')
    for i, (_, row) in enumerate(top_ratio.iterrows(), 1):
        name = str(row['三级类别'])[:50]
        w = min(row['内部配比'] / 35 * 100, 100)
        sections.append(f'<tr><td>{i}</td><td>{name}</td><td><span class="tag">{row["配比类别"]}</span></td><td>{row["语种简化"]}</td><td>{row["token_num(B)"]:.1f}</td><td class="bar-cell"><span class="bar bar-orange" style="width:{w}%"></span><span class="pct">{row["内部配比"]:.2f}</span></td></tr>')
    sections.append('</table></div>')

    # 8. Summary insights
    sections.append('<h2>数据观察与建议</h2>')
    sections.append('<div class="section"><ul style="padding-left:20px;line-height:2">')
    insights = [
        f'中英文占比均衡（中文 {df[df["语种简化"]=="中文"]["token_num(B)"].sum()/total_tokens*100:.1f}% / 英文 {df[df["语种简化"]=="英文"]["token_num(B)"].sum()/total_tokens*100:.1f}%），小语种覆盖广但量级小',
        '长文 + 书籍合计占 65.7%，头部集中度较高',
        f'百度小说单一数据源占总量 {top1_ratio:.1f}%，建议关注过拟合风险',
        '多轮/角色、问答类数据接近 0，不覆盖 MRCR/对话类长文训练场景',
        f'合成长文仅 {synth_ratio:.1f}%（{df[df["一级类别"]=="合成长文"]["token_num(B)"].sum():.1f}B），任务型训练数据需补充',
        '94% 数据来自 FlexCorpus，平台数据仅 29 条',
    ]
    for ins in insights:
        sections.append(f'<li>{ins}</li>')
    sections.append('</ul></div>')

    # Assemble
    html = TEMPLATE.replace('<!-- PLACEHOLDER_SECTIONS -->', '\n'.join(sections))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Dashboard written to {OUTPUT}')


if __name__ == '__main__':
    generate()
