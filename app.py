
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager as fm
import os
from datetime import datetime
from matplotlib.patches import Wedge
import io
import json

# 设置matplotlib字体（兼容 Streamlit Community Cloud / Linux）
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [
    'Noto Sans CJK SC',
    'Noto Sans CJK',
    'Noto Sans SC',
    'Noto Sans',
    'WenQuanYi Zen Hei',
    'SimHei',
    'Microsoft YaHei',
    'STHeiti',
    'Songti SC',
    'PingFang SC',
    'Arial Unicode MS',
    'DejaVu Sans',
]
plt.rcParams['axes.unicode_minus'] = False

CH_FONT = None
CH_FONT_NAME = None


def init_chinese_font():
    global CH_FONT, CH_FONT_NAME
    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK",
        "Noto Sans SC",
        "Noto Sans",
        "WenQuanYi Zen Hei",
        "SimHei",
        "Microsoft YaHei",
        "STHeiti",
        "Songti SC",
        "PingFang SC",
    ]
    name_map = {f.name: f for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in name_map:
            f = name_map[name]
            CH_FONT = fm.FontProperties(fname=f.fname)
            CH_FONT_NAME = f.name
            plt.rcParams["font.family"] = CH_FONT_NAME
            break
    if CH_FONT is None:
        for f in fm.fontManager.ttflist:
            if any(k in f.name for k in ["Noto", "WenQuanYi", "Hei", "YaHei", "Song", "Kai"]):
                CH_FONT = fm.FontProperties(fname=f.fname)
                CH_FONT_NAME = f.name
                plt.rcParams["font.family"] = CH_FONT_NAME
                break


init_chinese_font()

RULES_FILE = os.path.join(os.path.dirname(__file__), "code_20260417.csv")
OVERRIDE_FILE = os.path.join(os.path.dirname(__file__), ".classification_overrides.json")

st.set_page_config(page_title="基金资产穿透分析系统", layout="wide")

# --- 核心逻辑类 ---
class FundClassifier:
    def __init__(self, rules_path):
        self.rules = []
        self.l1_list = []
        self.l2_list = []
        self.l3_list = []
        self.load_rules(rules_path)

    def load_rules(self, path):
        try:
            df_rules = pd.read_csv(path, encoding='utf-8')
        except UnicodeDecodeError:
            df_rules = pd.read_csv(path, encoding='gbk')
        
        for _, row in df_rules.iterrows():
            rule = {
                'priority': int(row['优先级']),
                'l1': str(row['一级']).strip(),
                'l2': str(row['二级']).strip(),
                'l3': str(row['三级']).strip(),
                'keywords': [k.strip() for k in str(row['匹配关键词']).split('、') if k.strip()],
                'exclude': [k.strip() for k in str(row['排除']).split('、') if k.strip() and k != '-'],
            }
            self.rules.append(rule)
            if rule['l1'] not in self.l1_list: self.l1_list.append(rule['l1'])
            if rule['l2'] not in self.l2_list: self.l2_list.append(rule['l2'])
            if rule['l3'] not in self.l3_list: self.l3_list.append(rule['l3'])
        
        self.rules.sort(key=lambda x: x['priority'])
        self.l1_list = sorted(self.l1_list)
        self.l2_list = sorted(list(set(self.l2_list)))
        self.l3_list = sorted(list(set(self.l3_list)))

    def classify(self, name):
        name = str(name)
        for rule in self.rules:
            if any(k in name for k in rule['keywords']):
                if not any(e in name for e in rule['exclude']):
                    return rule['l1'], rule['l2'], rule['l3'], "规则匹配"
        
        # AI 智能猜测逻辑 (兜底)
        name_upper = name.upper()
        if any(k in name_upper for k in ['债', 'BOND', '国开']):
            return "债券", "混合债", "混合债", "AI猜测"
        if any(k in name_upper for k in ['股', '沪深', '中证', '创业', '科创', '红利']):
            return "A股", "主动", "主动混合", "AI猜测"
        if any(k in name_upper for k in ['QDII', '全球', '海外', '港', '美']):
            return "海外", "新兴", "新兴", "AI猜测"
        if any(k in name_upper for k in ['货', '现金', '理财']):
            return "货币", "货币基金", "货币基金", "AI猜测"
            
        return "其他", "待分类", "待分类", "待分类"


def load_overrides():
    if not os.path.exists(OVERRIDE_FILE):
        return {}
    try:
        with open(OVERRIDE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def save_overrides(df):
    data = {}
    for _, row in df.iterrows():
        code = str(row.get("基金代码", "")).strip()
        name = str(row.get("基金名称", "")).strip()
        data[code] = {
            "基金名称": name,
            "一级": str(row.get("一级", "")).strip(),
            "二级": str(row.get("二级", "")).strip(),
            "三级": str(row.get("三级", "")).strip(),
            "更新时间": datetime.now().isoformat(timespec="seconds"),
        }
    with open(OVERRIDE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def apply_overrides(df):
    overrides = load_overrides()
    if not overrides:
        return df
    out = df.copy()
    applied = 0
    for idx in out.index:
        code = str(out.at[idx, "基金代码"]).strip()
        name = str(out.at[idx, "基金名称"]).strip()
        hit = overrides.get(code)
        if not hit:
            # 兜底：同名匹配
            for _, v in overrides.items():
                if v.get("基金名称", "").strip() == name:
                    hit = v
                    break
        if hit:
            out.at[idx, "一级"] = hit.get("一级", out.at[idx, "一级"])
            out.at[idx, "二级"] = hit.get("二级", out.at[idx, "二级"])
            out.at[idx, "三级"] = hit.get("三级", out.at[idx, "三级"])
            out.at[idx, "匹配状态"] = "历史修正"
            applied += 1
    if applied > 0:
        st.info(f"已自动应用历史人工修正 {applied} 条。")
    return out


def aggregate_holdings(df, min_amount=10.0):
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    if '资产金额' in out.columns:
        out['资产金额'] = pd.to_numeric(out['资产金额'], errors='coerce').fillna(0)
    if '基金代码' in out.columns:
        out['基金代码'] = out['基金代码'].astype(str).str.strip()
    if '基金名称' in out.columns:
        out['基金名称'] = out['基金名称'].astype(str).str.strip()

    if '基金代码' in out.columns:
        gcols = ['基金代码']
        if '基金名称' in out.columns:
            gcols.append('基金名称')
        out = out.groupby(gcols, as_index=False).agg({'资产金额': 'sum'})

    out = out[out['资产金额'].abs() >= float(min_amount)].copy()
    return out


def parse_market_holdings(uploaded_market_file):
    xls = pd.ExcelFile(uploaded_market_file)
    sheet = '持仓数据' if '持仓数据' in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(uploaded_market_file, sheet_name=sheet)
    if '代码' not in df.columns or '名称' not in df.columns or '持有金额' not in df.columns:
        return None, sheet, list(df.columns)
    out = df[['代码', '名称', '持有金额']].copy()
    out = out.dropna(subset=['名称'])
    def norm_code(x):
        if pd.isna(x):
            return ''
        try:
            return str(int(float(x))).zfill(6)
        except Exception:
            return str(x).strip()
    out['代码'] = out['代码'].apply(norm_code)
    out['持有金额'] = pd.to_numeric(out['持有金额'], errors='coerce').fillna(0)
    out = out[out['持有金额'] != 0].copy()
    name_upper = out['名称'].astype(str).str.upper()
    is_stock_code = out['代码'].astype(str).str.match(r'^(00|30|60|68)\d{4}$')
    is_etf_code = out['代码'].astype(str).str.match(r'^(5\d{5}|15\d{4}|159\d{3})$')
    is_etf_name = name_upper.str.contains('ETF', na=False)
    keep = (~is_stock_code) & (is_etf_code | is_etf_name)
    out = out[keep].copy()
    out = out.rename(columns={'代码': '基金代码', '名称': '基金名称', '持有金额': '资产金额'})
    out = aggregate_holdings(out, min_amount=10.0)
    return out, sheet, None

def get_sunburst_plot(df_stats, total_assets):
    """生成三层旭日图的 Matplotlib Figure"""
    total_val = total_assets if total_assets > 0 else 1
    base_colors = {
        'A股': '#5D8AA8', '海外': '#ED7D31', '债券': '#00A2E8', 
        '货币': '#004B66', '商品': '#FFD700', '其他': '#A5A5A5'
    }
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.axis('off')
    
    inner_r, mid_r, outer_r, labels_r = 0.3, 0.55, 0.8, 1.05
    start_angle = 90
    
    # 1. 第一层 (一级分类)
    l1_stats = df_stats.groupby('一级').agg({'资产': 'sum'}).reset_index()
    l1_order = ['货币', '债券', '商品', '海外', 'A股', '其他']
    l1_stats['order'] = l1_stats['一级'].apply(lambda x: l1_order.index(x) if x in l1_order else 99)
    l1_stats = l1_stats[l1_stats['资产'] > 0].sort_values('order')
    
    l1_angles = {}
    curr_angle = start_angle
    for _, row in l1_stats.iterrows():
        name = row['一级']
        pct = row['资产'] / total_val
        width = pct * 360
        color = base_colors.get(name, '#A5A5A5')
        ax.add_patch(Wedge((0, 0), mid_r, curr_angle - width, curr_angle, width=mid_r-inner_r, facecolor=color, edgecolor='w'))
        mid_a = curr_angle - width / 2
        rad = np.deg2rad(mid_a)
        ax.text(
            (inner_r + mid_r) / 2 * np.cos(rad),
            (inner_r + mid_r) / 2 * np.sin(rad),
            f"{name}\n{pct:.1%}",
            ha='center',
            va='center',
            color='white',
            weight='bold',
            fontsize=12,
            fontproperties=CH_FONT,
        )
        l1_angles[name] = (curr_angle, width)
        curr_angle -= width

    # 2. 第二层 (二级分类)
    l2_angles = {}
    for l1_name, (l1_start, l1_width) in l1_angles.items():
        l2_stats = df_stats[df_stats['一级'] == l1_name].groupby('二级')['资产'].sum().reset_index().sort_values('资产', ascending=False)
        curr_l2_start = l1_start
        base_color = base_colors.get(l1_name, '#A5A5A5')
        for i, row in l2_stats.iterrows():
            pct = row['资产'] / total_val
            width = (row['资产'] / l2_stats['资产'].sum()) * l1_width
            color = matplotlib.colors.to_rgba(base_color, alpha=0.8 - (i % 3) * 0.1)
            ax.add_patch(Wedge((0, 0), outer_r, curr_l2_start - width, curr_l2_start, width=outer_r-mid_r, facecolor=color, edgecolor='w'))
            if width > 9:
                mid_a = curr_l2_start - width / 2
                rad = np.deg2rad(mid_a)
                fontsize = 9 if width < 14 else 10
                ax.text(
                    (mid_r + outer_r) / 2 * np.cos(rad),
                    (mid_r + outer_r) / 2 * np.sin(rad),
                    f"{row['二级']}\n{pct:.1%}",
                    ha='center',
                    va='center',
                    color='white',
                    fontsize=fontsize,
                    weight='bold',
                    fontproperties=CH_FONT,
                )
            l2_angles[row['二级']] = (curr_l2_start, width, base_color)
            curr_l2_start -= width

    # 3. 第三层 (三级分类)
    placed_y = {"left": [], "right": []}
    min_y_gap = 0.06
    for l2_name, (l2_start, l2_width, b_color) in l2_angles.items():
        l3_stats = df_stats[df_stats['二级'] == l2_name].groupby('三级')['资产'].sum().reset_index().sort_values('资产', ascending=False)
        curr_l3_start = l2_start
        for i, row in l3_stats.iterrows():
            pct = row['资产'] / total_val
            width = (row['资产'] / l3_stats['资产'].sum()) * l2_width
            color = matplotlib.colors.to_rgba(b_color, alpha=0.5 - (i % 3) * 0.1)
            ax.add_patch(Wedge((0, 0), labels_r, curr_l3_start - width, curr_l3_start, width=labels_r-outer_r, facecolor=color, edgecolor='w'))
            if pct > 0.003:
                mid_a = curr_l3_start - width / 2
                rad = np.deg2rad(mid_a)
                name = str(row['三级'])
                label = f"{name}\n{pct:.1%}"

                required_deg = 10 + len(name) * 1.5
                inside_ok = width >= required_deg and pct >= 0.004
                if inside_ok:
                    label_r = (outer_r + labels_r) / 2
                    tx, ty = label_r * np.cos(rad), label_r * np.sin(rad)
                    fontsize = int(round(min(13, max(9, 10 + (width - required_deg) / 18))))
                    ax.text(tx, ty, label, ha='center', va='center', fontsize=fontsize, color='white', weight='bold', fontproperties=CH_FONT)
                else:
                    anchor_x, anchor_y = labels_r * np.cos(rad), labels_r * np.sin(rad)
                    elbow_x, elbow_y = (labels_r + 0.05) * np.cos(rad), (labels_r + 0.05) * np.sin(rad)

                    side = "right" if np.cos(rad) >= 0 else "left"
                    label_x = (labels_r + 0.22) if side == "right" else -(labels_r + 0.22)
                    label_y = elbow_y
                    step = min_y_gap
                    for _ in range(80):
                        if all(abs(label_y - y0) >= min_y_gap for y0 in placed_y[side]):
                            break
                        label_y = label_y + step if label_y >= 0 else label_y - step
                        if label_y > 1.18:
                            label_y = 1.18
                            step = -step
                        if label_y < -1.18:
                            label_y = -1.18
                            step = -step
                    placed_y[side].append(label_y)

                    ax.plot([anchor_x, elbow_x, label_x], [anchor_y, elbow_y, label_y], color='gray', lw=0.5)
                    ha = 'left' if side == "right" else 'right'
                    ax.text(label_x, label_y, label, ha=ha, va='center', fontsize=10, color='#333333', weight='bold', fontproperties=CH_FONT)
    ax.text(0, 0, "资产配置", ha='center', va='center', fontsize=22, weight='bold', color='#2C3E50', fontproperties=CH_FONT)
    return fig

# --- UI 界面 ---
st.title("📊 基金资产穿透分析系统")
st.markdown("上传基金持仓 Excel 文件，系统将自动分类并生成三层旭日图。您可以手动微调分类结果。")

if not os.path.exists(RULES_FILE):
    st.error(f"找不到规则文件: {RULES_FILE}")
else:
    classifier = FundClassifier(RULES_FILE)

    uploaded_file = st.file_uploader("选择 场外基金持仓 文件", type=["xlsx"])
    uploaded_market_file = st.file_uploader("选择 场内持仓汇总 文件（可选，仅统计ETF）", type=["xlsx"], key="market_file")

    if uploaded_file:
        # 缓存数据处理
        file_id = f"{uploaded_file.name}:{uploaded_file.size}"
        if ('df' not in st.session_state) or (st.session_state.get("file_id") != file_id):
            with st.spinner("正在读取并分类..."):
                df = pd.read_excel(uploaded_file, sheet_name='持有信息', header=4, skipfooter=7)
                df = df.dropna(subset=['基金代码']).copy()
                col_asset = '资产情况\n（结算币种）'
                df['基金代码'] = df['基金代码'].apply(lambda x: str(int(x)).zfill(6))
                df['资产金额'] = pd.to_numeric(df[col_asset], errors='coerce').fillna(0)
                df = df[['基金代码', '基金名称', '资产金额']].copy()
                df = aggregate_holdings(df, min_amount=10.0)
                
                # 初始自动分类
                classified = df.apply(lambda r: classifier.classify(r.get('基金名称')), axis=1)
                df['一级'], df['二级'], df['三级'], df['匹配状态'] = zip(*classified)
                st.session_state.df = df[['基金代码', '基金名称', '资产金额', '一级', '二级', '三级', '匹配状态']]
                st.session_state.df = apply_overrides(st.session_state.df)
                st.session_state.df = st.session_state.df.sort_values('资产金额', ascending=False).reset_index(drop=True)
                st.session_state.file_id = file_id

        if uploaded_market_file:
            market_file_id = f"{uploaded_market_file.name}:{uploaded_market_file.size}"
            if ('market_df' not in st.session_state) or (st.session_state.get("market_file_id") != market_file_id):
                with st.spinner("正在读取场内持仓..."):
                    parsed, sheet, cols = parse_market_holdings(uploaded_market_file)
                    if parsed is None:
                        st.error(f"场内持仓文件解析失败（工作表：{sheet}）。需要包含列：代码、名称、持有金额。当前列：{cols}")
                        st.session_state.market_df = None
                    else:
                        classified_m = parsed.apply(lambda r: classifier.classify(r.get('基金名称')), axis=1)
                        parsed['一级'], parsed['二级'], parsed['三级'], parsed['匹配状态'] = zip(*classified_m)
                        parsed['来源'] = "场内"
                        parsed = apply_overrides(parsed)
                        st.session_state.market_df = parsed[['基金代码', '基金名称', '资产金额', '一级', '二级', '三级', '匹配状态', '来源']]
                        st.session_state.market_df = st.session_state.market_df.sort_values('资产金额', ascending=False).reset_index(drop=True)
                        st.session_state.market_file_id = market_file_id
        else:
            st.session_state.market_df = None

        # 交互式编辑
        st.subheader("🔍 分类核对与微调")
        
        # 统计异常情况
        unmatched_count = len(st.session_state.df[st.session_state.df['匹配状态'].isin(['AI猜测', '待分类'])])
        if unmatched_count > 0:
            st.warning(f"⚠️ 发现 {unmatched_count} 条分类不确定的记录（AI猜测或未匹配），请人工核对并调整。")
            st.markdown("""
                <style>
                .stDataFrame div[data-testid="stTable"] tr {
                    color: black !important;
                }
                /* 模拟标红效果：针对 st.data_editor 很难直接改色，我们通过提示文字和状态列提醒 */
                </style>
                """, unsafe_allow_html=True)

        # 增加一个状态列（带颜色圆点），用于直观显示
        df_for_editor = st.session_state.df.sort_values('资产金额', ascending=False).reset_index(drop=True).copy()
        df_for_editor['确认状态'] = df_for_editor['匹配状态'].apply(lambda x: "🟢 准确" if x == "规则匹配" else "🔴 待确认")
        
        edited_df = st.data_editor(
            df_for_editor,
            column_config={
                "确认状态": st.column_config.TextColumn(disabled=True),
                "匹配状态": st.column_config.TextColumn(disabled=True),
                "资产金额": st.column_config.NumberColumn(format="¥%.2f"),
                "一级": st.column_config.SelectboxColumn(options=classifier.l1_list),
                "二级": st.column_config.SelectboxColumn(options=classifier.l2_list),
                "三级": st.column_config.SelectboxColumn(options=classifier.l3_list),
            },
            disabled=["基金代码", "基金名称", "资产金额", "确认状态", "匹配状态"],
            hide_index=True,
            use_container_width=True,
            key="data_editor"
        )

        # 更新 session_state (去掉用于展示的辅助列)
        final_edited_df = edited_df.drop(columns=['确认状态'])
        if not final_edited_df.equals(st.session_state.df):
            old_df = st.session_state.df.copy()
            old_map = old_df.set_index('基金代码')[['一级', '二级', '三级']]
            new_map = final_edited_df.set_index('基金代码')[['一级', '二级', '三级']]
            common = new_map.index.intersection(old_map.index)
            diff_mask = (new_map.loc[common] != old_map.loc[common]).any(axis=1)
            changed_codes = set(common[diff_mask].tolist())
            final_edited_df.loc[final_edited_df['基金代码'].isin(changed_codes), '匹配状态'] = "人工修正"
            st.session_state.df = final_edited_df.sort_values('资产金额', ascending=False).reset_index(drop=True)

        c1, c2 = st.columns([1, 2])
        with c1:
            if st.button("保存人工修正", use_container_width=True):
                save_src = st.session_state.df
                if st.session_state.get("market_df") is not None:
                    save_src = pd.concat([save_src, st.session_state.market_df.drop(columns=['来源'])], ignore_index=True)
                    save_src = save_src.drop_duplicates(subset=['基金代码'], keep='last')
                save_overrides(save_src)
                st.success("已保存人工修正，下次导入会自动回灌。")
        with c2:
            st.caption("说明：修正将保存到本机隐藏文件，不需要你手工提供该文件。")

        if st.session_state.get("market_df") is not None:
            st.divider()
            st.subheader("🏦 场内持仓情况")
            market_df_for_editor = st.session_state.market_df.sort_values('资产金额', ascending=False).reset_index(drop=True).copy()
            market_df_for_editor['确认状态'] = market_df_for_editor['匹配状态'].apply(lambda x: "🟢 准确" if x in ["规则匹配", "历史修正"] else "🔴 待确认")
            edited_market = st.data_editor(
                market_df_for_editor,
                column_config={
                    "确认状态": st.column_config.TextColumn(disabled=True),
                    "匹配状态": st.column_config.TextColumn(disabled=True),
                    "来源": st.column_config.TextColumn(disabled=True),
                    "资产金额": st.column_config.NumberColumn(format="¥%.2f"),
                    "一级": st.column_config.SelectboxColumn(options=classifier.l1_list),
                    "二级": st.column_config.SelectboxColumn(options=classifier.l2_list),
                    "三级": st.column_config.SelectboxColumn(options=classifier.l3_list),
                },
                disabled=["基金代码", "基金名称", "资产金额", "确认状态", "匹配状态", "来源"],
                hide_index=True,
                use_container_width=True,
                key="market_data_editor"
            )
            final_market = edited_market.drop(columns=['确认状态'])
            if not final_market.equals(st.session_state.market_df):
                old_market = st.session_state.market_df.copy()
                old_map = old_market.set_index('基金代码')[['一级', '二级', '三级']]
                new_map = final_market.set_index('基金代码')[['一级', '二级', '三级']]
                common = new_map.index.intersection(old_map.index)
                diff_mask = (new_map.loc[common] != old_map.loc[common]).any(axis=1)
                changed_codes = set(common[diff_mask].tolist())
                final_market.loc[final_market['基金代码'].isin(changed_codes), '匹配状态'] = "人工修正"
                st.session_state.market_df = final_market.sort_values('资产金额', ascending=False).reset_index(drop=True)

        # 生成图表
        st.divider()
        st.subheader("🎡 资产分布旭日图")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            df_stats = st.session_state.df.rename(columns={'资产金额': '资产'})
            if st.session_state.get("market_df") is not None:
                market_stats = st.session_state.market_df.drop(columns=['来源']).rename(columns={'资产金额': '资产'})
                df_stats = pd.concat([df_stats, market_stats], ignore_index=True)
            fig = get_sunburst_plot(df_stats, df_stats['资产'].sum())
            st.pyplot(fig)
            
        with col2:
            st.write("### 统计摘要")
            if CH_FONT_NAME:
                st.caption(f"当前图形字体: {CH_FONT_NAME}")
            summary = df_stats.groupby('一级', as_index=False)['资产'].sum()
            summary['资产'] = pd.to_numeric(summary['资产'], errors='coerce').fillna(0)
            summary = summary.sort_values('资产', ascending=False, kind='mergesort').reset_index(drop=True)
            summary['占比'] = (summary['资产'] / summary['资产'].sum() * 100).map('{:.2f}%'.format)
            st.dataframe(summary, hide_index=True)
            
            # 下载按钮
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=200, bbox_inches='tight')
            st.download_button(
                label="📥 下载高清图表",
                data=buf.getvalue(),
                file_name=f"基金穿透分析_{datetime.now().strftime('%Y%m%d')}.png",
                mime="image/png"
            )
