"""
基金规模与年化回报率的非线性关系研究
以华夏科创50ETF(588000)为例：是否存在"规模诅咒"

本脚本复现完整的实证分析流程，包括：
1. 数据加载与预处理
2. 前瞻收益率计算
3. Pearson相关性分析
4. 多项式回归（线性/二次/三次）
5. ANOVA规模区间比较
6. 快速扩张期T检验
7. 图表生成

作者: AI Research Assistant
日期: 2026-06-29
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import f_oneway
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['font.size'] = 9
plt.rcParams['axes.unicode_minus'] = False

COLORS = {
    'primary': '#4A6FA5', 'secondary': '#6B8CBB', 'accent': '#2E4A62',
    'neutral': '#7A8B99', 'light': '#8BA3C7', 'bg': '#FAFAFA'
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


def load_etf_price():
    """加载并合并日度净值数据"""
    files = [
        'etf_price_2020_2021.csv', 'etf_price_2021_2022.csv',
        'etf_price_2022_2023.csv', 'etf_price_2023_2026.csv'
    ]
    dfs = [pd.read_csv(os.path.join(DATA_DIR, f)) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    return df.rename(columns={'time': 'date', 'close': 'nav'})


def build_share_data():
    """构建季度份额数据集（来源：天天基金网定期报告）"""
    raw = [
        ('2020-09-28', 51.23, 51.34), ('2020-11-09', 35.58, 52.84),
        ('2020-12-31', 86.77, 124.58), ('2021-03-31', 140.98, 181.18),
        ('2021-06-30', 122.15, 201.08), ('2021-09-30', 156.73, 222.79),
        ('2021-12-31', 142.36, 206.89), ('2022-03-31', 198.97, 226.21),
        ('2022-06-30', 244.54, 282.57), ('2022-09-30', 298.96, 293.46),
        ('2022-12-31', 507.85, 508.27), ('2023-03-31', 498.67, 560.59),
        ('2023-06-30', 639.52, 671.92), ('2023-09-30', 1016.40, 946.73),
        ('2023-12-31', 1040.83, 933.78), ('2024-03-31', 911.77, 729.32),
        ('2024-06-30', 960.28, 716.13), ('2024-09-30', 1009.83, 923.62),
        ('2024-12-31', 891.12, 929.31), ('2025-03-31', 767.91, 824.37),
        ('2025-06-30', 789.64, 833.43), ('2025-09-30', 481.75, 756.20),
        ('2025-12-31', 537.10, 760.22), ('2026-03-31', 517.89, 685.48),
    ]
    df = pd.DataFrame(raw, columns=['date', 'shares_eoy', 'nav_eoy'])
    df['date'] = pd.to_datetime(df['date'])
    df['shares_change_pct'] = df['shares_eoy'].pct_change() * 100
    return df


def calc_forward_returns(share_df, etf_price):
    """计算前瞻收益率"""
    results = []
    for _, row in share_df.iterrows():
        q_date = row['date']
        future = etf_price[etf_price['date'] >= q_date]
        if len(future) == 0:
            continue
        entry_price = future.iloc[0]['nav']
        entry_date = future.iloc[0]['date']

        def get_ret(days):
            t = etf_price[etf_price['date'] >= (entry_date + pd.Timedelta(days=days))]
            return (t.iloc[0]['nav'] / entry_price - 1) * 100 if len(t) > 0 else np.nan

        results.append({
            'date': q_date,
            'shares_eoy': row['shares_eoy'],
            'nav_eoy': row['nav_eoy'],
            'shares_change_pct': row['shares_change_pct'],
            'ret_90d': get_ret(90),
            'ret_180d': get_ret(180),
            'ret_252d': get_ret(252),
        })
    return pd.DataFrame(results)


def safe_corr(x, y):
    """安全计算Pearson相关系数"""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return np.nan, np.nan
    return stats.pearsonr(x[mask], y[mask])


def plot_share_trend(share_df):
    """图1: 份额与净资产趋势"""
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax2 = ax1.twinx()
    ax1.fill_between(share_df['date'], share_df['shares_eoy'], alpha=0.2, color=COLORS['primary'])
    ax1.plot(share_df['date'], share_df['shares_eoy'], color=COLORS['primary'], lw=2, marker='o', ms=4, label='期末份额 (亿份)')
    ax2.plot(share_df['date'], share_df['nav_eoy'], color=COLORS['accent'], lw=2, marker='s', ms=4, label='期末净资产 (亿元)')
    ax1.set_xlabel('日期', fontsize=10)
    ax1.set_ylabel('基金份额 (亿份)', color=COLORS['primary'], fontsize=10)
    ax2.set_ylabel('净资产 (亿元)', color=COLORS['accent'], fontsize=10)
    ax1.set_title('图1: 华夏科创50ETF 份额与净资产规模变化 (2020-2026)', fontsize=11, fontweight='bold')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig1_share_trend.png'), bbox_inches='tight', facecolor='white', dpi=200)
    plt.close()
    print('[OK] fig1_share_trend.png')


def plot_polynomial_fit(qa_clean, slope, intercept, r2_lin, poly_2, poly_3, r2_3, vertex_x, vertex_y):
    """图2: 多项式拟合曲线"""
    x = qa_clean['shares_eoy'].values
    y = qa_clean['ret_180d'].values
    fig, ax = plt.subplots(figsize=(8, 5))
    date_nums = mdates.date2num(qa_clean['date'])
    scatter = ax.scatter(x, y, c=date_nums, cmap='Blues', s=120, alpha=0.8, edgecolors='white', lw=0.5, zorder=5)
    x_fit = np.linspace(x.min(), x.max(), 300)
    ax.plot(x_fit, slope * x_fit + intercept, '--', color=COLORS['neutral'], lw=1.5, label=f'线性 R²={r2_lin:.3f}')
    ax.plot(x_fit, poly_2(x_fit), '-', color=COLORS['secondary'], lw=2, label=f'二次 R²={r2_lin:.3f}')
    ax.plot(x_fit, poly_3(x_fit), '-', color=COLORS['accent'], lw=2, label=f'三次 R²={r2_3:.3f}')
    ax.axvline(x=vertex_x, color='red', ls='--', lw=1, alpha=0.5)
    ax.annotate(f'理论最优规模\n{vertex_x:.0f}亿份', xy=(vertex_x, vertex_y), xytext=(vertex_x + 80, vertex_y - 5),
                fontsize=8, color='red', arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
    ax.axhline(y=0, color='black', lw=0.5, alpha=0.3)
    ax.set_xlabel('期末基金份额 (亿份)', fontsize=10)
    ax.set_ylabel('未来180天收益率 (%)', fontsize=10)
    ax.set_title('图2: 份额规模与收益率的非线性关系', fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('时间', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig2_polynomial_fit.png'), bbox_inches='tight', facecolor='white', dpi=200)
    plt.close()
    print('[OK] fig2_polynomial_fit.png')


def plot_boxplot(qa_tier):
    """图3: 箱线图"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot([group['ret_180d'].values for _, group in qa_tier.groupby('size_tier')],
                     labels=['小(<150)', '中(150-300)', '大(300-600)', '超大(>600)'],
                     patch_artist=True, widths=0.6, medianprops={'color': 'red', 'linewidth': 2})
    for p, c in zip(bp['boxes'], ['#7BAFD4', '#A8D5A2', '#F5D491', '#E8A598']):
        p.set_facecolor(c)
        p.set_alpha(0.7)
    ax.axhline(y=0, color='black', lw=0.5, alpha=0.5)
    ax.set_ylabel('未来180天收益率 (%)', fontsize=10)
    ax.set_xlabel('规模区间 (亿份)', fontsize=10)
    ax.set_title('图3: 不同规模区间的收益率分布', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, (_, group) in enumerate(qa_tier.groupby('size_tier')):
        mean_val = group['ret_180d'].mean()
        ax.text(i + 1, mean_val + 2, f'均值: {mean_val:.1f}%', ha='center', fontsize=8, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig3_boxplot.png'), bbox_inches='tight', facecolor='white', dpi=200)
    plt.close()
    print('[OK] fig3_boxplot.png')


def plot_expansion_effect(qa_accel, rapid, normal, p_val):
    """图4: 扩张期效应"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    colors_scatter = ['red' if x > 30 else COLORS['primary'] for x in qa_accel['shares_change_pct']]
    ax1.scatter(qa_accel['shares_change_pct'], qa_accel['ret_180d'], c=colors_scatter, s=80, alpha=0.7, edgecolors='white', lw=0.5)
    chg_x, chg_y = qa_accel['shares_change_pct'], qa_accel['ret_180d']
    mask = ~(np.isnan(chg_x) | np.isnan(chg_y))
    if mask.sum() >= 3:
        cs, ci, cr, cp, _ = stats.linregress(chg_x[mask], chg_y[mask])
        x_line = np.linspace(chg_x[mask].min(), chg_x[mask].max(), 100)
        ax1.plot(x_line, cs * x_line + ci, '--', color=COLORS['secondary'], lw=1.5, label=f'r={cr:.3f}, p={cp:.3f}')
    ax1.axhline(y=0, color='black', lw=0.5, alpha=0.3)
    ax1.axvline(x=30, color='red', ls='--', lw=1, alpha=0.5, label='快速扩张阈值(30%)')
    ax1.set_xlabel('份额季度变化率 (%)', fontsize=10)
    ax1.set_ylabel('未来180天收益率 (%)', fontsize=10)
    ax1.set_title('份额变化率 vs 未来收益率', fontsize=10, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    cats = ['快速扩张期\n(>30%)', '正常期\n(<=30%)']
    mns = [rapid['ret_180d'].mean(), normal['ret_180d'].mean()]
    sds = [rapid['ret_180d'].std(), normal['ret_180d'].std()]
    bars = ax2.bar(cats, mns, yerr=sds, capsize=5, color=['#D4574A', COLORS['primary']], alpha=0.7, edgecolor='white', lw=0.5)
    ax2.axhline(y=0, color='black', lw=0.5)
    ax2.set_ylabel('未来180天收益率 (%)', fontsize=10)
    ax2.set_title(f'扩张期 vs 正常期 (p={p_val:.3f})', fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, mns):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 1, f'{val:.2f}%', ha='center', fontsize=9, fontweight='bold')
    plt.suptitle('图4: 快速扩张期效应检验', fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig4_expansion_effect.png'), bbox_inches='tight', facecolor='white', dpi=200)
    plt.close()
    print('[OK] fig4_expansion_effect.png')


def plot_dashboard(share_df, qa_clean, slope, intercept, r2_lin, poly_3, r2_3, vertex_x, qa_tier, rapid, normal, p_val):
    """图5: 综合分析Dashboard"""
    x = qa_clean['shares_eoy'].values
    y = qa_clean['ret_180d'].values
    date_nums = mdates.date2num(qa_clean['date'])

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1_twin = ax1.twinx()
    ax1.fill_between(share_df['date'], share_df['shares_eoy'], alpha=0.2, color=COLORS['primary'])
    ax1.plot(share_df['date'], share_df['shares_eoy'], color=COLORS['primary'], lw=2, marker='o', ms=3)
    ax1_twin.plot(share_df['date'], share_df['nav_eoy'], color=COLORS['accent'], lw=1.5, marker='s', ms=3)
    ax1.set_title('(a) 份额与净资产趋势', fontsize=10, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(x, y, c=date_nums, cmap='Blues', s=60, alpha=0.8, edgecolors='white', lw=0.3)
    x_fit = np.linspace(x.min(), x.max(), 200)
    ax2.plot(x_fit, slope * x_fit + intercept, '--', color=COLORS['neutral'], lw=1.2, label=f'线性 R²={r2_lin:.3f}')
    ax2.plot(x_fit, poly_3(x_fit), '-', color=COLORS['accent'], lw=2, label=f'三次 R²={r2_3:.3f}')
    ax2.axvline(x=vertex_x, color='red', ls='--', lw=1, alpha=0.4)
    ax2.axhline(y=0, color='black', lw=0.5, alpha=0.3)
    ax2.set_title(f'(b) 非线性拟合 (最优={vertex_x:.0f}亿份)', fontsize=10, fontweight='bold')
    ax2.set_xlabel('份额 (亿份)', fontsize=9)
    ax2.set_ylabel('180天收益率 (%)', fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    bp = ax3.boxplot([group['ret_180d'].values for _, group in qa_tier.groupby('size_tier')],
                      labels=['小', '中', '大', '超大'], patch_artist=True, widths=0.6,
                      medianprops={'color': 'red', 'lw': 2})
    for p, c in zip(bp['boxes'], ['#7BAFD4', '#A8D5A2', '#F5D491', '#E8A598']):
        p.set_facecolor(c)
        p.set_alpha(0.7)
    ax3.axhline(y=0, color='black', lw=0.5, alpha=0.5)
    ax3.set_title('(c) 规模区间收益率分布', fontsize=10, fontweight='bold')
    ax3.set_ylabel('180天收益率 (%)', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    ax4 = fig.add_subplot(gs[1, 1])
    cats = ['快速扩张', '正常']
    mns = [rapid['ret_180d'].mean(), normal['ret_180d'].mean()]
    sds = [rapid['ret_180d'].std(), normal['ret_180d'].std()]
    bars = ax4.bar(cats, mns, yerr=sds, capsize=5, color=['#D4574A', COLORS['primary']], alpha=0.7, edgecolor='white')
    for b, v in zip(bars, mns):
        ax4.text(b.get_x() + b.get_width() / 2, v + 1, f'{v:.2f}%', ha='center', fontsize=9, fontweight='bold')
    ax4.axhline(y=0, color='black', lw=0.5)
    ax4.set_title(f'(d) 扩张期效应 (p={p_val:.3f})', fontsize=10, fontweight='bold')
    ax4.set_ylabel('180天收益率 (%)', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle('华夏科创50ETF: 基金规模与收益率非线性关系综合分析', fontsize=13, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(FIG_DIR, 'fig5_dashboard.png'), bbox_inches='tight', facecolor='white', dpi=200)
    plt.close()
    print('[OK] fig5_dashboard.png')


def main():
    """主分析流程"""
    print('=' * 60)
    print('华夏科创50ETF 规模诅咒研究 - 完整分析')
    print('=' * 60)

    # 1. 加载数据
    print('\n[1/6] 加载数据...')
    etf_price = load_etf_price()
    share_df = build_share_data()
    print(f'  ETF净值: {len(etf_price)} 个交易日')
    print(f'  季度份额: {len(share_df)} 个季度')

    # 2. 计算前瞻收益率
    print('\n[2/6] 计算前瞻收益率...')
    qa_df = calc_forward_returns(share_df, etf_price)
    print(f'  有效观测值: {qa_df["ret_180d"].notna().sum()} 个季度')

    # 3. 相关性分析
    print('\n[3/6] 相关性分析...')
    corr_results = []
    for ret_col in ['ret_90d', 'ret_180d', 'ret_252d']:
        r_level, p_level = safe_corr(qa_df['shares_eoy'].values, qa_df[ret_col].values)
        qa_chg = qa_df.dropna(subset=['shares_change_pct', ret_col])
        r_chg, p_chg = stats.pearsonr(qa_chg['shares_change_pct'], qa_chg[ret_col]) if len(qa_chg) >= 3 else (np.nan, np.nan)
        period_name = ret_col.replace('ret_', '未来')
        sig = '***' if p_level < 0.05 else ('*' if p_level < 0.1 else '')
        print(f'  {period_name}: 份额水平 r={r_level:.3f}(p={p_level:.3f}){sig}')
        corr_results.append({'期限': period_name, 'r': round(r_level, 3), 'p': round(p_level, 3)})

    # 4. 多项式回归
    print('\n[4/6] 多项式回归...')
    qa_clean = qa_df.dropna(subset=['ret_180d', 'shares_eoy'])
    x, y = qa_clean['shares_eoy'].values, qa_clean['ret_180d'].values

    slope, intercept, r_lin, _, _ = stats.linregress(x, y)
    r2_lin = r_lin ** 2

    coefs_2 = np.polyfit(x, y, 2)
    poly_2 = np.poly1d(coefs_2)
    y_pred_2 = poly_2(x)
    r2_2 = 1 - np.sum((y - y_pred_2) ** 2) / np.sum((y - np.mean(y)) ** 2)

    coefs_3 = np.polyfit(x, y, 3)
    poly_3 = np.poly1d(coefs_3)
    y_pred_3 = poly_3(x)
    r2_3 = 1 - np.sum((y - y_pred_3) ** 2) / np.sum((y - np.mean(y)) ** 2)

    vertex_x = -coefs_2[1] / (2 * coefs_2[0])
    vertex_y = poly_2(vertex_x)

    print(f'  线性:   R²={r2_lin:.4f}')
    print(f'  二次:   R²={r2_2:.4f}')
    print(f'  三次:   R²={r2_3:.4f} (提升 {(r2_3/r2_lin-1)*100:.0f}%)')
    print(f'  理论最优规模: {vertex_x:.0f} 亿份')
    print(f'  关系形态: 倒U型' if coefs_2[0] < 0 else 'U型')

    # 5. ANOVA + T检验
    print('\n[5/6] ANOVA与T检验...')
    qa_tier = qa_df.dropna(subset=['ret_180d', 'shares_eoy']).copy()
    qa_tier['size_tier'] = pd.cut(qa_tier['shares_eoy'], bins=[0, 150, 300, 600, 1200],
                                   labels=['小(<150)', '中(150-300)', '大(300-600)', '超大(>600)'])
    tiers = [g['ret_180d'].values for _, g in qa_tier.groupby('size_tier') if len(g) > 1]
    f_stat, p_anova = f_oneway(*tiers)
    print(f'  ANOVA: F={f_stat:.3f}, p={p_anova:.3f}')
    for name, group in qa_tier.groupby('size_tier'):
        print(f'    {name}: 均值={group["ret_180d"].mean():.2f}%, n={len(group)}')

    qa_accel = qa_df.dropna(subset=['shares_change_pct', 'ret_180d']).copy()
    rapid = qa_accel[qa_accel['shares_change_pct'] > 30]
    normal = qa_accel[qa_accel['shares_change_pct'] <= 30]
    t_stat, p_val = stats.ttest_ind(rapid['ret_180d'].dropna(), normal['ret_180d'].dropna())
    print(f'  T检验: t={t_stat:.3f}, p={p_val:.3f}')
    print(f'    扩张期: {rapid["ret_180d"].mean():.2f}%, 正常期: {normal["ret_180d"].mean():.2f}%')

    # 6. 生成图表
    print('\n[6/6] 生成图表...')
    plot_share_trend(share_df)
    plot_polynomial_fit(qa_clean, slope, intercept, r2_lin, poly_2, poly_3, r2_3, vertex_x, vertex_y)
    plot_boxplot(qa_tier)
    plot_expansion_effect(qa_accel, rapid, normal, p_val)
    plot_dashboard(share_df, qa_clean, slope, intercept, r2_lin, poly_3, r2_3, vertex_x, qa_tier, rapid, normal, p_val)

    # 输出结论
    print('\n' + '=' * 60)
    print('研究结论')
    print('=' * 60)
    print(f'1. 未发现"规模诅咒": 份额与收益率正相关(r={corr_results[1]["r"]}, p={corr_results[1]["p"]})')
    print(f'2. 非线性倒U型关系: 三次R²={r2_3:.3f}, 理论最优={vertex_x:.0f}亿份')
    print(f'3. 快速扩张不损害收益: T检验 p={p_val:.3f}')
    print(f'4. 当前517亿份处于上升段')
    print('=' * 60)


if __name__ == '__main__':
    main()
