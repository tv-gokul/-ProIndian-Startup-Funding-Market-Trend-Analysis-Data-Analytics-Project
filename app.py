import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import numpy as np

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG — must be the first Streamlit command
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="ProIndian Market Trends", layout="wide")

# ═══════════════════════════════════════════════════════════════
# 1. DATA LOADING VIA SQL
# ═══════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    conn = sqlite3.connect('data/proindian_funding.db')
    df = pd.read_sql_query("SELECT * FROM startup_funding WHERE Year > 0", conn)
    conn.close()
    return df

df = load_data()

# ═══════════════════════════════════════════════════════════════
# 2. HEADER
# ═══════════════════════════════════════════════════════════════
st.title("🇮🇳 ProIndian: Startup Funding & Market Trend Analysis")
st.markdown(
    "An interactive dashboard analyzing **{:,} startup funding records** "
    "to uncover sector-wise, city-wise, and year-wise investment trends.".format(len(df))
)

# ═══════════════════════════════════════════════════════════════
# 3. SIDEBAR FILTERs
# ═══════════════════════════════════════════════════════════════
st.sidebar.header("🔎 Filter Data")
industry_list = sorted(df['IndustryVertical'].dropna().unique().tolist())
city_list = sorted(df['CityLocation'].dropna().unique().tolist())
invest_type_list = sorted(df['InvestmentType'].dropna().unique().tolist())

selected_year = st.sidebar.slider(
    "Funding Year",
    int(df['Year'].min()), int(df['Year'].max()),   
    (int(df['Year'].min()), int(df['Year'].max()))
)
selected_industry = st.sidebar.multiselect("Industry Vertical", industry_list, default=[])
selected_city = st.sidebar.multiselect("City / Location", city_list, default=[])
selected_invest_type = st.sidebar.multiselect("Investment Type", invest_type_list, default=[])

# Apply Filters
filtered_df = df[(df['Year'] >= selected_year[0]) & (df['Year'] <= selected_year[1])]
if selected_industry:
    filtered_df = filtered_df[filtered_df['IndustryVertical'].isin(selected_industry)]
if selected_city:
    filtered_df = filtered_df[filtered_df['CityLocation'].isin(selected_city)]
if selected_invest_type:
    filtered_df = filtered_df[filtered_df['InvestmentType'].isin(selected_invest_type)]

# ═══════════════════════════════════════════════════════════════
# 4. KPI METRICS
# ═══════════════════════════════════════════════════════════════
st.markdown("---")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Deals", f"{len(filtered_df):,}")
col2.metric("Total Investment", f"${filtered_df['AmountInUSD'].sum() / 1e9:.2f}B")
col3.metric("Avg Deal Size", f"${filtered_df['AmountInUSD'].mean() / 1e6:.1f}M")
col4.metric("Max Single Funding", f"${filtered_df['AmountInUSD'].max() / 1e6:.1f}M")
col5.metric("Unique Industries", f"{filtered_df['IndustryVertical'].nunique()}")

# ═══════════════════════════════════════════════════════════════
# 5. EXPLORATORY DATA ANALYSIS — TABS
# ═══════════════════════════════════════════════════════════════
st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Yearly Trends",
    "🏭 Top Industries",
    "🏙️ Geographic Hubs",
    "📊 Funding Distribution",
    "💰 Investment Types",
    "🔍 VC Insights",
])

# ── helper: close all figures after rendering to free memory ──
plt.rcParams.update({'figure.max_open_warning': 0})
sns.set_style("whitegrid")

# ─────────── TAB 1  ·  YEARLY TRENDS ─────────────────────────
with tab1:
    st.subheader("Year-over-Year Funding Analysis")

    yearly = filtered_df.groupby('Year').agg(
        TotalFunding=('AmountInUSD', 'sum'),
        DealCount=('AmountInUSD', 'count'),
        AvgDeal=('AmountInUSD', 'mean'),
    ).reset_index()

    fig1, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Total funding trend (in $M)
    yearly['TotalFunding_M'] = yearly['TotalFunding'] / 1e6
    sns.lineplot(data=yearly, x='Year', y='TotalFunding_M', marker='o', color='#2563eb', ax=axes[0])
    axes[0].fill_between(yearly['Year'], yearly['TotalFunding_M'], alpha=0.15, color='#2563eb')
    axes[0].set_title("Total Funding by Year", fontweight='bold')
    axes[0].set_ylabel("Funding ($M)")
    axes[0].ticklabel_format(style='plain', axis='y')

    # Deal count trend
    sns.barplot(data=yearly, x='Year', y='DealCount', color='#10b981', ax=axes[1])
    axes[1].set_title("Number of Deals by Year", fontweight='bold')
    axes[1].set_ylabel("Deal Count")
    for tick in axes[1].get_xticklabels():
        tick.set_rotation(45)

    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    # Year-over-Year Growth Table
    if len(yearly) > 1:
        yearly['YoY Growth (%)'] = yearly['TotalFunding'].pct_change() * 100
        st.markdown("**Year-over-Year Funding Growth**")
        display_yearly = yearly.copy()
        display_yearly['TotalFunding'] = display_yearly['TotalFunding'].apply(lambda x: f"${x/1e6:,.1f}M")
        display_yearly['AvgDeal'] = display_yearly['AvgDeal'].apply(lambda x: f"${x/1e6:,.1f}M")
        display_yearly['YoY Growth (%)'] = display_yearly['YoY Growth (%)'].apply(
            lambda x: f"{x:+.1f}%" if pd.notna(x) else "—"
        )
        st.dataframe(display_yearly.rename(columns={
            'TotalFunding': 'Total Funding', 'DealCount': 'Deals', 'AvgDeal': 'Avg Deal'
        }), use_container_width=True, hide_index=True)

# ─────────── TAB 2  ·  TOP INDUSTRIES ────────────────────────
with tab2:
    st.subheader("Top Industries by Funding & Deal Volume")

    ind_agg = filtered_df.groupby('IndustryVertical').agg(
        TotalFunding=('AmountInUSD', 'sum'),
        Deals=('AmountInUSD', 'count'),
    ).reset_index()

    top_n = st.slider("Number of industries to display", 5, 25, 10, key='ind_slider')

    col_a, col_b = st.columns(2)

    with col_a:
        top_funding = ind_agg.nlargest(top_n, 'TotalFunding').copy()
        top_funding['TotalFunding_M'] = top_funding['TotalFunding'] / 1e6
        fig2, ax2 = plt.subplots(figsize=(7, 0.45 * top_n + 1))
        sns.barplot(data=top_funding, y='IndustryVertical', x='TotalFunding_M',
                    palette='Blues_d', ax=ax2)
        ax2.set_title(f"Top {top_n} Industries by Funding", fontweight='bold')
        ax2.set_xlabel("Total Funding ($M)")
        ax2.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    with col_b:
        top_deals = ind_agg.nlargest(top_n, 'Deals')
        fig2b, ax2b = plt.subplots(figsize=(7, 0.45 * top_n + 1))
        sns.barplot(data=top_deals, y='IndustryVertical', x='Deals',
                    palette='Greens_d', ax=ax2b)
        ax2b.set_title(f"Top {top_n} Industries by Deal Count", fontweight='bold')
        ax2b.set_xlabel("Number of Deals")
        ax2b.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig2b)
        plt.close(fig2b)

# ─────────── TAB 3  ·  GEOGRAPHIC STARTUP HUBS ───────────────
with tab3:
    st.subheader("Geographic Startup Hubs")

    city_agg = filtered_df.groupby('CityLocation').agg(
        TotalFunding=('AmountInUSD', 'sum'),
        Deals=('AmountInUSD', 'count'),
        AvgDeal=('AmountInUSD', 'mean'),
        UniqueIndustries=('IndustryVertical', 'nunique'),
    ).reset_index()

    top_cities = city_agg.nlargest(10, 'TotalFunding')

    col_c, col_d = st.columns(2)

    top_cities['TotalFunding_M'] = top_cities['TotalFunding'] / 1e6

    with col_c:
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        sns.barplot(data=top_cities, y='CityLocation', x='TotalFunding_M',
                    palette='Oranges_d', ax=ax3)
        ax3.set_title("Top 10 Cities by Total Funding", fontweight='bold')
        ax3.set_xlabel("Total Funding ($M)")
        ax3.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    with col_d:
        fig3b, ax3b = plt.subplots(figsize=(7, 5))
        sns.barplot(data=top_cities, y='CityLocation', x='Deals',
                    palette='Purples_d', ax=ax3b)
        ax3b.set_title("Top 10 Cities by Deal Count", fontweight='bold')
        ax3b.set_xlabel("Number of Deals")
        ax3b.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig3b)
        plt.close(fig3b)

    # City hub summary table
    st.markdown("**City Hub Summary**")
    display_cities = top_cities.copy()
    display_cities['TotalFunding'] = display_cities['TotalFunding'].apply(lambda x: f"${x/1e6:,.1f}M")
    display_cities['AvgDeal'] = display_cities['AvgDeal'].apply(lambda x: f"${x/1e6:,.1f}M")
    st.dataframe(
        display_cities.rename(columns={
            'CityLocation': 'City', 'TotalFunding': 'Total Funding',
            'AvgDeal': 'Avg Deal Size', 'UniqueIndustries': 'Industry Diversity',
        }),
        use_container_width=True, hide_index=True,
    )

# ─────────── TAB 4  ·  FUNDING DISTRIBUTION PATTERNS ─────────
with tab4:
    st.subheader("Funding Distribution Patterns")

    # Only consider non-zero funding for distribution analysis
    funded = filtered_df[filtered_df['AmountInUSD'] > 0]['AmountInUSD']

    col_e, col_f = st.columns(2)

    with col_e:
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        sns.histplot(np.log10(funded + 1), bins=40, kde=True, color='#6366f1', ax=ax4)
        ax4.set_title("Funding Amount Distribution (log₁₀ scale)", fontweight='bold')
        ax4.set_xlabel("log₁₀(Funding in USD)")
        ax4.set_ylabel("Frequency")
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

    with col_f:
        fig4b, ax4b = plt.subplots(figsize=(7, 5))
        top5_ind = filtered_df.groupby('IndustryVertical')['AmountInUSD'].sum().nlargest(5).index
        box_data = filtered_df[
            (filtered_df['IndustryVertical'].isin(top5_ind)) & (filtered_df['AmountInUSD'] > 0)
        ]
        box_data = box_data.copy()
        box_data['AmountInUSD_M'] = box_data['AmountInUSD'] / 1e6
        sns.boxplot(data=box_data, y='IndustryVertical', x='AmountInUSD_M',
                    palette='Set2', ax=ax4b)
        ax4b.set_xscale('log')
        ax4b.set_title("Funding Spread — Top 5 Industries", fontweight='bold')
        ax4b.set_xlabel("Funding ($M, log scale)")
        ax4b.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig4b)
        plt.close(fig4b)

    # Summary stats
    st.markdown("**Descriptive Statistics (non-zero deals, in $M)**")
    desc = (funded / 1e6).describe().to_frame('Value ($M)')
    desc['Value ($M)'] = desc['Value ($M)'].apply(lambda x: f"${x:,.2f}M")
    st.dataframe(desc, use_container_width=True)

# ─────────── TAB 5  ·  INVESTMENT TYPES ──────────────────────
with tab5:
    st.subheader("Funding Breakdown by Investment Type")

    inv_agg = filtered_df.groupby('InvestmentType').agg(
        TotalFunding=('AmountInUSD', 'sum'),
        Deals=('AmountInUSD', 'count'),
    ).reset_index()

    col_g, col_h = st.columns(2)

    with col_g:
        top_inv = inv_agg.nlargest(6, 'TotalFunding')
        fig5, ax5 = plt.subplots(figsize=(7, 7))
        wedges, texts, autotexts = ax5.pie(
            top_inv['TotalFunding'], labels=top_inv['InvestmentType'],
            autopct='%1.1f%%', startangle=90, pctdistance=0.8,
            colors=sns.color_palette('pastel')
        )
        ax5.set_title("Funding Share by Investment Type", fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)

    with col_h:
        fig5b, ax5b = plt.subplots(figsize=(7, 7))
        wedges2, texts2, autotexts2 = ax5b.pie(
            top_inv['Deals'], labels=top_inv['InvestmentType'],
            autopct='%1.1f%%', startangle=90, pctdistance=0.8,
            colors=sns.color_palette('Set3')
        )
        ax5b.set_title("Deal Count by Investment Type", fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig5b)
        plt.close(fig5b)

# ─────────── TAB 6  ·  VC-STYLE BUSINESS INSIGHTS ────────────
with tab6:
    st.subheader("🔍 Business & VC-Style Insights")
    st.markdown(
        "Actionable insights on **emerging sectors**, **declining industries**, "
        "and **regional funding disparities** — modeled after venture capital analysis."
    )

    # ── 6A. EMERGING vs DECLINING SECTORS ───────────────────────
    st.markdown("### 🚀 Emerging & 📉 Declining Sectors")

    years = sorted(filtered_df['Year'].unique())
    if len(years) >= 2:
        mid = len(years) // 2
        early_years = years[:mid]
        late_years = years[mid:]

        early = filtered_df[filtered_df['Year'].isin(early_years)]
        late = filtered_df[filtered_df['Year'].isin(late_years)]

        early_agg = early.groupby('IndustryVertical')['AmountInUSD'].sum()
        late_agg = late.groupby('IndustryVertical')['AmountInUSD'].sum()

        growth = pd.DataFrame({
            'EarlyPeriod': early_agg,
            'LatePeriod': late_agg,
        }).fillna(0)
        growth = growth[(growth['EarlyPeriod'] > 0) | (growth['LatePeriod'] > 0)]
        growth['GrowthPct'] = np.where(
            growth['EarlyPeriod'] > 0,
            ((growth['LatePeriod'] - growth['EarlyPeriod']) / growth['EarlyPeriod']) * 100,
            np.where(growth['LatePeriod'] > 0, 100.0, 0.0),
        )

        col_i, col_j = st.columns(2)

        with col_i:
            st.markdown(f"**🚀 Top Emerging Sectors** ({early_years[0]}–{early_years[-1]} → {late_years[0]}–{late_years[-1]})")
            emerging = growth.nlargest(8, 'GrowthPct')
            fig6a, ax6a = plt.subplots(figsize=(7, 4))
            colors_em = ['#22c55e' if v >= 0 else '#ef4444' for v in emerging['GrowthPct']]
            ax6a.barh(emerging.index, emerging['GrowthPct'], color=colors_em)
            ax6a.set_xlabel("Funding Growth (%)")
            ax6a.set_title("Fastest Growing Sectors", fontweight='bold')
            ax6a.axvline(0, color='gray', linewidth=0.8)
            plt.tight_layout()
            st.pyplot(fig6a)
            plt.close(fig6a)

        with col_j:
            st.markdown(f"**📉 Declining Sectors** ({early_years[0]}–{early_years[-1]} → {late_years[0]}–{late_years[-1]})")
            declining = growth[growth['GrowthPct'] < 0].nsmallest(8, 'GrowthPct')
            if not declining.empty:
                fig6b, ax6b = plt.subplots(figsize=(7, 4))
                ax6b.barh(declining.index, declining['GrowthPct'], color='#ef4444')
                ax6b.set_xlabel("Funding Decline (%)")
                ax6b.set_title("Sectors Losing Momentum", fontweight='bold')
                ax6b.axvline(0, color='gray', linewidth=0.8)
                plt.tight_layout()
                st.pyplot(fig6b)
                plt.close(fig6b)
            else:
                st.info("No declining sectors detected in the selected period.")

        # Insight callouts
        if not emerging.empty:
            top_em = emerging.index[0]
            top_em_pct = emerging['GrowthPct'].iloc[0]
            st.success(
                f"**Emerging Insight:** *{top_em}* has shown the strongest momentum "
                f"with **{top_em_pct:+,.0f}%** funding growth between the early and late periods."
            )
        if not declining.empty:
            top_dec = declining.index[0]
            top_dec_pct = declining['GrowthPct'].iloc[0]
            st.warning(
                f"**Decline Alert:** *{top_dec}* has contracted by **{top_dec_pct:,.0f}%**, "
                "signaling potential market saturation or investor pullback."
            )
    else:
        st.info("Need at least 2 years of data to compute growth trends. Adjust the year filter.")

    # ── 6B. REGIONAL FUNDING DISPARITIES ────────────────────────
    st.markdown("---")
    st.markdown("### 🌍 Regional Funding Disparities")

    city_share = filtered_df.groupby('CityLocation')['AmountInUSD'].sum().sort_values(ascending=False)
    total_funding = city_share.sum()

    if total_funding > 0:
        top3 = city_share.head(3)
        top3_share = top3.sum() / total_funding * 100
        rest_share = 100 - top3_share

        col_k, col_l = st.columns(2)

        with col_k:
            fig6c, ax6c = plt.subplots(figsize=(7, 5))
            labels = list(top3.index) + ['All Other Cities']
            sizes = list(top3.values) + [city_share.iloc[3:].sum()]
            explode = [0.05] * len(top3) + [0.0]
            ax6c.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
                     explode=explode, colors=sns.color_palette('coolwarm', len(labels)))
            ax6c.set_title("Funding Concentration — Top 3 Cities vs Rest", fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig6c)
            plt.close(fig6c)

        with col_l:
            # Deals per city (top 10) vs funding per deal
            city_eff = filtered_df.groupby('CityLocation').agg(
                Deals=('AmountInUSD', 'count'),
                AvgDeal=('AmountInUSD', 'mean'),
            ).nlargest(10, 'Deals').reset_index()
            fig6d, ax6d = plt.subplots(figsize=(7, 5))
            scatter = ax6d.scatter(
                city_eff['Deals'], city_eff['AvgDeal'] / 1e6,
                s=city_eff['Deals'] * 3, alpha=0.7, c='#6366f1', edgecolors='white',
            )
            for _, row in city_eff.iterrows():
                ax6d.annotate(row['CityLocation'], (row['Deals'], row['AvgDeal'] / 1e6),
                              fontsize=8, ha='center', va='bottom')
            ax6d.set_title("City Efficiency: Deal Volume vs Avg Deal Size", fontweight='bold')
            ax6d.set_xlabel("Number of Deals")
            ax6d.set_ylabel("Avg Deal Size ($M)")
            plt.tight_layout()
            st.pyplot(fig6d)
            plt.close(fig6d)

        st.error(
            f"**Regional Disparity:** The top 3 cities ({', '.join(top3.index)}) "
            f"capture **{top3_share:.1f}%** of all funding, "
            f"leaving just **{rest_share:.1f}%** for {city_share.shape[0] - 3} other cities. "
            "This underscores a significant geographic concentration of VC capital."
        )

    # ── 6C. SECTOR HEATMAP BY YEAR ─────────────────────────────
    st.markdown("---")
    st.markdown("### 🗺️ Sector × Year Heatmap")

    top10_ind = filtered_df.groupby('IndustryVertical')['AmountInUSD'].sum().nlargest(10).index
    heatmap_data = filtered_df[filtered_df['IndustryVertical'].isin(top10_ind)].pivot_table(
        index='IndustryVertical', columns='Year', values='AmountInUSD',
        aggfunc='sum', fill_value=0,
    )

    if not heatmap_data.empty:
        fig6e, ax6e = plt.subplots(figsize=(12, 5))
        sns.heatmap(heatmap_data / 1e6, annot=True, fmt='.0f', cmap='YlOrRd',
                    linewidths=0.5, ax=ax6e, cbar_kws={'label': 'Funding ($M)'})
        ax6e.set_title("Funding Heatmap — Top 10 Industries by Year ($M)", fontweight='bold')
        ax6e.set_ylabel("")
        ax6e.set_xlabel("")
        plt.tight_layout()
        st.pyplot(fig6e)
        plt.close(fig6e)

# ═══════════════════════════════════════════════════════════════
# 6. DETAILED DATA VIEW
# ═══════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📋 Filtered Data View")
st.caption(f"Showing top 50 of {len(filtered_df):,} filtered records.")
display_cols = [c for c in ['StartupName', 'IndustryVertical', 'SubVertical',
                            'CityLocation', 'InvestmentType', 'AmountInUSD', 'Year']
                if c in filtered_df.columns]
table_df = filtered_df[display_cols].sort_values(by='AmountInUSD', ascending=False).head(50).copy()
table_df['AmountInUSD'] = (table_df['AmountInUSD'] / 1e6).apply(lambda x: f"${x:,.2f}M")
table_df = table_df.rename(columns={'AmountInUSD': 'Funding ($M)'})
st.dataframe(
    table_df,
    use_container_width=True, hide_index=True,
)