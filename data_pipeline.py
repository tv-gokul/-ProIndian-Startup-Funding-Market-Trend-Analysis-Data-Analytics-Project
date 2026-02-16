import pandas as pd
import numpy as np
import sqlite3

def build_pipeline(csv_path):
    """
    Full ETL pipeline: Load raw CSV → Clean → Transform → Export to SQLite.
    Handles missing values, inconsistent categories, and incorrect funding entries.
    """
    print("=" * 60)
    print("PROINDIAN DATA PIPELINE — Starting ETL Process")
    print("=" * 60)

    # ── STEP 1: LOAD RAW DATA ──────────────────────────────────
    print("\n[1/7] Loading raw CSV data...")
    df = pd.read_csv(csv_path)
    print(f"       Loaded {len(df):,} records with {len(df.columns)} columns.")

    # ── STEP 2: NORMALIZE COLUMN HEADERS ───────────────────────
    print("[2/7] Normalizing column headers...")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        'Startup Name': 'StartupName',
        'Location': 'CityLocation',
        'Industry': 'IndustryVertical',
        'Sub-vertical': 'SubVertical',
        'Investment Type': 'InvestmentType',
        'Amount in USD': 'AmountInUSD',
        'Website URL': 'WebsiteURL',
    })

    # ── STEP 3: HANDLE MISSING VALUES ──────────────────────────
    print("[3/7] Handling missing values...")
    missing_before = df.isnull().sum().sum()
    df.dropna(subset=['CityLocation', 'IndustryVertical'], inplace=True)
    df['AmountInUSD'] = df['AmountInUSD'].fillna('0')
    df['SubVertical'] = df['SubVertical'].fillna('Unknown')
    df['InvestmentType'] = df['InvestmentType'].fillna('Undisclosed')
    df['Investors'] = df['Investors'].fillna('Undisclosed')
    df['StartupName'] = df['StartupName'].fillna('Unknown')
    missing_after = df.isnull().sum().sum()
    print(f"       Resolved {missing_before - missing_after:,} missing values.")

    # ── STEP 4: FIX INCORRECT FUNDING ENTRIES ──────────────────
    print("[4/7] Fixing incorrect funding entries...")
    df['AmountInUSD'] = df['AmountInUSD'].astype(str).str.replace(',', '', regex=False)
    df['AmountInUSD'] = df['AmountInUSD'].str.replace(r'(?i)undisclosed|unknown|n/a', '0', regex=True)
    df['AmountInUSD'] = pd.to_numeric(df['AmountInUSD'], errors='coerce').fillna(0)
    invalid_count = (df['AmountInUSD'] == 0).sum()
    print(f"       {invalid_count:,} entries had zero or unparseable funding values.")

    # ── STEP 5: STANDARDIZE CATEGORIES ─────────────────────────
    print("[5/7] Standardizing categories (cities, industries, investment types)...")

    # ── Cities ──
    df['CityLocation'] = df['CityLocation'].astype(str).str.strip().str.title()
    # Extract only the first city when "City1 / City2" or "City1, State" patterns appear
    df['CityLocation'] = (df['CityLocation']
        .str.replace(r'\s*/\s*.*', '', regex=True)     # "Pune / Dubai" → "Pune"
        .str.replace(r',\s*.*', '', regex=True)        # "Jaipur, Rajasthan" → "Jaipur"
        .str.strip()
    )
    city_map = {
        # Spelling variants
        'Bengaluru': 'Bangalore',
        'Bengaluru And Gurugram': 'Bangalore',
        'Ahemadabad': 'Ahmedabad', 'Ahemdabad': 'Ahmedabad',
        'Bhubneswar': 'Bhubaneswar',
        'Kolkatta': 'Kolkata',
        'Nw Delhi': 'Delhi',
        # Regions → canonical city
        'New Delhi': 'Delhi', 'Ncr': 'Delhi NCR', 'Noida': 'Delhi NCR',
        'Gurugram': 'Gurgaon',
        'Navi Mumbai': 'Mumbai',
        'Andheri': 'Mumbai', 'Chembur': 'Mumbai',
        'Kormangala': 'Bangalore', 'Taramani': 'Chennai',
        # Foreign / multi-city → primary Indian city or "International"
        'Sfo': 'International', 'San Francisco': 'International',
        'Palo Alto': 'International', 'New York': 'International',
        'London': 'International', 'California': 'International',
        'Menlo Park': 'International', 'Santa Monica': 'International',
        'Seattle': 'International', 'Nairobi': 'International',
        'Burnsville': 'International', 'Washington': 'International',
        'San Jose': 'International', 'Newark': 'International',
        'Wilmington': 'International', 'Stanford': 'International',
        'Us': 'International', 'Usa': 'International',
        # State names → keep but standardise
        'Haryana': 'Gurgaon', 'Karnataka': 'Bangalore',
        'Kerala': 'Kochi', 'Uttar Pradesh': 'Delhi NCR',
        'India': 'Other India', 'Missourie': 'Other India',
        'Tulangan': 'Other India', 'Panaji': 'Goa',
    }
    df['CityLocation'] = df['CityLocation'].replace(city_map)

    # ── Industries ──
    df['IndustryVertical'] = df['IndustryVertical'].astype(str).str.strip().str.title()
    industry_map = {
        # E-Commerce variants
        'Ecommerce': 'E-Commerce', 'E Commerce': 'E-Commerce',
        'Ecommece': 'E-Commerce', 'B2B E-Commerce': 'E-Commerce',
        'Saas, Ecommerce': 'E-Commerce',
        # Fintech
        'Fin-Tech': 'Fintech', 'Fin Tech': 'Fintech',
        'Financial Tech': 'Fintech', 'Rural Fintech': 'Fintech',
        'Finance': 'Fintech', 'Financial Services': 'Fintech',
        'Fiinance': 'Fintech', 'Bfsi': 'Fintech', 'Nbfc': 'Fintech',
        'Digital Reconcilation And Financial Services': 'Fintech',
        # Edtech
        'Ed-Tech': 'Edtech', 'Ed Tech': 'Edtech',
        'Edu Tech': 'Edtech', 'Edu-Tech': 'Edtech', 'Edutech': 'Edtech',
        'Education': 'Edtech', 'Online Education': 'Edtech',
        'E-Tech': 'Edtech',
        # Healthcare
        'Health Care': 'Healthcare', 'Health Tech': 'Healthcare',
        'Health Tech Startup': 'Healthcare', 'Healthtech': 'Healthcare',
        'Health And Wellness': 'Healthcare',
        # Food & Beverage
        'Food & Beverages': 'Food & Beverage', 'Food And Beverage': 'Food & Beverage',
        'Food And Beverages': 'Food & Beverage', 'Food': 'Food & Beverage',
        'Food Startup': 'Food & Beverage', 'Food Production': 'Food & Beverage',
        'Food Tech': 'Foodtech', 'Food-Tech': 'Foodtech',
        'Food Delivery': 'Foodtech', 'Online Food Delivery': 'Foodtech',
        'B2B-Focused Foodtech Startup': 'Foodtech',
        # Transport / Logistics
        'Transport': 'Transportation', 'Last Mile Transportation': 'Transportation',
        'Logistics Solution Provider': 'Logistics', 'Logistics Tech': 'Logistics',
        'Hyper-Local Logistics': 'Logistics',
        # Technology / AI
        'Consumer Internet': 'Consumer Internet',
        'Consumer Interne': 'Consumer Internet', 'Consumer Portal': 'Consumer Internet',
        'Consumer Technology': 'Technology', 'Information Technology': 'Technology',
        'Information Technology And Services': 'Technology', 'It': 'Technology',
        'Tech': 'Technology', 'Technology Provider': 'Technology',
        'Software': 'Technology', 'Digital Solutions': 'Technology',
        'Ai': 'Artificial Intelligence', 'Deep Tech Ai': 'Artificial Intelligence',
        'Deep-Tech': 'Technology',
        # Auto
        'Auto': 'Automobile', 'Automotive': 'Automobile',
        'Automotive Rental': 'Automobile',
        # Media / Entertainment
        'Digital Media': 'Media & Entertainment', 'Media': 'Media & Entertainment',
        'Entertainment': 'Media & Entertainment', 'Video': 'Media & Entertainment',
        'Ott Player': 'Media & Entertainment', 'Video Games': 'Gaming',
        'Gaming And Entertainment': 'Gaming', 'Online Gaming': 'Gaming',
        # Consumer goods
        'Consumer Goods Company': 'Consumer Goods',
        # Advertising
        'Advertising, Marketing': 'Advertising',
        'B2B Marketing': 'Advertising', 'Intelligent Marketing Cloud': 'Advertising',
        # Agritech
        'Agriculture': 'Agritech', 'Agritech Startup': 'Agritech', 'Agtech': 'Agritech',
        # SaaS
        'Saas': 'SaaS',
        # Misc cleanups
        'Block Chain, Cryptocurrency': 'Blockchain',
        'Storytelling': 'Publishing',
        'Clean-Tech': 'Clean Energy', 'Energy': 'Clean Energy',
        'Customer Service': 'Customer Service', 'Customer Service Platform': 'Customer Service',
        'Social Network': 'Social Media',
        'Video Customer Experience(Cx) Platform': 'Customer Service',
        'Co-Working Spaces': 'Real Estate',
        'Lifestyle': 'Consumer Goods', 'Fashion': 'Consumer Goods',
        'Fashion And Apparel': 'Consumer Goods', 'Personal Care': 'Consumer Goods',
        'Luxury Label': 'Consumer Goods',
    }
    df['IndustryVertical'] = df['IndustryVertical'].replace(industry_map)

    # ── Investment types ──
    df['InvestmentType'] = df['InvestmentType'].astype(str).str.strip().str.title()
    invest_map = {
        # Seed / Angel consolidation
        'Seed/ Angel Funding': 'Seed/Angel', 'Seed / Angel Funding': 'Seed/Angel',
        'Seed Funding': 'Seed/Angel', 'Angel': 'Seed/Angel',
        'Angel Funding': 'Seed/Angel', 'Angel Round': 'Seed/Angel',
        'Angel / Seed Funding': 'Seed/Angel', 'Seed / Angle Funding': 'Seed/Angel',
        'Seed/Angel Funding': 'Seed/Angel', 'Seed': 'Seed/Angel',
        'Seed Round': 'Seed/Angel', 'Seed Funding Round': 'Seed/Angel',
        'Pre Seed': 'Pre-Seed',
        # Private Equity
        'Private Equity Round': 'Private Equity', 'Privateequity': 'Private Equity',
        'Private Funding': 'Private Equity', 'Private': 'Private Equity',
        'Equity': 'Private Equity', 'Equity Based Funding': 'Private Equity',
        'Equity And Debt': 'Private Equity',
        # Debt consolidation
        'Debt Financing': 'Debt', 'Debt-Funding': 'Debt', 'Debt Funding': 'Debt',
        'Structured Debt': 'Debt', 'Term Loan': 'Debt',
        'Debt And Preference Capital': 'Debt', 'Mezzanine': 'Debt',
        # Venture consolidation
        'Venture': 'Venture Round', 'Venture Series': 'Venture Round',
        'Venture - Series Unknown': 'Venture Round',
        'Venture-Series Unknown': 'Venture Round',
        # Pre-Series
        'Pre Series A': 'Pre-Series A',
        # Bridge
        'Bridge Funding': 'Bridge Round',
        # Misc → Undisclosed
        'Funding': 'Undisclosed', 'Funding Round': 'Undisclosed',
        'In Progress': 'Undisclosed', 'Unspecified': 'Undisclosed',
        'Maiden Round': 'Undisclosed', 'Follow-On': 'Undisclosed',
        'Single Venture': 'Undisclosed', 'Inhouse Funding': 'Undisclosed',
        'Personal Investment': 'Undisclosed',
        'Series 1': 'Series A', 'Series': 'Undisclosed',
    }
    df['InvestmentType'] = df['InvestmentType'].replace(invest_map)

    # ── STEP 6: DATE & YEAR EXTRACTION ─────────────────────────
    print("[6/7] Extracting year from dates...")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='mixed', dayfirst=True)
    df['Year'] = df['Date'].dt.year.fillna(0).astype(int)
    df['Month'] = df['Date'].dt.month.fillna(0).astype(int)

    # ── STEP 7: EXPORT TO SQLITE ───────────────────────────────
    print("[7/7] Exporting cleaned data to SQLite...")
    conn = sqlite3.connect('data/proindian_funding.db')
    df.to_sql('startup_funding', conn, if_exists='replace', index=False)
    conn.close()

    # ── SUMMARY ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE — Summary")
    print("=" * 60)
    print(f"  Records exported   : {len(df):,}")
    print(f"  Year range         : {df['Year'][df['Year']>0].min()} – {df['Year'].max()}")
    print(f"  Unique industries  : {df['IndustryVertical'].nunique()}")
    print(f"  Unique cities      : {df['CityLocation'].nunique()}")
    print(f"  Investment types   : {df['InvestmentType'].nunique()}")
    print(f"  Total funding      : ${df['AmountInUSD'].sum()/1e9:.2f} Billion USD")
    print("=" * 60)

if __name__ == "__main__":
    build_pipeline("data/indian_startup_funding.csv")