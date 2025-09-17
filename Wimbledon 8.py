import pandas as pd
import numpy as np
import glob
import os
from collections import defaultdict
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

#hyperparamter tuning 
def tune_surface_model(X_train, y_train):
    param_grid = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [400, 600, 800],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2]
    }
    xgb = XGBClassifier(tree_method="hist", random_state=42, eval_metric="logloss")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(xgb, param_grid, n_iter=20, cv=cv, scoring="accuracy", n_jobs=-1, verbose=2)
    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)
    print("Best CV score:", search.best_score_)
    return search.best_estimator_


def normalize_rounds(df):
    df = df.copy()
    round_map = {
        # Main draw
        'Final': 'The Final',
        'F': 'The Final',
        'Semifinal': 'Semifinals',
        'Semifinals': 'Semifinals',
        'SF': 'Semifinals',
        'Quarterfinal': 'Quarterfinals',
        'Quarterfinals': 'Quarterfinals',
        'QF': 'Quarterfinals',
        'Round of 16': 'Round of 16',
        'R16': 'Round of 16',
        'Round of 32': 'Round of 32',
        'R32': 'Round of 32',
        'Round of 64': 'Round of 64',
        'R64': 'Round of 64',
        'Round of 128': 'Round of 128',
        'R128': 'Round of 128',

        
        '1st Round Qualifying': '1st Round',
        '2nd Round Qualifying': '2nd Round',
        '3rd Round Qualifying': '3rd Round',
        '4th Round Qualifying': '4th Round',  
    }

    df['Round'] = df['Round'].astype(str).replace(round_map)

    round_order = [
        
        '1st Round','2nd Round','3rd Round','4th Round',
        'Round of 128','Round of 64','Round of 32','Round of 16',
        'Quarterfinals','Semifinals','The Final','Unknown'
    ]
    df['Round'] = pd.Categorical(df['Round'], categories=round_order, ordered=True)
    return df

def get_wimbledon_2025_masks(df):
    wim_mask = df['Tournament'].str.contains("Wimbledon", case=False, na=False)
    wim_2025_mask = wim_mask & (df['Date'].dt.year == 2025)
    if not wim_2025_mask.any():
        raise ValueError("No Wimbledon 2025 matches found.")
    wim_start_date = df.loc[wim_2025_mask, 'Date'].min()
    return wim_mask, wim_2025_mask, wim_start_date

def dedupe_matches_for_display(df):
        df = df.copy()
        p1_first = df['Player_1'].astype(str) <= df['Player_2'].astype(str)
    
        canon_p1 = np.where(p1_first, df['Player_1'], df['Player_2'])
        canon_p2 = np.where(p1_first, df['Player_2'], df['Player_1'])
        
        df['__canon_key__'] = list(zip(
            df['Tournament'].astype(str),
            df['Round'].astype(str),
            df['Date'].dt.year,
            canon_p1, canon_p2
        ))
        
        df = df.sort_values(['Date', 'Round']).drop_duplicates('__canon_key__', keep='first')
        return df.drop(columns='__canon_key__')


#Step 1: Load + Mirror Sackmann Files

sackmann_folder = r"C:/Users/brade/Desktop/Phyton Projects/"

def load_sackmann_files(folder=sackmann_folder):
    files = glob.glob(os.path.join(folder, "atp_matches_*.csv"))
    print(f" Found {len(files)} Sackmann files")
    if not files:
        raise FileNotFoundError("No Sackmann files found")

    df_list = []
    for file in sorted(files):
        df_year = pd.read_csv(file, low_memory=False)
        if 'tourney_date' in df_year.columns:
            df_year['Date'] = pd.to_datetime(df_year['tourney_date'].astype(str),
                                             format='%Y%m%d', errors='coerce')
        else:
            df_year['Date'] = pd.to_datetime(df_year['Date'], errors='coerce')

        standardized = pd.DataFrame({
            'Tournament': df_year.get('tourney_name', np.nan),
            'Date': df_year['Date'],
            'Series': df_year.get('tourney_level', np.nan),
            'Court': df_year.get('court', 'Unknown'),
            'Surface': df_year.get('surface', np.nan),
            'Round': df_year.get('round', np.nan),
            'Best of': df_year.get('best_of', 3),
            'Player_1': df_year.get('winner_name', np.nan),
            'Player_2': df_year.get('loser_name', np.nan),
            'Winner': df_year.get('winner_name', np.nan),
            'Loser': df_year.get('loser_name', np.nan),
            'Rank_1': df_year.get('winner_rank', np.nan),
            'Rank_2': df_year.get('loser_rank', np.nan),
            'Pts_1': df_year.get('winner_rank_points', -1),
            'Pts_2': df_year.get('loser_rank_points', -1),
            'Odd_1': -1.0, 'Odd_2': -1.0,
            'Score': df_year.get('score', np.nan),
            # serve stats
            'Ace_1': df_year.get('w_ace', np.nan),
            'Ace_2': df_year.get('l_ace', np.nan),
            'DF_1': df_year.get('w_df', np.nan),
            'DF_2': df_year.get('l_df', np.nan),
            'BP_1': df_year.get('w_bpSaved', np.nan) / (df_year.get('w_bpFaced', 1)+1e-6),
            'BP_2': df_year.get('l_bpSaved', np.nan) / (df_year.get('l_bpFaced', 1)+1e-6)
        })
        standardized['Source'] = 'Sackmann'

        # Mirror matches
        mirror = standardized.copy()
        mirror['Player_1'], mirror['Player_2'] = standardized['Player_2'], standardized['Player_1']
        mirror['Rank_1'], mirror['Rank_2'] = standardized['Rank_2'], standardized['Rank_1']
        mirror['Pts_1'], mirror['Pts_2'] = standardized['Pts_2'], standardized['Pts_1']
        mirror['Ace_1'], mirror['Ace_2'] = standardized['Ace_2'], standardized['Ace_1']
        mirror['DF_1'], mirror['DF_2'] = standardized['DF_2'], standardized['DF_1']
        mirror['BP_1'], mirror['BP_2'] = standardized['BP_2'], standardized['BP_1']
        df_list.append(pd.concat([standardized, mirror], ignore_index=True))

    combined = pd.concat(df_list, ignore_index=True)
    print(f"\\n Sackmann loaded: {len(combined)} rows (mirror included)")
    return combined


# Step 2: Load 2025 Excel Competitions + Mirror

def load_excel_data(file_path="ATP_2025_Competitions.xlsx"):
    df = pd.read_excel(file_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Source"] = "Excel"

    # Map columns to Sackmann-style schema
    mapped = pd.DataFrame({
        "Tournament": df.get("Tournament"),
        "Date": df["Date"],
        "Surface": df.get("Surface"),
        "Round": df.get("Round"),
        "Best of": df.get("BestOf", 3),
        "Player_1": df.get("Player1 Name"),
        "Player_2": df.get("Player2 Name"),
        "Winner": df["Winner"],
        "Loser": df["Loser"],
        "Rank_1": df.get("P1 Rank"),
        "Rank_2": df.get("P2 Rank"),
        "Ace_1": df.get("Aces P1"),
        "Ace_2": df.get("Aces P2"),
        "DF_1": df.get("Double Faults P1"),
        "DF_2": df.get("Double Faults P2"),
        "ServeRating_P1": df.get("Serve Rating P1"),
        "ServeRating_P2": df.get("Serve Rating P2"),
        "ReturnRating_P1": df.get("Return Rating P1"),
        "ReturnRating_P2": df.get("Return Rating P2"),
        "P1 Age": df.get("P1 Age"),
        "P2 Age": df.get("P2 Age"),
        "P1 Height": df.get("P1 Height"),
        "P2 Height": df.get("P2 Height"),
        "P1 Weight": df.get("P1 Weight"),
        "P2 Weight": df.get("P2 Weight"),
        "P1 Plays": df.get("P1 Plays"),
        "P2 Plays": df.get("P2 Plays"),
        "P1 Backhand": df.get("P1 Backhand"),
        "P2 Backhand": df.get("P2 Backhand"),
        "P1 Career W/L": df.get("P1 Career W/L"),
        "P2 Career W/L": df.get("P2 Career W/L"),
        "P1 YTD W/L": df.get("P1 YTD W/L"),
        "P2 YTD W/L": df.get("P2 YTD W/L"),
        "Source": "Excel",
    })

    # Mirror to Player_2 perspective
    mirror = mapped.copy()
    mirror[["Player_1","Player_2"]] = mapped[["Player_2","Player_1"]].values
    mirror[["Rank_1","Rank_2"]] = mapped[["Rank_2","Rank_1"]].values
    mirror[["Ace_1","Ace_2"]] = mapped[["Ace_2","Ace_1"]].values
    mirror[["DF_1","DF_2"]] = mapped[["DF_2","DF_1"]].values
    mirror[["ServeRating_P1","ServeRating_P2"]] = mapped[["ServeRating_P2","ServeRating_P1"]].values
    mirror[["ReturnRating_P1","ReturnRating_P2"]] = mapped[["ReturnRating_P2","ReturnRating_P1"]].values
    mirror[["P1 Age","P2 Age"]] = mapped[["P2 Age","P1 Age"]].values
    mirror[["P1 Height","P2 Height"]] = mapped[["P2 Height","P1 Height"]].values
    mirror[["P1 Weight","P2 Weight"]] = mapped[["P2 Weight","P1 Weight"]].values
    mirror[["P1 Plays","P2 Plays"]] = mapped[["P2 Plays","P1 Plays"]].values
    mirror[["P1 Backhand","P2 Backhand"]] = mapped[["P2 Backhand","P1 Backhand"]].values
    mirror[["P1 Career W/L","P2 Career W/L"]] = mapped[["P2 Career W/L","P1 Career W/L"]].values
    mirror[["P1 YTD W/L","P2 YTD W/L"]] = mapped[["P2 YTD W/L","P1 YTD W/L"]].values

    df_all = pd.concat([mapped, mirror], ignore_index=True)
    df2025 = df_all[df_all["Date"].dt.year == 2025]
    print(f" Excel 2025 matches (with mirror): {len(df2025)} rows")
    return df2025


#Step 3: Merge

def combine_sources(sackmann_df, excel_df):
    all_cols = set(sackmann_df.columns) | set(excel_df.columns)
    for col in all_cols:
        if col not in sackmann_df: 
            sackmann_df[col] = np.nan
        if col not in excel_df: 
            excel_df[col] = np.nan
    merged = pd.concat([sackmann_df, excel_df], ignore_index=True).sort_values("Date")
    print(f" Combined: {len(merged)} rows")
    return merged


#Step 4: Elo Ratings

def calculate_atp_elo(df, start_rating=1500, layoff_threshold=100, tau=4.0):
    elo = defaultdict(lambda: start_rating)
    matches_played = defaultdict(int)
    last_played = {}
    pre_ratings_p1, pre_ratings_p2 = [], []
    latest_date = pd.to_datetime(df['Date']).max()

    def tourney_weight(t, s):
        t, s = str(t).lower(), str(s).lower()
        if any(x in t for x in ['wimbledon','us open','roland garros','australian']):
            return 2.0
        elif "m" in s or "masters" in s:
            return 1.5
        elif "500" in s:
            return 1.25
        return 1.0

    for _, row in df.iterrows():
        p1, p2, date = row['Player_1'], row['Player_2'], pd.to_datetime(row['Date'])
        r1, r2 = elo[p1], elo[p2]
        pre_ratings_p1.append(r1)
        pre_ratings_p2.append(r2)
        k1 = 800/(matches_played[p1]+1)
        k2 = 800/(matches_played[p2]+1)

        def layoff_bonus(p):
            if p not in last_played: 
                return 1.0
            gap = (date - last_played[p]).days
            return 1.25 if gap > layoff_threshold else 1.0
        
        k1 *= layoff_bonus(p1)
        k2 *= layoff_bonus(p2)

        age_days = (latest_date-date).days
        recency = np.exp(-age_days/(365*tau))
        weight = tourney_weight(row['Tournament'], row['Series'])

        e1 = 1/(1+10**((r2-r1)/400))
        s1 = 1 if row['Winner']==p1 else 0
        s2 = 1-s1
        elo[p1] = r1+(k1*recency*weight)*(s1-e1)
        elo[p2] = r2+(k2*recency*weight)*(s2-(1-e1))
        matches_played[p1] += 1
        matches_played[p2] += 1
        last_played[p1] = date
        last_played[p2] = date
    
    df['Elo_1'] = pre_ratings_p1
    df['Elo_2'] = pre_ratings_p2
    df['EloDiff'] = df['Elo_1'] - df['Elo_2']
    return df


# STEP 5: Streamlined Feature Engineering

def engineer_features(df):
    df['Target'] = (df['Winner'] == df['Player_1']).astype(int)
    df['RankDiff'] = df['Rank_1'] - df['Rank_2']
    df['RankDiff_Log'] = np.sign(df['RankDiff']) * np.log1p(df['RankDiff'].abs())

    
    surface_dummies = pd.get_dummies(df['Surface'], prefix="Surface")
    df = df.sort_values("Date")

    
    for stat in ["Ace_1", "DF_1", "BP_1"]:
        df[f"{stat}_form"] = (
            df.groupby("Player_1")[stat]
            .rolling(5, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

    
    elo_momentum = []
    last_elo = defaultdict(lambda: (None, None))

    for _, row in df.iterrows():
        p1, elo1, date = row['Player_1'], row['Elo_1'], row['Date']
        last_date, last_val = last_elo[p1]

        if last_date is None:
            elo_momentum.append(0.0)
        else:
            days_diff = (date - last_date).days
            if days_diff <= 90:
                elo_momentum.append(elo1 - last_val)
            else:
                elo_momentum.append(0.0)

        last_elo[p1] = (date, elo1)

    df['EloMomentum'] = elo_momentum
    return pd.concat([df, surface_dummies], axis=1)

#Step 6: Updated Training Function

def train_model(df, features, train_mask, test_mask):
    X = df[features].fillna(0)
    y = df['Target'].astype(int)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    print(f"Training size: {X_train.shape}, Test size: {X_test.shape}")
    print("Target distribution:", y_train.value_counts().to_dict())

    
    params = {
        'max_depth': [4, 5],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [400, 600],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1, 1.5]
    }
    
    xgb = XGBClassifier(tree_method="hist", random_state=42, eval_metric="logloss")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    
    search = RandomizedSearchCV(xgb, params, n_iter=8, cv=cv, scoring="accuracy", n_jobs=1, verbose=1)
    search.fit(X_train, y_train)
    
    preds = search.best_estimator_.predict(X_test)
    test_acc = accuracy_score(y_test, preds)
    
    print(" Best params:", search.best_params_)
    print(" Best CV score:", round(search.best_score_, 3))
    print(" Test acc:", round(test_acc, 3))
    print(classification_report(y_test, preds, digits=3))

    
    importances = search.best_estimator_.feature_importances_
    feat_imp = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    print("\\n Top Feature Importances:")
    print(feat_imp.to_string(index=False))

    
    plt.figure(figsize=(10,6))
    plt.barh(feat_imp['Feature'], feat_imp['Importance'], color="skyblue")
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title("Feature Importances (Streamlined XGBoost)")
    plt.tight_layout()
    
    out_path = "feature_importances_streamlined.png"
    plt.savefig(out_path)
    print(f" Feature importance plot saved as {out_path}")
    
    
    plt.show(block=False)

    return search.best_estimator_, test_acc, preds, y_test


#Step 7: Quick Correlation Check

def plot_feature_correlations(df):
    feat_cols = ['RankDiff_Log','EloDiff','EloMomentum',
                 'Ace_1_form','DF_1_form','BP_1_form',
                 'Surface_Clay','Surface_Grass','Surface_Hard']
    
    corr = df[feat_cols + ['Target']].corr()
    
    print("\\n Correlation with Target (Player 1 win = 1):")
    target_corr = corr['Target'].sort_values(ascending=False)
    print(target_corr)

    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Streamlined Feature Correlation Heatmap")
    plt.tight_layout()

    out_path = "feature_correlations_streamlined.png"
    plt.savefig(out_path)
    print(f" Correlation heatmap saved as {out_path}")

    
    plt.show(block=False)
    


# FILTER FUNCTIONS

def filter_wimbledon(df):
    wim_df = df[df['Tournament'].str.contains("Wimbledon", case=False, na=False)].copy()
    print(f"\\n Wimbledon subset: {len(wim_df)} rows")
    return wim_df

def filter_slam(df, slam_name):
    slam_df = df[df['Tournament'].str.contains(slam_name, case=False, na=False)].copy()
    print(f"\\n {slam_name} subset: {len(slam_df)} rows")
    return slam_df

def slam_sandbox(df, slam_name):
    slam_df = filter_slam(df.copy(), slam_name)
    if len(slam_df) > 0:
        slam_df = calculate_atp_elo(slam_df)
        slam_df = engineer_features(slam_df)
        features_slam = ['RankDiff_Log','EloDiff','EloMomentum']
        
        X = slam_df[features_slam].fillna(0)
        y = slam_df['Target']
        
        train_mask = slam_df['Date'] < "2025-01-01"
        test_mask = slam_df['Date'] >= "2025-01-01"
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        if X_test.shape[0] > 0:
            print(f"Training size: {X_train.shape}, Test size: {X_test.shape}")
            xgb = XGBClassifier(
                max_depth=4, learning_rate=0.05, n_estimators=400,
                subsample=0.9, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1,
                tree_method="hist", random_state=42, eval_metric="logloss"
            )
            xgb.fit(X_train, y_train)
            preds = xgb.predict(X_test)
            acc = accuracy_score(y_test, preds)
            print(f"🏆 {slam_name} 2025 Test Accuracy: {acc:.3f}")
            print(classification_report(y_test, preds, digits=3))
        else:
            print(f"️ No {slam_name} 2025 matches found.")
    else:
        print(f"️ {slam_name} subset empty.")


# SURFACE-SPECIFIC ELO CALCULATION

def calculate_surface_elo(df, start_rating=1500):
    surfaces = ["Grass","Clay","Hard"]
    elos = {s: defaultdict(lambda: start_rating) for s in surfaces}
    pre_elo1, pre_elo2 = [], []
    
    for _, row in df.iterrows():
        surf = str(row['Surface'])
        if surf not in surfaces:
            surf = "Hard"  
        
        p1, p2 = row['Player_1'], row['Player_2']
        r1, r2 = elos[surf][p1], elos[surf][p2]
        pre_elo1.append(r1)
        pre_elo2.append(r2)
        
        
        e1 = 1/(1+10**((r2-r1)/400))
        s1 = 1 if row['Winner']==p1 else 0
        s2 = 1-s1
        k = 32
        elos[surf][p1] = r1+k*(s1-e1)
        elos[surf][p2] = r2+k*(s2-(1-e1))
    
    df['SurfElo_1'] = pre_elo1
    df['SurfElo_2'] = pre_elo2
    df['SurfEloDiff'] = df['SurfElo_1'] - df['SurfElo_2']
    return df


# MAIN EXECUTION

print(" Loading data...")
sack = load_sackmann_files()
excel = load_excel_data()
df = combine_sources(sack, excel)

df = normalize_rounds(df)

print("\\n Calculating Elo ratings...")
df = calculate_atp_elo(df)

print("\\n Calculating Surface Elo ratings...")
df = calculate_surface_elo(df)

df['CombinedEloDiff'] = 0.7 * df['EloDiff'] + 0.3 * df['SurfEloDiff']

print("\\n Engineering features...")
df = engineer_features(df)


wim_mask, wim_2025_mask, wim_start_date = get_wimbledon_2025_masks(df)
print(f"Wimbledon 2025 start date detected: {wim_start_date.date()}")


train_mask_global = df['Date'] < wim_start_date
test_mask_global  = wim_2025_mask  

print("\\nTraining streamlined XGBoost model (train < Wimbledon 2025; test = Wimbledon 2025)...")
features_general = ['RankDiff_Log', 'EloDiff', 'EloMomentum']
best_model, final_accuracy, preds_general_test, y_general_test = train_model(
    df, features_general, train_mask_global, test_mask_global
)

print("\\n Analyzing feature correlations...")
plot_feature_correlations(df)

print(f"\\n Final Test Accuracy: {final_accuracy:.3f}")
print(" Streamlined pipeline complete!")


# WIMBLEDON-ONLY SANDBOX

print("\\n Testing Wimbledon-only subset...")
df_wim = filter_wimbledon(df.copy())

if len(df_wim) > 0:
    
    df_wim = calculate_atp_elo(df_wim)
    df_wim = engineer_features(df_wim)
    
    
    features_wim = ['RankDiff_Log', 'EloDiff', 'EloMomentum']
    
    X = df_wim[features_wim].fillna(0)
    y = df_wim['Target']
    
    train_mask = df_wim['Date'] < wim_start_date
    test_mask = df_wim['Date'] >= wim_start_date
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    if X_test.shape[0] > 0:
        print(f"Training size: {X_train.shape}, Test size: {X_test.shape}")
        xgb = XGBClassifier(
            max_depth=4,
            learning_rate=0.05,
            n_estimators=400,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1,
            tree_method="hist",
            random_state=42,
            eval_metric="logloss"
        )
        xgb.fit(X_train, y_train)
        preds = xgb.predict(X_test)
        acc_wim = accuracy_score(y_test, preds)
        
        print(f"\\n Wimbledon 2025 Test Accuracy: {acc_wim:.3f}")
        print(classification_report(y_test, preds, digits=3))
    else:
        print("️ No Wimbledon 2025 matches found.")
else:
    print("️ Wimbledon subset is empty.")


# GRAND SLAM SANDBOXES

for slam in ["Wimbledon", "US Open", "Roland Garros", "Australian Open"]:
    slam_sandbox(df.copy(), slam)


# SURFACE-SPECIFIC ELO SANDBOX

print("\nTesting Surface-Specific Elo with pre-Wimbledon training and Wimbledon-only test...")

# Recompute SurfElo on full df (history) once, then engineer features
df_surf = calculate_surface_elo(df.copy())
df_surf = engineer_features(df_surf)


df_surf = df_surf.loc[:, ~df_surf.columns.duplicated()]


if 'CombinedEloDiff' not in df_surf.columns:
    if 'EloDiff' in df_surf.columns and 'SurfEloDiff' in df_surf.columns:
        df_surf['CombinedEloDiff'] = 0.7 * df_surf['EloDiff'] + 0.3 * df_surf['SurfEloDiff']
    else:
        
        df_surf['CombinedEloDiff'] = df_surf.get('EloDiff', 0).fillna(0)


features_surf = [
    'RankDiff_Log', 'CombinedEloDiff', 'EloMomentum',
    'Ace_1_form', 'DF_1_form', 'BP_1_form',
    'Surface_Grass', 'Surface_Clay', 'Surface_Hard'
]


present = [c for c in features_surf if c in df_surf.columns]
missing = [c for c in features_surf if c not in df_surf.columns]
print("Surface model feature check - present:", present)
print("Surface model feature check - missing:", missing)

for col in missing:
    df_surf[col] = 0.0

X_surf = df_surf[features_surf].copy()

def coerce_col(col):
    if pd.api.types.is_bool_dtype(col):
        return col.astype(int)
    
    if pd.api.types.is_object_dtype(col):
        try:
            coerced = pd.to_numeric(col, errors='coerce')
            if coerced.notna().any():
                return coerced
            else:
                return col.fillna(0)
        except:
            return col.fillna(0)
    return col

X_surf = X_surf.apply(coerce_col, axis=0).fillna(0)


for col in X_surf.columns:
    if isinstance(X_surf[col], pd.DataFrame):
        print(f" Column '{col}' is unexpectedly a DataFrame. Replacing with first subcolumn.")
        X_surf[col] = X_surf[col].iloc[:, 0].fillna(0)


print("\nX_surf column types after coercion:")
print(X_surf.dtypes)

y_surf = df_surf['Target'].astype(int)

# Train/test split (pre-Wimbledon training; Wimbledon 2025 test)
train_mask_surf = df_surf['Date'] < wim_start_date
test_mask_surf = (
    df_surf['Tournament'].astype(str).str.contains("Wimbledon", case=False, na=False) &
    (df_surf['Date'].dt.year == 2025)
)

X_train_surf, y_train_surf = X_surf[train_mask_surf], y_surf[train_mask_surf]
X_test_surf,  y_test_surf  = X_surf[test_mask_surf],  y_surf[test_mask_surf]

print(f"\nSurface model training size: {X_train_surf.shape}, test size: {X_test_surf.shape}")

if X_test_surf.shape[0] > 0 and X_train_surf.shape[0] > 0:
    xgb_surf = XGBClassifier(
        max_depth=4, learning_rate=0.05, n_estimators=400,
        subsample=0.9, colsample_bytree=0.8, tree_method="hist",
        random_state=42, eval_metric="logloss"
    )
    try:
        xgb_surf.fit(X_train_surf, y_train_surf)
        preds_surf_test = xgb_surf.predict(X_test_surf)
        acc_surf = accuracy_score(y_test_surf, preds_surf_test)
        print(f"\nWimbledon 2025 (Surface Elo) Accuracy: {acc_surf:.3f}")
        print(classification_report(y_test_surf, preds_surf_test, digits=3))
    except Exception as e:
        print("\n Error during xgb_surf.fit():")
        print(e)
        print("\n-- X_train_surf dtypes --")
        print(X_train_surf.dtypes)
        print("\n-- X_train_surf head --")
        print(X_train_surf.head().to_string())
        raise
else:
    print("No Wimbledon 2025 matches found or insufficient training data")


# VALIDATION CHECKS: Wimbledon 2025 Surface Elo Predictions

print("\\n Post-Analysis of Wimbledon 2025 Surface Elo Results...")

if X_test_surf.shape[0] > 0:
    cm = confusion_matrix(y_test_surf, preds_surf_test)
    print("\\n Confusion Matrix [Wimbledon 2025, Surface Elo]:")
    print(cm)

    print("\\n Prediction Distribution:")
    print(pd.Series(preds_surf_test).value_counts(normalize=True))

    
    results = df_surf[test_mask_surf].copy()
    results['Pred'] = preds_surf_test
    
    player_accuracy_data = []
    for player in results['Player_1'].unique():
        player_matches = results[results['Player_1'] == player]
        if len(player_matches) > 0:
            accuracy = (player_matches['Target'] == player_matches['Pred']).mean()
            player_accuracy_data.append({'Player': player, 'Accuracy': accuracy, 'Matches': len(player_matches)})

    player_acc_df = pd.DataFrame(player_accuracy_data).sort_values('Accuracy', ascending=False)
    print("\\n Top 10 Player Prediction Accuracies (Wimbledon 2025):")
    print(player_acc_df.head(10).to_string(index=False))
else:
    print(" No Wimbledon 2025 matches found for validation.")


# WIMBLEDON 2025 DETAILED ANALYSIS

print("\\n" + "="*60)
print("WIMBLEDON 2025 DETAILED PREDICTIONS ANALYSIS")
print("="*60)


df_wim_2025 = df[wim_2025_mask].copy()

df_wim_2025 = df_wim_2025[~df_wim_2025['Round'].astype(str).str.contains('Qualifying', case=False, na=False)].copy()

df_wim_2025 = df_wim_2025[~df_wim_2025['Round'].isna()].copy()

df_wim_2025 = normalize_rounds(df_wim_2025)

print(df_wim_2025['Source'].value_counts(dropna=False))


_ = dedupe_matches_for_display(df_wim_2025)



print(df_wim_2025['Round'].value_counts(dropna=False))

print(df_wim_2025[df_wim_2025['Source']=='Excel']['Round'].value_counts(dropna=False))


if 'SurfEloDiff' not in df_wim_2025.columns:
    df_wim_2025 = calculate_surface_elo(df_wim_2025)


Xg = df_wim_2025[features_general].fillna(0)
df_wim_2025['Pred_general'] = best_model.predict(Xg)
df_wim_2025['PredProb_general'] = best_model.predict_proba(Xg)[:, 1]

Xs = df_wim_2025[features_surf].fillna(0)
df_wim_2025['Pred_surf'] = xgb_surf.predict(Xs)
df_wim_2025['PredProb_surf'] = xgb_surf.predict_proba(Xs)[:, 1]


df_wim_2025['Predicted_Winner_surf'] = df_wim_2025.apply(
    lambda r: r['Player_1'] if r['Pred_surf'] == 1 else r['Player_2'], axis=1
)
df_wim_2025['Actual_Winner'] = df_wim_2025.apply(
    lambda r: r['Player_1'] if r['Target'] == 1 else r['Player_2'], axis=1
)
df_wim_2025['Correct_surf'] = (df_wim_2025['Pred_surf'] == df_wim_2025['Target'])
df_wim_2025['Correct_general'] = (df_wim_2025['Pred_general'] == df_wim_2025['Target'])


df_wim_2025 = normalize_rounds(df_wim_2025)


df_display = dedupe_matches_for_display(df_wim_2025)


print(f"\\nGeneral Model Accuracy: {accuracy_score(df_display['Target'], df_display['Pred_general']):.3%}")
print(classification_report(df_display['Target'], df_display['Pred_general'], digits=3))

print(f"\\nSurface-Specific Elo Model Accuracy: {accuracy_score(df_display['Target'], df_display['Pred_surf']):.3%}")
print(classification_report(df_display['Target'], df_display['Pred_surf'], digits=3))


# Accuracy by Round (clean, ordered, with counts)
preferred_round_order = [
    'The Final',
    'Semifinals',
    'Quarterfinals',
    'Round of 16',
    'Round of 32',
    'Round of 64',
    'Round of 128',
    
    '4th Round', '3rd Round', '2nd Round', '1st Round',
    'Unknown'
]

acc_general = df_display.groupby('Round', observed=False)['Correct_general'].mean()
acc_surf    = df_display.groupby('Round', observed=False)['Correct_surf'].mean()
counts      = df_display['Round'].value_counts()

acc_table = pd.DataFrame(index=preferred_round_order)
acc_table['N'] = counts.reindex(preferred_round_order).fillna(0).astype(int)
acc_table['Acc_General'] = acc_general.reindex(preferred_round_order)
acc_table['Acc_Surface'] = acc_surf.reindex(preferred_round_order)

def fmt_acc(series, n_series):
    out = []
    for a, n in zip(series, n_series):
        out.append('n/a' if n == 0 or pd.isna(a) else f"{a:.2%}")
    return out

print("\nAccuracy by Round (General vs Surface) — ordered and with counts:")
acc_table_print = acc_table.copy()
acc_table_print['Acc_General'] = fmt_acc(acc_table['Acc_General'], acc_table['N'])
acc_table_print['Acc_Surface'] = fmt_acc(acc_table['Acc_Surface'], acc_table['N'])
print(acc_table_print.to_string())

# Optional plot for rounds with N>0
plot_mask = acc_table['N'] > 0
plt.figure(figsize=(12,6))
acc_table.loc[plot_mask, 'Acc_General'].plot(kind='bar', color='skyblue', alpha=0.8, label='General')
acc_table.loc[plot_mask, 'Acc_Surface'].plot(kind='bar', color='mediumseagreen', alpha=0.6, label='Surface')
plt.title("Model Accuracy by Round — Wimbledon 2025 (deduped)")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show(block=False)

df_display['PredProb_Player_1'] = df_display['PredProb_surf']
df_display['PredProb_Player_2'] = 1 - df_display['PredProb_surf']


df_display['PredProb_Player_1'] = (df_display['PredProb_Player_1'] * 100).round(1).astype(str) + '%'
df_display['PredProb_Player_2'] = (df_display['PredProb_Player_2'] * 100).round(1).astype(str) + '%'


display_cols_surf = [
    'Round', 'Player_1', 'Player_2',
    'PredProb_Player_1', 'PredProb_Player_2',
    'Predicted_Winner_surf', 'Actual_Winner', 'Correct_surf'
]
page_size = 10
total_rows = len(df_display)
print("\\n" + "="*80)
print("DETAILED PREDICTIONS - SURFACE-SPECIFIC ELO MODEL (DEDUPED)")
print("="*80)
for start in range(0, total_rows, page_size):
    end = min(start + page_size, total_rows)
    print(f"\\nRows {start+1} to {end}:")
    print("-" * 80)
    print(df_display.iloc[start:end][display_cols_surf].to_string(index=False))
    if end < total_rows:
        user_input = input(f"\\nShow rows {end+1} to {min(end+page_size, total_rows)}? (Press Enter to continue, 'q' to quit): ")
        if user_input.lower() == 'q':
            break

print("\\n Analysis Complete")
