import time
import os
import re
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

rounds_order = [
    "1st Round Qualifying",
    "2nd Round Qualifying",
    "3rd Round Qualifying",
    "Round of 128",
    "Round of 64",
    "Round of 32",
    "Round of 16",
    "Quarterfinal",
    "Semifinal",
    "Final"
]

def get_score_from_stats_page(driver, stats_url):
    """Extract score, duration and date from ATP stats page with tiebreak handling"""
    if not stats_url:
        return "", "", ""

    try:
        driver.get(stats_url)

      
        selectors = ["div.scores", "div.match-stats-scores", "section.match-stats"]
        container = None
        for sel in selectors:
            try:
                WebDriverWait(driver, 8).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, sel))
                )
                container = sel
                break
            except:
                continue
        if not container:
            raise ValueError("No scoreboard container found")

        soup = BeautifulSoup(driver.page_source, "html.parser")
        block = soup.select_one(container)

        #Extracting duration
        duration = ""
        dur = block.find("span", string=re.compile(r"^\d{1,2}:\d{2}:\d{2}$"))
        if dur:
            duration = dur.get_text(strip=True)
        else:
            dur2 = soup.find(string=re.compile(r"^\s*\d{1,2}:\d{2}:\d{2}\s*$"))
            if dur2:
                duration = dur2.strip()
        print(f"Duration: {duration}")

        #Extracting match date
        match_date = ""
        date_elem = soup.find("span", string=re.compile(r"\d{1,2}\s+\w+\s+\d{4}"))
        if date_elem:
            match_date = date_elem.strip()
        print(f"Date: {match_date}")

        #Extracting scores
        numbers = [s.get_text(strip=True) for s in block.select("span") if s.get_text(strip=True).isdigit()]
        print("All numbers:", numbers)

        if len(numbers) < 4:
            return "", duration, match_date

        if len(numbers) % 2 != 0:
            numbers = numbers[:-1]

        half = len(numbers) // 2
        left = numbers[:half]
        right = numbers[half:]

        print("Player A:", left)
        print("Player B:", right)

        sets, i = [], 0
        while i < len(left):
            lval, rval = left[i], right[i]
            ltb, rtb = None, None

            if i + 1 < len(left) and (lval in ["6", "7"]) and (rval in ["6", "7"]):
                if i + 1 < len(left) and i + 1 < len(right) and left[i+1].isdigit() and right[i+1].isdigit():
                    ltb, rtb = left[i+1], right[i+1]
                    i += 1

            if ltb and rtb:
                sets.append(f"{lval}({ltb})-{rval}({rtb})")
            else:
                sets.append(f"{lval}-{rval}")

            i += 1

        return " ".join(sets), duration, match_date

    except Exception as e:
        print(f"Stats page score extraction failed: {e}")
        return "", "", ""

def get_dates_by_round(driver):
    dates_by_round = {}
    try:
        soup = BeautifulSoup(driver.page_source, "html.parser")

        
        day_groups = soup.select("div.tournament-day")
        for day in day_groups:
            
            h4 = day.find("h4")
            if not h4:
                continue
            raw = h4.get_text(" ", strip=True)
            clean_text = raw.split("Day")[0].strip().rstrip(",")
            try:
                date_dt = datetime.strptime(clean_text, "%a, %d %B, %Y")
                date_str = date_dt.strftime("%Y-%m-%d")
            except Exception:
                continue

            
            accordions = day.select("div.atp_accordion-item")
            for acc in accordions:
                strong = acc.select_one("div.match-header strong")
                if strong:
                    raw_round = strong.get_text(" ", strip=True)
                    round_name = clean_round_name(raw_round.split("-", 1)[0].strip())
                    if round_name:
                        dates_by_round[round_name] = date_str
    except Exception as e:
        print(f"Error extracting dates by round: {e}")

    return dates_by_round

def get_tourney_dates(driver):
    
    tourney_start_date_dt = None
    tourney_final_date_dt = None

    try:
        
        date_span = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, "//div[@class='date-location']/span[2]"))
        )
        date_text = date_span.text.strip()
        print(f"Tournament dates raw: {date_text}")

    
        date_text = date_text.replace(',', '')
        year_match = re.search(r"\d{4}", date_text)
        year = year_match.group(0) if year_match else None

        if not year:
            return None, None

        
        if "-" in date_text:
            start_part, end_part = date_text.split("-", 1)
            start_part = start_part.strip()
            end_part = end_part.strip()
            
            
            month_match = re.search(r"[A-Za-z]+", end_part)
            if not month_match:
                month = re.search(r"[A-Za-z]+", start_part).group(0)
                end_part = f"{end_part} {month}"
            
            start_date_str = f"{start_part} {year}"
            end_date_str = f"{end_part} {year}"

            
            for fmt in ("%d %b %Y", "%d %B %Y"):
                try:
                    tourney_start_date_dt = datetime.strptime(start_date_str, fmt).date()
                    tourney_final_date_dt = datetime.strptime(end_date_str, fmt).date()
                    break
                except:
                    continue
        else:
            
            start_date_str = f"{date_text} {year}"
            for fmt in ("%d %b %Y", "%d %B %Y"):
                try:
                    tourney_start_date_dt = tourney_final_date_dt = datetime.strptime(start_date_str, fmt).date()
                    break
                except:
                    continue

        return tourney_start_date_dt, tourney_final_date_dt

    except Exception as e:
        print(f"Could not scrape tournament dates: {e}")
        return None, None

def duration_to_minutes(duration_str):
    if not duration_str:
        return None
    parts = duration_str.split(":")
    if len(parts) != 3:
        return None
    hours, minutes, seconds = map(int, parts)
    total_minutes = hours * 60 + minutes + (1 if seconds >= 30 else 0)
    return total_minutes

def compute_match_date_dynamic(round_name, start_date, final_date, rounds_order=rounds_order):
    try:
        total_days = (final_date - start_date).days
        if round_name not in rounds_order:
            return None
        round_index = rounds_order.index(round_name)
        max_index = len(rounds_order) - 1
        day_offset = int(round_index * total_days / max_index)
        match_date = start_date + timedelta(days=day_offset)
        return match_date.strftime("%Y-%m-%d")
    except Exception as e:
        print(f"Error computing match date: {e}")
        return None

def get_surface_for_tournament(tournament_name: str) -> str:
    name = tournament_name.lower()
    surfaces = {
        # Grand Slams
        "australian open": "Hard",
        "roland garros": "Clay",
        "french open": "Clay",
        "wimbledon": "Grass",
        "us open": "Hard",
        # ATP Finals
        "atp finals": "Hard (i)",
        # Masters 1000
        "indian wells": "Hard",
        "miami open": "Hard",
        "monte-carlo masters": "Clay",
        "madrid open": "Clay",
        "italian open": "Clay",
        "canadian open": "Hard",
        "cincinnati open": "Hard",
        "shanghai masters": "Hard (i)",
        "paris masters": "Hard (i)",
        # ATP 500 (2025 list)
        "dallas open": "Hard (i)",
        "rotterdam open": "Hard (i)",
        "qatar open": "Hard",
        "rio open": "Clay",
        "dubai tennis championships": "Hard",
        "mexican open": "Hard",
        "barcelona open": "Clay",
        "bavarian international": "Clay",
        "queens club championships": "Grass",
        "halle open": "Grass",
        "hamburg european open": "Clay",
        "china open": "Hard",
        "japan open": "Hard",
        "vienna open": "Hard (i)",
        "swiss indoors": "Hard (i)",
        # ATP 250 (2025 list)
        "brisbane international": "Hard",
        "hong kong tennis open": "Hard",
        "adelaide international": "Hard",
        "asb classic": "Hard",
        "open occitanie": "Hard (i)",
        "open 13 provence": "Hard (i)",
        "delray beach open": "Hard",
        "argentina open": "Clay",
        "chile open": "Clay",
        "u.s. men's clay court championships": "Clay",
        "grand prix hassan ii": "Clay",
        "romanian open": "Clay",
        "geneva open": "Clay",
        "stuttgart open": "Grass",
        "rosmalen grass court championships": "Grass",
        "mallorca open": "Grass",
        "eastbourne international": "Grass",
        "los cabos open": "Hard",
        "swedish open": "Clay",
        "swiss open": "Clay",
        "croatia open": "Clay",
        "generali open kitzbühel": "Clay",
        "winston-salem open": "Hard",
        "chengdu open": "Hard",
        "hangzhou open": "Hard",
        "almaty open": "Hard (i)",
        "european open": "Hard (i)",
        "stockholm open": "Hard (i)",
        "moselle open": "Hard (i)",
        "hellenic championship": "Hard (i)",
    }
    for key, surf in surfaces.items():
        if key in name:
            return surf
    return "Unknown"

def get_tournament_level(tournament_name: str) -> str:
    name = tournament_name.lower()
    if "australian open" in name or "roland garros" in name or "french open" in name or "wimbledon" in name or "us open" in name:
        return "G"   # Grand Slam
    if "masters" in name or "indian wells" in name or "miami" in name or "monte-carlo" in name or "madrid" in name or "italian" in name or "canadian" in name or "cincinnati" in name or "shanghai" in name or "paris" in name:
        return "M"   # Masters 1000
    if "atp finals" in name:
        return "F"   # Finals
    if "open" in name or "championships" in name or "indoors" in name:
        return "500/250"
    return "Unknown"

def get_draw_size_for_tournament(tournament_name: str) -> int:
    name = tournament_name.lower()
    draw_sizes = {
        "australian open": 128,
        "roland garros": 128,
        "french open": 128,
        "wimbledon": 128,
        "us open": 128,
        "atp finals": 8,
        "indian wells": 96,
        "miami open": 96,
        "monte-carlo masters": 56,
        "madrid open": 96,
        "italian open": 96,
        "canadian open": 56,
        "cincinnati open": 56,
        "shanghai masters": 96,
        "paris masters": 48,
        "dallas open": 32,
        "rotterdam open": 32,
        "qatar open": 32,
        "rio open": 32,
        "dubai tennis championships": 32,
        "mexican open": 32,
        "barcelona open": 48,
        "bavarian international": 32,
        "queens club championships": 32,
        "halle open": 32,
        "hamburg european open": 32,
        "china open": 32,
        "japan open": 32,
        "vienna open": 32,
        "swiss indoors": 32,
        "brisbane international": 32,
        "hong kong tennis open": 32,
        "adelaide international": 28,
        "asb classic": 28,
        "open occitanie": 28,
        "open 13 provence": 28,
        "delray beach open": 28,
        "argentina open": 28,
        "chile open": 28,
        "u.s. men's clay court championships": 28,
        "grand prix hassan ii": 28,
        "romanian open": 28,
        "geneva open": 28,
        "stuttgart open": 28,
        "rosmalen grass court championships": 28,
        "mallorca open": 28,
        "eastbourne international": 28,
        "los cabos open": 28,
        "swedish open": 28,
        "swiss open": 28,
        "croatia open": 28,
        "generali open kitzbühel": 28,
        "winston-salem open": 48,
        "chengdu open": 28,
        "hangzhou open": 28,
        "almaty open": 28,
        "european open": 28,
        "stockholm open": 28,
        "moselle open": 28,
        "hellenic championship": 28,
    }
    for key, size in draw_sizes.items():
        if key in name:
            return size
    return None

def clean_round_name(raw_round: str) -> str:
    if not raw_round:
        return ""

    
    clean = raw_round.split("-", 1)[0].strip()

    lower_clean = clean.lower()

    
    if "final" in lower_clean and "semi" not in lower_clean and "quarter" not in lower_clean:
        return "Final"
    elif "semi" in lower_clean:
        return "Semifinal"
    elif "quarter" in lower_clean:
        return "Quarterfinal"

    
    return clean

def extract_date_from_round(raw_round: str) -> str:
    if not raw_round or "-" not in raw_round:
        return ""
    date_part = raw_round.split("-", 1)[1].strip().replace("\n", " ")
    return date_part

def determine_winner_from_score(score: str, p1_name: str, p2_name: str):
    if not score:
        return None, None, 0, 0
    p1_sets, p2_sets = 0, 0
    for set_score in score.split():
        if "-" not in set_score:
            continue
        left_str, right_str = set_score.split("-", 1)
        left_num_str = left_str.split("(")[0].strip()
        right_num_str = right_str.split("(")[0].strip()
        if not left_num_str.isdigit() or not right_num_str.isdigit():
            # Skip invalid or empty scores
            continue
        left = int(left_num_str)
        right = int(right_num_str)
        if left > right:
            p1_sets += 1
        elif right > left:
            p2_sets += 1
    if p1_sets > p2_sets:
        return p1_name, p2_name, p1_sets, p2_sets
    elif p2_sets > p1_sets:
        return p2_name, p1_name, p2_sets, p1_sets
    else:
        return None, None, p1_sets, p2_sets

def get_h2h_data(driver, p1_profile, p2_profile):
    if not p1_profile or not p2_profile:
        return {}
    def extract_player_stats(soup, section_class):
        stats = {}
        section = soup.select_one(f"div.{section_class}")
        if section:
            for item in section.select("li"):
                label_elem = item.select_one("span.label")
                value_elem = item.select_one("span.value")
                if label_elem and value_elem:
                    label = label_elem.get_text(strip=True)
                    value = value_elem.get_text(strip=True)
                    stats[label] = value
        return stats
    try:
        p1_slug, p1_id = p1_profile.split("/")[-3], p1_profile.split("/")[-2]
        p2_slug, p2_id = p2_profile.split("/")[-3], p2_profile.split("/")[-2]
        h2h_url = f"https://www.atptour.com/en/players/atp-head-2-head/{p1_slug}-vs-{p2_slug}/{p1_id}/{p2_id}"
        driver.get(h2h_url)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.win-stats"))
        )
        time.sleep(2)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        h2h_left, h2h_right = "", ""
        win_stats = soup.select("div.win-stats div.stats")
        if win_stats:
            players = win_stats[0].select("div.player, div.opponent")
            if len(players) >= 2:
                h2h_left, h2h_right = players[0].get_text(strip=True), players[1].get_text(strip=True)
        shared_stats_p1, shared_stats_p2 = {}, {}
        shared_section = soup.select_one("div.shared-data ul")
        if shared_section:
            for item in shared_section.select("li"):
                spans = item.select("span")
                if len(spans) >= 3:
                    label = spans[1].get_text(strip=True)
                    shared_stats_p1[label] = spans[0].get_text(strip=True)
                    shared_stats_p2[label] = spans[2].get_text(strip=True)
        p1_individual = extract_player_stats(soup, "player-data")
        p2_individual = extract_player_stats(soup, "opponent-data")
        h2h_data = {
            "H2H Wins P1": h2h_left,
            "H2H Wins P2": h2h_right,
        }
        for k, v in p1_individual.items():
            h2h_data[f"P1 {k}"] = v
        for k, v in p2_individual.items():
            h2h_data[f"P2 {k}"] = v
        for k, v in shared_stats_p1.items():
            h2h_data[f"P1 {k}"] = v
        for k, v in shared_stats_p2.items():
            h2h_data[f"P2 {k}"] = v
        return h2h_data
    except Exception as e:
        print(f"H2H scrape failed: {e}")
        return {}


stat_name_map = {
    "serverating": "Serve Rating",
    "aces": "Aces",
    "doublefaults": "Double Faults",
    "firstserve": "1st Serve",
    "1stservepointswon": "1st Serve Points Won",
    "2ndservepointswon": "2nd Serve Points Won",
    "servicepointswon": "Service Points Won",
    "breakpointssaved": "Break Points Saved",
    "servicegamesplayed": "Service Games Played",
    "returnrating": "Return Rating",
    "1stservereturnpointswon": "1st Serve Return Points Won",
    "2ndservereturnpointswon": "2nd Serve Return Points Won",
    "breakpointsconverted": "Break Points Converted",
    "returngamesplayed": "Return Games Played",
    "returnpointswon": "Return Points Won",
    "totalpointswon": "Total Points Won",
    "netpointswon": "Net Points Won",
    "winners": "Winners",
    "unforcederrors": "Unforced Errors",
    
}

def normalize_stat_name(name):
    return name.lower().replace(" ", "").replace("-", "")

def map_stat_name(name):
    norm_name = normalize_stat_name(name)
    if norm_name in stat_name_map:
        return stat_name_map[norm_name]
    for key in stat_name_map:
        if key in norm_name:
            return stat_name_map[key]
    return None

def get_match_stats(driver, stats_url):
    stats_data = {}

    def safe_text(elem):
        try:
            
            return elem.find_element(By.TAG_NAME, "a").text.strip()
        except:
            try:
                
                return elem.find_element(By.TAG_NAME, "span").text.strip()
            except:
                return elem.text.strip()

    driver.get(stats_url)
    time.sleep(5)  

    stat_tiles = driver.find_elements(By.CSS_SELECTOR, "div.statTileWrapper")

    for tile in stat_tiles:
        try:
            desktop_view = tile.find_element(By.CSS_SELECTOR, "div.desktopView")
            
            stat_name_raw = desktop_view.find_element(By.CSS_SELECTOR, "div.labelWrappper > div.labelBold").text.strip()

            mapped_name = map_stat_name(stat_name_raw)
            if mapped_name is None:
                
                continue

            # Player 1 container
            p1_container = desktop_view.find_element(By.CSS_SELECTOR, "div.statWrapper.p1Stats")
            # Player 2 container
            p2_container = desktop_view.find_element(By.CSS_SELECTOR, "div.statWrapper.p2Stats")

            # Player 1 value: try labelBold or label with player1 class
            try:
                p1_val_elem = p1_container.find_element(By.CSS_SELECTOR, "div.labelBold.player1")
            except:
                p1_val_elem = p1_container.find_element(By.CSS_SELECTOR, "div.label.player1")
            p1_val = safe_text(p1_val_elem)

            # Player 2 value: try labelBold or label with player2 class
            try:
                p2_val_elem = p2_container.find_element(By.CSS_SELECTOR, "div.labelBold.player2")
            except:
                p2_val_elem = p2_container.find_element(By.CSS_SELECTOR, "div.label.player2")
            p2_val = safe_text(p2_val_elem)

            stats_data[f"{mapped_name} P1"] = p1_val
            stats_data[f"{mapped_name} P2"] = p2_val

            print(f"Extracted {mapped_name}: P1={p1_val}, P2={p2_val}")

        except Exception as e:
            print(f"Error extracting stat tile: {e}")
            continue

    return stats_data

def build_score_with_tiebreaks(p1_scores_raw, p2_scores_raw):
    sets = []
    i, j = 0, 0
    while i < len(p1_scores_raw) and j < len(p2_scores_raw):
        p1 = p1_scores_raw[i]
        p2 = p2_scores_raw[j]
        tiebreak = None
        if (p1 == '6' and p2 == '7') or (p1 == '7' and p2 == '6'):
            if p1 == '6' and i + 1 < len(p1_scores_raw):
                tiebreak = p1_scores_raw[i + 1]
                i += 1
            elif p2 == '6' and j + 1 < len(p2_scores_raw):
                tiebreak = p2_scores_raw[j + 1]
                j += 1
        pair = f"{p1}-{p2}"
        if tiebreak is not None:
            pair += f"({tiebreak})"
        sets.append(pair)
        i += 1
        j += 1
    while i < len(p1_scores_raw):
        sets.append(p1_scores_raw[i])
        i += 1
    while j < len(p2_scores_raw):
        sets.append(p2_scores_raw[j])
        j += 1
    return " ".join(sets)

#MAIN SCRIPT
driver = uc.Chrome(version_main=140)

tournament_urls = [
    "https://www.atptour.com/en/scores/archive/eastbourne/741/2025/results",
    "https://www.atptour.com/en/scores/archive/mallorca/8994/2025/results",
    "https://www.atptour.com/en/scores/archive/halle/500/2025/results",
    "https://www.atptour.com/en/scores/archive/london/311/2025/results",
    "https://www.atptour.com/en/scores/archive/s-hertogenbosch/440/2025/results",
    "https://www.atptour.com/en/scores/archive/stuttgart/321/2025/results",
    "https://www.atptour.com/en/scores/archive/roland-garros/520/2025/results",
    "https://www.atptour.com/en/scores/archive/geneva/322/2025/results",
    "https://www.atptour.com/en/scores/archive/hamburg/414/2025/results",
    "https://www.atptour.com/en/scores/archive/rome/416/2025/results",
    "https://www.atptour.com/en/scores/archive/madrid/1536/2025/results",
    "https://www.atptour.com/en/scores/archive/munich/308/2025/results",
    "https://www.atptour.com/en/scores/archive/barcelona/425/2025/results",
    "https://www.atptour.com/en/scores/archive/monte-carlo/410/2025/results",
    "https://www.atptour.com/en/scores/archive/bucharest/4462/2025/results",
    "https://www.atptour.com/en/scores/archive/marrakech/360/2025/results",
    "https://www.atptour.com/en/scores/archive/houston/717/2025/results",
    "https://www.atptour.com/en/scores/archive/miami/403/2025/results",
    "https://www.atptour.com/en/scores/archive/indian-wells/404/2025/results",
    "https://www.atptour.com/en/scores/archive/santiago/8996/2025/results",
    "https://www.atptour.com/en/scores/archive/acapulco/807/2025/results",
    "https://www.atptour.com/en/scores/archive/dubai/495/2025/results",
    "https://www.atptour.com/en/scores/archive/rio-de-janeiro/6932/2025/results",
    "https://www.atptour.com/en/scores/archive/doha/451/2025/results",
    "https://www.atptour.com/en/scores/archive/buenos-aires/506/2025/results",
    "https://www.atptour.com/en/scores/archive/delray-beach/499/2025/results",
    "https://www.atptour.com/en/scores/archive/marseille/496/2025/results",
    "https://www.atptour.com/en/scores/archive/rotterdam/407/2025/results",
    "https://www.atptour.com/en/scores/archive/dallas/424/2025/results",
    "https://www.atptour.com/en/scores/archive/montpellier/375/2025/results",
    "https://www.atptour.com/en/scores/archive/australian-open/580/2025/results",
    "https://www.atptour.com/en/scores/archive/auckland/301/2025/results",
    "https://www.atptour.com/en/scores/archive/adelaide/8998/2025/results",
    "https://www.atptour.com/en/scores/archive/hong-kong/336/2025/results",
    "https://www.atptour.com/en/scores/archive/brisbane/339/2025/results",
    "https://www.atptour.com/en/scores/archive/perth-sydney/9900/2025/country-results",
]

all_data = []

#Loop through all tournaments
for url in tournament_urls:
    print(f"\n=== Processing tournament: {url} ===")
    driver.get(url)

    # Cookie acceptance
    try:
        accept_btn = WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Accept')]"))
        )
        accept_btn.click()
        print("Cookies accepted")
    except:
        print("No cookie banner")

    
    try:
        tournament = driver.find_element(By.CSS_SELECTOR, "h3.title a").text.strip()
    except:
        tournament = "Unknown Tournament"

    
    tourney_start_date_dt, tourney_final_date_dt = get_tourney_dates(driver)

    surface = get_surface_for_tournament(tournament)

    dates_by_round = get_dates_by_round(driver)
    print(f"Dates extraites par round : {dates_by_round}")

    
    WebDriverWait(driver, 20).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.atp_accordion-item"))
    )

    accordion_items = driver.find_elements(By.CSS_SELECTOR, "div.atp_accordion-item")

    
    for accordion in accordion_items:
        try:
            if accordion.get_attribute("data-default-state") == "close":
                toggler = accordion.find_element(By.CSS_SELECTOR, "div.atp_accordion-item-toggler")
                driver.execute_script("arguments[0].click();", toggler)
                time.sleep(1)
        except Exception as e:
            print(f"Couldn't expand section: {e}")

    time.sleep(2)

    accordion_items = driver.find_elements(By.CSS_SELECTOR, "div.atp_accordion-item")

    print(f"Total accordion items: {len(accordion_items)}")
    for i, acc in enumerate(accordion_items, 1):
        matches = acc.find_elements(By.CSS_SELECTOR, "div.match")
        print(f"Accordion {i} ({acc.text[:30]}...) has {len(matches)} matches")

    match_info = []

    # Process each accordion (round)
    for acc in accordion_items:
        try:
            raw_round = acc.find_element(By.CSS_SELECTOR, "div.match-header strong").text.strip()
            round_name = clean_round_name(raw_round.split("-", 1)[0].strip())
        except:
            round_name = "Unknown Round"

        
        date_info = None
        date_text_raw = acc.text.strip()

       
        m = re.search(r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+\d{1,2}\s+[A-Za-z]+,\s+\d{4}", date_text_raw)
        if m:
            raw_date_str = m.group(0)
            try:
                dt = datetime.strptime(raw_date_str, "%a, %d %B, %Y")
                date_info = dt.strftime("%Y-%m-%d")
            except Exception as e:
                print(f"Failed to parse {raw_date_str}: {e}")

        print(f"[Date] Accordion raw: {date_text_raw[:50]!r} -> parsed date_info: {date_info}")

        matches = acc.find_elements(By.CSS_SELECTOR, "div.match")
        for match in matches:
            print("Processing a match...")
            players, profile_links, iocs = [], [], []

            # Extract players
            try:
                for stats_item in match.find_elements(By.CSS_SELECTOR, "div.stats-item"):
                    try:
                        link = stats_item.find_element(By.CSS_SELECTOR, "div.name a")
                        players.append(link.text.strip())
                        profile_links.append(link.get_attribute("href"))
                        try:
                            use_elem = stats_item.find_element(By.CSS_SELECTOR, "svg.atp-flag use")
                            href = use_elem.get_attribute("href")
                            ioc = href.split("#flag-")[-1].upper()
                        except:
                            ioc = None
                        iocs.append(ioc)
                    except:
                        name_div = stats_item.find_element(By.CSS_SELECTOR, "div.name")
                        name_text = name_div.text.strip().split("(")[0].strip()
                        if name_text:
                            players.append(name_text)
                            profile_links.append(None)
                            iocs.append(None)
            except Exception as e:
                print(f"Error extracting players: {e}")

            print(f"Players found: {players}")
            if len(players) == 0:
                # Print a snippet of the match HTML to fix selectors if needed
                html_snippet = match.get_attribute("innerHTML")
                print(html_snippet[:400])

            # Extract score (from results page)
            try:
                score_divs = match.find_elements(By.CSS_SELECTOR, "div.scores")
                if len(score_divs) >= 2:
                    p1_spans = score_divs[0].find_elements(By.CSS_SELECTOR, "div.score-item span")
                    p2_spans = score_divs[1].find_elements(By.CSS_SELECTOR, "div.score-item span")
                    p1_scores_raw = [span.text.strip() for span in p1_spans if span.text.strip()]
                    p2_scores_raw = [span.text.strip() for span in p2_spans if span.text.strip()]
                    score_text = build_score_with_tiebreaks(p1_scores_raw, p2_scores_raw)
                else:
                    score_text = ""
            except Exception as e:
                print(f"Error extracting full score: {e}")
                score_text = ""

            
            stats_link = None
            try:
                selectors = [
                    "div.match-cta a[href*='stats-centre']",
                    "div.match-cta a[href*='match-stats']",
                    "a.match-stats-button",
                    "a[href*='match-stats']",
                ]
                for sel in selectors:
                    elems = match.find_elements(By.CSS_SELECTOR, sel)
                    if elems:
                        stats_link = elems[0].get_attribute("href")
                        if stats_link.startswith("/"):
                            stats_link = "https://www.atptour.com" + stats_link
                        break
            except Exception:
                stats_link = None

            # Append one row per match
            if len(players) >= 2:
                print(f"Appending match: {players[0]} vs {players[1]} in {round_name}")
                match_info.append({
                    "Tournament": tournament,
                    "Surface": surface,
                    "Round": round_name,
                    "Date": date_info,
                    "Player1": players[0],
                    "Player2": players[1],
                    "Player1 IOC": iocs[0] if len(iocs) > 0 else None,
                    "Player2 IOC": iocs[1] if len(iocs) > 1 else None,
                    "P1_Profile": profile_links[0] if len(profile_links) > 0 else None,
                    "P2_Profile": profile_links[1] if len(profile_links) > 1 else None,
                    "Score": score_text,
                    "Stats_URL": stats_link,
                })
                print(f"Nombre total de matchs collectés : {len(match_info)}")
            else:
                print("Skipping match due to insufficient players")

    
    for i, mi in enumerate(match_info, 1):
        print(f"Processing match {i}/{len(match_info)}: {mi['Player1']} vs {mi['Player2']} [{mi['Round']}]")

        
        match_date = mi.get("Date")
        stats_url = mi.get("Stats_URL")
        duration = ""
        score = ""

        
        if stats_url and isinstance(stats_url, str):
            print("Getting score, duration, and date from stats page...")
            score, duration, match_date_stats = get_score_from_stats_page(driver, stats_url)
            if duration:
                duration_minutes = duration_to_minutes(duration)
            
            if not match_date and match_date_stats:
                match_date = match_date_stats

        else:
            print("No Stats_URL found, skipping stats page extraction")
            duration_minutes = None

        winner, loser, w_sets, l_sets = determine_winner_from_score(mi.get("Score", ""), mi["Player1"], mi["Player2"])

        row_data = {
            "Tournament": mi["Tournament"],
            "Surface": mi["Surface"],
            "Round": mi["Round"],
            "Date": match_date or mi.get("Date", ""),
            "Player1 Name": mi["Player1"],
            "Player2 Name": mi["Player2"],
            "Player1 IOC": mi.get("Player1 IOC"),
            "Player2 IOC": mi.get("Player2 IOC"),
            "Score": mi.get("Score", score),
            "Duration": duration,
            "Duration Minutes": duration_to_minutes(duration) if duration else None,
            "Winner": winner,
            "Loser": loser,
            "Winner Sets": w_sets,
            "Loser Sets": l_sets,
            "Level": get_tournament_level(mi["Tournament"]),
            "DrawSize": get_draw_size_for_tournament(mi["Tournament"]),
            "BestOf": 5 if get_tournament_level(mi["Tournament"]) == "G" else 3,
        }

        #Attaching H2H data
        print("Getting H2H data...")
        h2h_data = get_h2h_data(driver, mi.get("P1_Profile"), mi.get("P2_Profile"))
        row_data.update(h2h_data)

        #Attaching Match Stats
        if stats_url and isinstance(stats_url, str):
            print("Getting match stats (bulletproof)...")
            stats_data = get_match_stats(driver, stats_url)
            row_data.update(stats_data)
            print("Match stats extracted")
        else:
            print("No Stats_URL, skipping match stats")

        all_data.append(row_data)
        print(f"Match {i} done\n")

    print(f"Tournament {tournament} completed. Matches: {len(match_info)}")

   
    output_file = "ATP_2025_Competitions.xlsx"
    new_data_df = pd.DataFrame(all_data)

    if os.path.exists(output_file):
        existing_df = pd.read_excel(output_file)
        existing_df = existing_df.dropna(axis=1, how='all')
        new_data_df = new_data_df.dropna(axis=1, how='all')
        combined_df = pd.concat([existing_df, new_data_df], ignore_index=True)
    else:
        combined_df = new_data_df

    combined_df.to_excel(output_file, index=False)
    print(f"Saved {len(new_data_df)} total matches so far into {output_file}")

    
    all_data = []
