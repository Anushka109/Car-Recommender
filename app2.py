from flask import Flask, render_template, request, send_file
import csv
import os
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Set up logging to debug image issues
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# -----------------------------
# Feature index mapping (for clarity)
# -----------------------------
# Index 0: male
# Index 1: age 20-40
# Index 2: age 40-60
# Index 3: age 60+
# Index 4: Regular Office (work)
# Index 5: Hybrid (work)
# Index 6: Freelance (work)
# Index 7: Purpose = 通勤・通学
# Index 8: Purpose = お子様の送迎
# Index 9: Purpose = 旅行
# Index 10: Purpose = 買い物・お出かけ
# Index 11: Purpose = 介護
# Index 12: Purpose = 趣味（アウトドア等）
# Index 13: Purpose = 勤務時での利用
# Index 14: Seating = 1
# Index 15: Seating = 2
# Index 16: Seating = 3
# Index 17: Seating = 4
# Index 18: Seating = 5
# Index 19: Seating = 6+
# Index 20: Budget = 100万円未満
# Index 21: Budget = 100万円～199万円
# Index 22: Budget = 200万円～299万円
# Index 23: Budget = 300万円～399万円
# Index 24: Budget = 400万円～499万円
# Index 25: Budget = 500万円以上
# Index 26: Priority = 価格
# Index 27: Priority = 燃費
# Index 28: Priority = デザイン
# Index 29: Priority = 安全装備
# Index 30: Priority = 収納・トランクスペース
# Index 31: Priority = 社内空間
# Index 32: Priority = 運転のしやすさ
# Index 33: Priority = 環境への配慮
# Index 34: Priority = 乗り心地
# Index 35: Priority = エンジンパワー
# Index 36: Priority = 先進性
# Index 37: Priority = 乗り降りしやすさ
# Index 38: Hobby = アウトドア
# Index 39: Hobby = 旅行・ドライブ・バイク
# Index 40: Hobby = ショッピング・グルメ
# Index 41: Hobby = ペット

# -----------------------------
# Car feature matrix (42 binary features)
# -----------------------------
car_data = {
    "Honda N-Van":   [0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1],
    "Honda N-One":   [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "Honda N-One E": [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    "Honda N-WGN":   [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "Honda N-Box":   [0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "Honda Fit":     [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "Honda Vezel":   [0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "Honda ZR-V":    [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
    "Honda WR-V":    [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "Honda Civic":   [1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
    "Honda Accord":  [0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    "Honda CRV":     [0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "Honda Freed":   [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "Honda StepWGN": [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "Honda Odyssey":  [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
    "Honda N-Van E": [0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1],
}

# -----------------------------
# Convert car_data to contextual vectors using PCA
# -----------------------------
car_names = list(car_data.keys())
car_vectors = np.array(list(car_data.values()))
pca = PCA(n_components=10)  # Reduce to 10-dimensional contextual vectors
car_contextual_vectors = pca.fit_transform(car_vectors)
car_contextual_data = dict(zip(car_names, car_contextual_vectors))

# Use PCA explained variance ratio as weights
FEATURE_WEIGHTS = pca.explained_variance_ratio_ / np.sum(pca.explained_variance_ratio_)  # Normalize weights

# -----------------------------
# Car images
# -----------------------------
car_images = {
    "Honda N-Van":   "image/honda-nvan.png",
    "Honda N-One":   "image/honda-none.png",
    "Honda N-One E": "image/honda-none-e.jpg",
    "Honda N-WGN":   "image/honda-nwgn.png",
    "Honda N-Box":   "image/honda-nbox.png",
    "Honda Fit":     "image/honda-fit.jpg",
    "Honda Vezel":   "image/honda-vezel.png",
    "Honda ZR-V":    "image/honda-zrv.png",
    "Honda WR-V":    "image/honda-wrv.png",
    "Honda Civic":   "image/honda-civic.jpg",
    "Honda Accord":  "image/honda-accord.png",
    "Honda CRV":     "image/honda-crv.jpg",
    "Honda Freed":   "image/honda-freed.png",
    "Honda StepWGN": "image/honda-stepwgn.png",
    "Honda Odyssey": "image/honda-odyssey.png",
    "Honda N-Van E": "image/honda-nvane.jpg",
}

# -----------------------------
# Short descriptions
# -----------------------------
car_descriptions = {
    "Honda N-Van":   "A versatile kei van with a spacious cargo area — great for small businesses, freelancers, and city deliveries.",
    "Honda N-One":   "A stylish kei car with retro charm and easy city maneuverability — perfect for short commutes and shopping.",
    "Honda N-One E": "An electric version of the N-One, offering eco-friendly commuting with modern features and a stylish design.",
    "Honda N-WGN":   "A practical kei with efficient packaging and low running costs — ideal for daily commuting and errands.",
    "Honda N-Box":   "A roomy kei car with a tall cabin — surprisingly spacious and excellent for families in tight urban areas.",
    "Honda Fit":     "A compact, agile hatchback with great fuel efficiency — ideal for commuting and weekend errands.",
    "Honda Vezel":   "A compact crossover that balances style, utility, and fuel efficiency — good for small families and weekend trips.",
    "Honda ZR-V":    "A stylish crossover that blends everyday practicality with a sporty feeling for weekend drives.",
    "Honda WR-V":    "A compact SUV offering higher ride height and versatility for city and light off-road outings.",
    "Honda Civic":   "A sporty compact car with modern design and strong driving character — well-suited for younger commuters.",
    "Honda Accord":  "A comfortable, refined sedan with long-distance cruising in mind — great for executives and weekend comfort.",
    "Honda CRV":     "A mid-size SUV that offers space, safety, and capability — ideal for family adventures and longer trips.",
    "Honda Freed":   "A compact minivan with flexible seating arrangements — focused on family convenience and city driving.",
    "Honda StepWGN": "A roomy minivan engineered for families who need space and comfort on longer journeys.",
    "Honda Odyssey": "A premium family minivan with advanced features and comfortable seating for longer family trips.",
    "Honda N-Van E": "An electric version of the N-Van, combining eco-friendly driving with the practicality and spaciousness of a kei car.",
}

# -----------------------------
# Feature explanations for personalized reasons (in Japanese for consistency)
# -----------------------------
feature_expls = {
    # Gender (0)
    0: "男性向けのダイナミックなデザインとパワフルな走行フィール",
    # Age (1-3)
    1: "20代～30代のアクティブなライフスタイルにマッチしたスタイリッシュさ",
    2: "40代～50代の落ち着いた日常に適した上質な快適性",
    3: "60代以上の安心ドライブを支える扱いやすさと視界の良さ",
    # Work (4-6)
    4: "オフィス通勤に最適な燃費効率と都市部の機動性",
    5: "ハイブリッドワークの柔軟な移動に適したコンパクト設計",
    6: "フリーランスの多様な活動をサポートする実用的なスペース",
    # Purposes (7-13)
    7: "通勤・通学の負担を軽減する優れた燃費と取り回し",
    8: "お子様送迎の安全性を高める先進安全装備と広い開口部",
    9: "旅行の楽しさを倍増させる長距離快適性と荷物収納",
    10: "買い物・お出かけを快適にする駐車しやすさと積載力",
    11: "介護の負担を和らげる低床構造と補助機能",
    12: "趣味（アウトドア等）を満喫できるタフさと多用途性",
    13: "勤務利用の効率を上げるプロフェッショナルなレイアウト",
    # Seating (14-19)
    14: "一人乗りのシンプルで集中できるコックピット",
    15: "二人乗りの親密でリラックスした空間",
    16: "三人での移動にバランスよく配置されたシート",
    17: "四人家族の日常をカバーするゆとりある室内",
    18: "五人乗車時の快適さを確保したシートアレンジ",
    19: "六人以上のグループ旅行に耐える広大なキャビン",
    # Hobbies (38-41)
    38: "アウトドアの冒険を支える耐久フレームと拡張収納",
    39: "旅行・ドライブの喜びを高めるパワーとハンドリング",
    40: "ショッピング・グルメのシーンにフィットする都市適性",
    41: "ペットとの時間を豊かにするフレンドリーなインテリア",
}

# -----------------------------
# Official Honda Japan URLs and used car URLs
# -----------------------------
car_urls = {
    "Honda N-Van":   {"new": "https://www.honda.co.jp/N-VAN/", "used": "https://www.carsensor.net/usedcar/search.php?STID=CS210610&SORT=19&CARC=HO_S111&KW=%E3%83%9B%E3%83%B3%E3%83%80%E3%82%AB%E3%83%BC%E3%82%BA%E8%8C%A8%E5%9F%8E%E8%A5%BF%20%E7%A6%8F%E5%B3%B6%E5%8D%97"},
    "Honda N-One":   {"new": "https://www.honda.co.jp/N-ONE/", "used": "https://www.carsensor.net/usedcar/search.php?STID=CS210610&SORT=19&CARC=HO_S098&KW=%E3%83%9B%E3%83%B3%E3%83%80%E3%82%AB%E3%83%BC%E3%82%BA%E8%8C%A8%E5%9F%8E%E8%A5%BF%20%E7%A6%8F%E5%B3%B6%E5%8D%97"},
    "Honda N-One E": {"new": "https://global.honda/jp/design/feature/2025nonee/", "used": ""},
    "Honda N-WGN":   {"new": "https://www.honda.co.jp/N-WGN/", "used": "https://www.carsensor.net/usedcar/search.php?CARC=HO_S100&KW=%E3%83%9B%E3%83%B3%E3%83%80%E3%82%AB%E3%83%BC%E3%82%BA%E8%8C%A8%E5%9F%8E%E8%A5%BF%20%E7%A6%8F%E5%B3%B6%E5%8D%97&SORT=19"},
    "Honda N-Box":   {"new": "https://www.honda.co.jp/Nbox/", "used": "https://www.carsensor.net/usedcar/search.php?CARC=HO_S094&KW=%E3%83%9B%E3%83%B3%E3%83%80%E3%82%AB%E3%83%BC%E3%82%BA%E8%8C%A8%E5%9F%8E%E8%A5%BF%20%E7%A6%8F%E5%B3%B6%E5%8D%97&SORT=19"},
    "Honda Fit":     {"new": "https://www.honda.co.jp/Fit/", "used": "https://www.carsensor.net/usedcar/search.php?STID=CS210610&SORT=19&CARC=HO_S028&KW=%E3%83%9B%E3%83%B3%E3%83%80%E3%82%AB%E3%83%BC%E3%82%BA%E8%8C%A8%E5%9F%8E%E8%A5%BF%20%E7%A6%8F%E5%B3%B6%E5%8D%97"},
    "Honda Vezel":   {"new": "https://www.honda.co.jp/VEZEL/", "used": "https://www.carsensor.net/usedcar/search.php?STID=CS210610&SORT=19&CARC=HO_S101&KW=%E3%83%9B%E3%83%B3%E3%83%80%E3%82%AB%E3%83%BC%E3%82%BA%E8%8C%A8%E5%9F%8E%E8%A5%BF%20%E7%A6%8F%E5%B3%B6%E5%8D%97"},
    "Honda ZR-V":    {"new": "https://www.honda.co.jp/ZR-V/", "used": "https://www.carsensor.net/usedcar/search.php?STID=CS210610&SORT=19&CARC=HO_S114&KW=%E3%83%9B%E3%83%B3%E3%83%80%E3%82%AB%E3%83%BC%E3%82%BA%E8%8C%A8%E5%9F%8E%E8%A5%BF%20%E7%A6%8F%E5%B3%B6%E5%8D%97"},
    "Honda WR-V":    {"new": "https://www.honda.co.jp/WR-V/", "used": "https://www.carsensor.net/usedcar/search.php?CARC=HO_S116&KW=%E3%83%9B%E3%83%B3%E3%83%80%E3%82%AB%E3%83%BC%E3%82%BA%E8%8C%A8%E5%9F%8E%E8%A5%BF%20%E7%A6%8F%E5%B3%B6%E5%8D%97&SORT=19"},
    "Honda Civic":   {"new": "https://www.honda.co.jp/CIVIC/", "used": "https://www.carsensor.net/usedcar/search.php?CARC=HO_S019&KW=%E3%83%9B%E3%83%B3%E3%83%80%E3%82%AB%E3%83%BC%E3%82%BA%E8%8C%A8%E5%9F%8E%E8%A5%BF%20%E7%A6%8F%E5%B3%B6%E5%8D%97&SORT=19"},
    "Honda Accord":  {"new": "https://www.honda.co.jp/ACCORD/", "used": "https://www.carsensor.net/usedcar/search.php?CARC=HO_S001&KW=%E3%83%9B%E3%83%B3%E3%83%80%E3%82%AB%E3%83%BC%E3%82%BA%E8%8C%A8%E5%9F%8E%E8%A5%BF%20%E7%A6%8F%E5%B3%B6%E5%8D%97&SORT=19"},
    "Honda CRV":     {"new": "https://www.honda.co.jp/auto-archive/cr-v/2022/", "used": "https://www.carsensor.net/usedcar/search.php?CARC=HO_S018&KW=%E3%83%9B%E3%83%B3%E3%83%80%E3%82%AB%E3%83%BC%E3%82%BA%E8%8C%A8%E5%9F%8E%E8%A5%BF%20%E7%A6%8F%E5%B3%B6%E5%8D%97&SORT=19"},
    "Honda Freed":   {"new": "https://www.honda.co.jp/FREED/", "used": "https://www.carsensor.net/usedcar/search.php?CARC=HO_S083&KW=%E3%83%9B%E3%83%B3%E3%83%80%E3%82%AB%E3%83%BC%E3%82%BA%E8%8C%A8%E5%9F%8E%E8%A5%BF%20%E7%A6%8F%E5%B3%B6%E5%8D%97&SORT=19"},
    "Honda StepWGN": {"new": "https://www.honda.co.jp/STEPWGN/", "used": "https://www.carsensor.net/usedcar/search.php?STID=CS210610&SORT=19&CARC=HO_S003&KW=%E3%83%9B%E3%83%B3%E3%83%80%E3%82%AB%E3%83%BC%E3%82%BA%E8%8C%A8%E5%9F%8E%E8%A5%BF%20%E7%A6%8F%E5%B3%B6%E5%8D%97"},
    "Honda Odyssey":  {"new": "https://www.honda.co.jp/ODYSSEY/", "used": "https://www.carsensor.net/usedcar/search.php?CARC=HO_S002&KW=%E3%83%9B%E3%83%B3%E3%83%80%E3%82%AB%E3%83%BC%E3%82%BA%E8%8C%A8%E5%9F%8E%E8%A5%BF%20%E7%A6%8F%E5%B3%B6%E5%8D%97&SORT=19"},
    "Honda N-Van E": {"new": "https://www.honda.co.jp/N-VAN-e/", "used": ""}
}

# -----------------------------
# Helper: weighted cosine similarity
# -----------------------------
def weighted_cosine_similarity(vec_a, vec_b):
    """
    Compute weighted cosine similarity between two vectors.
    Higher similarity (closer to 1) means better match.
    """
    # Apply weights to vectors
    weighted_a = vec_a * FEATURE_WEIGHTS
    weighted_b = vec_b * FEATURE_WEIGHTS
    # Compute cosine similarity (returns array, take first element)
    similarity = cosine_similarity([weighted_a], [weighted_b])[0][0]
    return similarity

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/questionnaire')
def questionnaire():
    return render_template("questionnaire.html")

@app.route('/result', methods=['POST'])
def result():
    # Collect user responses
    gender = request.form.get('gender', '')
    age_group = request.form.get('age', '')
    work_type = request.form.get('work', '')
    purpose_list = request.form.getlist('purpose')
    seating = request.form.get('seating', '')
    budget = request.form.get('budget', '')
    primary_priority = request.form.get('primary_priority', '')
    secondary_priorities = request.form.getlist('secondary_priority')[:2]
    hobbies = request.form.getlist('hobby')

    # Default purpose if none selected
    if not purpose_list:
        purpose_list = ["買い物・お出かけ"]
        logging.info("No purpose selected, defaulting to '買い物・お出かけ'")

    # Pad secondary priorities if less than 2
    if len(secondary_priorities) < 2:
        secondary_priorities += [None] * (2 - len(secondary_priorities))

    # Basic input validation
    warnings = []
    if budget == "100万円未満" and primary_priority in ["エンジンパワー", "先進性"]:
        warnings.append("低予算と高性能/先進性の優先順位は一致しない可能性があります。")
    if seating == "6人以上" and budget == "100万円未満":
        warnings.append("大型車（6人以上）と低予算は一致しない可能性があります。")

    # Encode user features in the same order as car_data vectors
    user_features = [
        1 if gender == "男性" else 0,                       # index 0
        1 if age_group == "20代～30代" else 0,             # index 1
        1 if age_group == "40代～50代" else 0,             # index 2
        1 if age_group == "60代以上" else 0,               # index 3
        1 if "Regular Office" in work_type else 0,          # index 4
        1 if "Hybrid" in work_type else 0,                  # index 5
        1 if "Freelance" in work_type else 0,               # index 6
        1 if "通勤・通学" in purpose_list else 0,          # index 7
        1 if "お子様の送迎" in purpose_list else 0,        # index 8
        1 if "旅行" in purpose_list else 0,                 # index 9
        1 if "買い物・お出かけ" in purpose_list else 0,    # index 10
        1 if "介護" in purpose_list else 0,                 # index 11
        1 if "趣味（アウトドア等）" in purpose_list else 0, # index 12
        1 if "勤務時での利用" in purpose_list else 0,       # index 13
        1 if seating == "1人" else 0,                      # index 14
        1 if seating == "2人" else 0,                      # index 15
        1 if seating == "3人" else 0,                      # index 16
        1 if seating == "4人" else 0,                      # index 17
        1 if seating == "5人" else 0,                      # index 18
        1 if seating == "6人以上" else 0,                  # index 19
        1 if budget == "100万円未満" else 0,               # index 20
        1 if budget == "100万円～199万円" else 0,          # index 21
        1 if budget == "200万円～299万円" else 0,          # index 22
        1 if budget == "300万円～399万円" else 0,          # index 23
        1 if budget == "400万円～499万円" else 0,          # index 24
        1 if budget == "500万円以上" else 0,               # index 25
        1 if primary_priority == "価格" else 0,             # index 26
        1 if primary_priority == "燃費" else 0,             # index 27
        1 if primary_priority == "デザイン" else 0,         # index 28
        1 if primary_priority == "安全装備" else 0,         # index 29
        1 if primary_priority == "収納・トランクスペース" else 0, # index 30
        1 if primary_priority == "社内空間" else 0,         # index 31
        1 if primary_priority == "運転のしやすさ" else 0,   # index 32
        1 if primary_priority == "環境への配慮" else 0,     # index 33
        1 if primary_priority == "乗り心地" else 0,         # index 34
        1 if primary_priority == "エンジンパワー" else 0,   # index 35
        1 if primary_priority == "先進性" else 0,           # index 36
        1 if primary_priority == "乗り降りしやすさ" else 0, # index 37
        1 if "アウトドア" in hobbies else 0,               # index 38
        1 if "旅行・ドライブ・バイク" in hobbies else 0,  # index 39
        1 if "ショッピング・グルメ" in hobbies else 0,     # index 40
        1 if "ペット" in hobbies else 0,                   # index 41
    ]

    # Convert user features to contextual vector using PCA
    user_contextual_vector = pca.transform([user_features])[0]

    # Compute cosine similarity to every car's contextual vector
    similarities = []
    for model, contextual_vector in car_contextual_data.items():
        sim = weighted_cosine_similarity(contextual_vector, user_contextual_vector)
        similarities.append((sim, model))

    # Sort by similarity (higher is better) then by name for tie-breaking
    similarities.sort(key=lambda x: (-x[0], x[1]))  # Negative for descending similarity
    best_similarity, best_car = similarities[0]

    # Prepare display data
    image = car_images.get(best_car, "image/default-car.jpg")
    # Log image path for debugging
    image_path = os.path.join(app.static_folder, image)
    if not os.path.exists(image_path):
        logging.error(f"Image file not found: {image_path}")
    else:
        logging.debug(f"Image file found: {image_path}")

    # Priority indices for secondary handling
    priority_indices = {
        "価格": 26,
        "燃費": 27,
        "デザイン": 28,
        "安全装備": 29,
        "収納・トランクスペース": 30,
        "社内空間": 31,
        "運転のしやすさ": 32,
        "環境への配慮": 33,
        "乗り心地": 34,
        "エンジンパワー": 35,
        "先進性": 36,
        "乗り降りしやすさ": 37
    }

    # Build personalized reasons based on matching features
    reasons = []

    # Primary priority (insert at front if match)
    if primary_priority in priority_indices and car_data[best_car][priority_indices[primary_priority]] == 1:
        reasons.insert(0, f"最優先事項の「{primary_priority}」を重視した選択です。")

    # Secondary priorities
    for sp in [p for p in secondary_priorities if p]:
        if sp in priority_indices and car_data[best_car][priority_indices[sp]] == 1:
            reasons.append(f"加えて「{sp}」の点でも優れています。")

    # Matching features (demographics, work, purposes, seating, hobbies)
    matching_indices = [0,1,2,3,4,5,6] + list(range(7,14)) + list(range(14,20)) + [38,39,40,41]
    for idx in matching_indices:
        if user_features[idx] == 1 and car_data[best_car][idx] == 1:
            expl = feature_expls.get(idx, "")
            if expl:
                reasons.append(f"{expl}が特徴です。")

    # Budget match
    for idx in range(20, 26):
        if user_features[idx] == 1 and car_data[best_car][idx] == 1:
            reasons.append("ご指定の予算内で購入可能な手頃な価格帯です。")
            break

    # Warnings if any
    if warnings:
        for w in warnings:
            reasons.append(f"<em>注意:</em> {w}")

    # Default if no reasons
    if not reasons:
        reasons = ["総合的な適合度が高く、あなたのニーズにバランスよく応える車種です。"]

    # Base description + dynamic reasons
    base_description = car_descriptions.get(best_car, "No description available.")
    dynamic_part = f"<br><br><strong>なぜこの車があなたに最適か:</strong><br><ul><li>" + "</li><li>".join(reasons) + "</li></ul>"
    description = base_description + dynamic_part

    # Save response to CSV
    csv_file = "responses.csv"
    file_exists = os.path.isfile(csv_file)
    try:
        with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Gender", "Age Group", "Work Type", "Purpose(s)", "Seating", "Budget", "Primary Priority", "Secondary Priorities", "Hobbies", "Recommended Car"])
            writer.writerow([
                gender, age_group, work_type, ", ".join(purpose_list), seating, budget, 
                primary_priority, ", ".join([p for p in secondary_priorities if p]), 
                ", ".join(hobbies), best_car
            ])
    except Exception as e:
        logging.error(f"Error writing CSV: {e}")

    # Get URLs for the recommended car
    car_url = car_urls.get(best_car, {"new": "#", "used": ""})
    new_car_url = car_url["new"]
    used_car_url = car_url["used"]

    # Render the result template
    return render_template("result.html", car=best_car, image=image, description=description, 
                         new_car_url=new_car_url, used_car_url=used_car_url, score=best_similarity)

@app.route('/download')
def download():
    csv_file = "responses.csv"
    if os.path.isfile(csv_file):
        return send_file(csv_file, as_attachment=True)
    return "<h3>No responses to download yet.</h3>"

@app.route('/download_pdf')
def download_pdf():
    csv_file = "responses.csv"
    if not os.path.isfile(csv_file):
        return "<h3>No responses to export.</h3>"
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        logging.error(f"Error reading CSV for PDF: {e}")
        return "<h3>Unable to read responses file.</h3>"

    pdf_file = "responses.pdf"
    try:
        doc = SimpleDocTemplate(pdf_file, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = [Paragraph("User Responses Report", styles['Title'])]
        data = [df.columns.tolist()] + df.values.tolist()
        table = Table(data, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2E8B57")), 
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ]))
        elements.append(table)
        doc.build(elements)
        return send_file(pdf_file, as_attachment=True)
    except Exception as e:
        logging.error(f"Error generating PDF: {e}")
        return "<h3>Failed to generate PDF.</h3>"

if __name__ == "__main__":
    app.run(debug=True, port=5017)