"""Deterministic nutrition and recipe scoring. No LLM calls are performed here."""
from __future__ import annotations
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

DATA_DIR = Path(__file__).parent / "data"
FIELDS = ("kcal", "carbs_g", "protein_g", "fat_g", "fiber_g", "sodium_mg")
DISPLAY_FIELDS = {"kcal": "열량(kcal)", "carbs_g": "탄수화물(g)", "protein_g": "단백질(g)", "fat_g": "지방(g)", "fiber_g": "식이섬유(g)", "sodium_mg": "나트륨(mg)"}
DEFAULT_AMOUNTS = {"계란": 50, "두부": 150, "양파": 100, "시금치": 100, "우유": 200,
                   "토마토": 150, "식용유": 5, "참기름": 5, "올리브유": 5, "밥": 150,
                   "당근": 80, "김치": 100, "간장": 10, "닭가슴살": 150, "양상추": 80,
                   "참치캔": 100, "감자": 150, "느타리버섯": 100, "오이": 100,
                   "플레인요거트": 150, "바나나": 100, "아몬드": 20, "대파": 50, "마늘": 10, "양배추": 150, "애호박":150, "치즈":25, "식빵":35, "새우":100, "소고기":150, "돼지고기":150, "사과":170, "고구마":200, "파프리카":150, "콩나물":150, "김":3, "오트밀":45}
ALIASES = {"달걀":"계란","달걀 ":"계란","에그":"계란","연두부":"두부", "버섯":"느타리버섯", "요거트":"플레인요거트", "양배추잎":"양배추", "파":"대파", "참치":"참치캔", "쌀밥":"밥"}
ALLERGENS = {"계란":{"난류"}, "두부":{"대두"}, "간장":{"대두","밀"},
             "우유":{"우유"},"플레인요거트":{"우유"},"아몬드":{"견과류"}, "참치캔":{"생선"}, "새우":{"갑각류"}, "고등어":{"생선"}, "치즈":{"우유"}, "버터":{"우유"}, "식빵":{"밀"}, "어묵":{"생선","밀"}, "소고기":{"소고기"}, "돼지고기":{"돼지고기"}, "고추장":{"대두","밀"}, "다진고기":{"돼지고기"}}
GOALS = ("균형", "고단백", "재료 소진")
DEFAULT_WEIGHTS = {"재료 활용":30, "영양 목표":30, "우선 소진":20, "조리 시간":10, "맛 조합":10}


def effective_weights(custom: Mapping[str, float]|None = None) -> dict[str,float]:
    """Normalize nonnegative experiment weights to a 100-point scale."""
    raw = dict(DEFAULT_WEIGHTS)
    if custom:
        raw.update(custom)
    if any(not math.isfinite(float(v)) or float(v)<0 for v in raw.values()):
        raise ValueError('가중치는 0 이상의 유한한 숫자여야 합니다.')
    total=sum(raw.values())
    if total<=0: raise ValueError('가중치 하나 이상을 0보다 크게 설정하세요.')
    return {k:100*float(v)/total for k,v in raw.items()}



def load_demo_nutrients() -> dict:
    with (DATA_DIR / "demo_nutrition.csv").open(encoding="utf-8-sig", newline="") as f:
        return {row['food']: {k: float(row[k]) for k in FIELDS} for row in csv.DictReader(f)}


def load_recipes() -> list[dict]:
    return json.loads((DATA_DIR / "recipes.json").read_text(encoding="utf-8"))


def load_compatibility() -> dict:
    with (DATA_DIR / 'compatibility.csv').open(encoding='utf-8-sig', newline='') as f:
        return {frozenset((r['food_a'], r['food_b'])): float(r['score']) for r in csv.DictReader(f)}


def normalize_food(text: str) -> str:
    text = str(text).strip()
    return ALIASES.get(text, text)


def parse_inventory(text: str, known_foods: Mapping[str, object]) -> tuple[dict, list[str]]:
    """Best-effort Korean quantities; all approximate unit conversions explicitly flagged."""
    inventory = {}
    warnings = []
    for chunk in re.split(r"[,，\n;、]+", text):
        chunk = chunk.strip()
        if not chunk:
            continue
        options = list(known_foods) + [k for k in ALIASES if k not in known_foods]
        matched = next((item for item in sorted(options, key=len, reverse=True) if item in chunk), None)
        if not matched:
            warnings.append(f"'{chunk}': 등록되지 않은 재료여서 제외했어요.")
            continue
        food = normalize_food(matched)
        if food not in known_foods:
            warnings.append(f"'{matched}': 현재 영양 데이터에 없어서 제외했어요.")
            continue
        suffix = chunk.replace(matched, " ").strip()
        mass = re.search(r"(\d+(?:\.\d+)?)\s*(?:g|그램)\b", suffix, re.I)
        if mass:
            grams = float(mass.group(1))
        else:
            n = re.search(r"(\d+(?:\.\d+)?)\s*(개|모|팩|캔|공기|알|장|컵)\b", suffix)
            if n:
                unit = n.group(2)
                unit_g = {"개": DEFAULT_AMOUNTS.get(food, 100),"모": 300,"팩": 150,"캔": 100,
                          "공기": 210,"알": DEFAULT_AMOUNTS.get(food, 50),"장": 25,"컵": 200}[unit]
                grams = float(n.group(1))*unit_g
                warnings.append(f"{food} {n.group(0)} → {grams:g}g으로 임시 환산했어요. 실제 중량을 확인해 주세요.")
            elif re.search(r"반\s*모", suffix):
                grams = 150
                warnings.append(f"{food} 반 모 → 150g으로 임시 환산했어요.")
            else:
                grams = DEFAULT_AMOUNTS.get(food, 100)
                warnings.append(f"{food}: 중량 정보가 없어 {grams:g}g으로 가정했어요.")
        if not math.isfinite(grams) or grams <= 0:
            warnings.append(f"{food}: 0 이하 또는 잘못된 중량은 제외했어요.")
            continue
        inventory[food] = inventory.get(food, 0.0) + grams
    return inventory, warnings


def recipe_nutrition(recipe: Mapping, database: Mapping[str, Mapping[str,float]]) -> tuple[dict|None, list[str]]:
    missing = [name for name in recipe['ingredients'] if name not in database]
    if missing:
        return None, missing
    out = {f: 0.0 for f in FIELDS}
    for food, grams in recipe['ingredients'].items():
        for field in FIELDS:
            out[field] += database[food][field] * grams / 100.0
    return {k: round(v, 1) for k, v in out.items()}, []


def compatibility_score(recipe: Mapping, compat: Mapping) -> float:
    foods = list(recipe['ingredients'])
    values = [compat[frozenset((foods[i],foods[j]))] for i in range(len(foods)) for j in range(i+1,len(foods)) if frozenset((foods[i],foods[j])) in compat]
    # 0.5 is a neutral assumption for unlisted pairings; not a scientific food interaction.
    return sum(values)/len(values) if values else 0.5


def coverage(recipe: Mapping, inventory: Mapping[str, float]) -> float:
    required = recipe['ingredients']
    return sum(min(max(0,inventory.get(k,0)),v) for k,v in required.items())/sum(required.values())


def nutrition_suitability(nutri: Mapping|None, goal: str) -> float:
    if not nutri: return 0.0
    protein = nutri['protein_g']
    energy = nutri['kcal']
    if goal == '고단백':
        return min(protein/30,1)*0.8 + min(energy/400,1)*0.2
    if goal == '재료 소진':
        return 0.5
    # illustrative meal-level targets, not personalized daily recommendations
    kcal_closeness = max(0, 1-abs(energy-550)/550)
    protein_score = min(protein/25,1)
    fiber_score = min(nutri['fiber_g']/7,1)
    return 0.45*kcal_closeness + 0.35*protein_score + 0.20*fiber_score


def recipe_allergens(recipe: Mapping) -> list[str]:
    found = set()
    for food in recipe['ingredients']:
        found.update(ALLERGENS.get(food,set()))
    return sorted(found)


def rank_recipes(inventory: Mapping[str,float], goal: str, time_limit: int,
                 soon: set[str]|None=None, excluded: set[str]|None=None,
                 allergen_exclusions: set[str]|None=None, nutrients: Mapping|None=None,
                 top_k: int=3, weights: Mapping[str,float]|None=None, recipes: list[dict]|None=None) -> list[dict]:
    nutrient_db = nutrients if nutrients is not None else load_demo_nutrients()
    compat = load_compatibility()
    w = effective_weights(weights)
    soon = soon or set()
    excluded = excluded or set()
    allergen_exclusions = allergen_exclusions or set()
    results=[]
    for recipe in (load_recipes() if recipes is None else recipes):
        foods = recipe['ingredients']
        if set(foods) & excluded:
            continue
        if set(recipe_allergens(recipe)) & allergen_exclusions:
            continue
        nutrients_sum, missing_data = recipe_nutrition(recipe, nutrient_db)
        fraction = coverage(recipe, inventory)
        ingredient_score = w['재료 활용'] * fraction
        nutr_score = w['영양 목표'] * nutrition_suitability(nutrients_sum,goal)
        soon_used = len(soon & set(foods))
        urgent_score = w['우선 소진'] * (soon_used/len(soon)) if soon else 0.5*w['우선 소진']
        time_score = w['조리 시간'] * (1 if recipe['minutes'] <= time_limit else
                                      max(0,1 - (recipe['minutes']-time_limit)/10))
        pair_score = w['맛 조합'] * compatibility_score(recipe,compat)
        total = ingredient_score+nutr_score+urgent_score+time_score+pair_score
        missing_quantity={food:round(max(0,grams-inventory.get(food,0)),1) for food,grams in foods.items() if grams > inventory.get(food,0)}
        breakdown={"재료 활용":round(ingredient_score,1),"영양 목표":round(nutr_score,1),"우선 소진":round(urgent_score,1),"조리 시간":round(time_score,1),"맛 조합":round(pair_score,1)}
        results.append({**recipe,"score":round(total,1),"breakdown":breakdown,
                        "coverage":round(fraction*100,1),"missing":missing_quantity,
                        "nutrients":nutrients_sum,"missing_nutrition":missing_data,
                        "allergens":recipe_allergens(recipe)})
    return sorted(results,key=lambda r:(r['score'],r['coverage']),reverse=True)[:top_k]


def radar_target_ratio(nutri: Mapping, goal: str) -> dict:
    # Radar = fraction of illustrative per-meal targets, NOT a health score.
    targets={"열량":550,"단백질":25 if goal != '고단백' else 30,
             "탄수화물":65,"지방":20,"식이섬유":7}
    source={"열량":"kcal","단백질":"protein_g","탄수화물":"carbs_g","지방":"fat_g","식이섬유":"fiber_g"}
    return {k:round(min(100,max(0,nutri[source[k]]/v*100)),1) for k,v in targets.items()}


def normalize_official_csv(df):
    """Convert public-data-style 100g rows to internal schema; return values plus skipped count.

    The source must use 100g basis. All other bases are skipped to prevent silent miscalculation.
    """
    mapping={
       'food':['food','식품명'], 'kcal':['kcal','에너지(kcal)','에너지'],
       'carbs_g':['carbs_g','탄수화물(g)','탄수화물'],
       'protein_g':['protein_g','단백질(g)','단백질'],
       'fat_g':['fat_g','지방(g)','지방'],
       'fiber_g':['fiber_g','식이섬유(g)','식이섬유'],
       'sodium_mg':['sodium_mg','나트륨(mg)','나트륨']}
    cols={key:next((c for c in variants if c in df.columns),None) for key,variants in mapping.items()}
    absent=[k for k,v in cols.items() if v is None]
    if absent: raise ValueError(f"필수 열 없음: {', '.join(absent)}")
    basis=next((c for c in ('영양성분함량기준량','기준량','basis') if c in df.columns),None)
    if basis is None and 'food' not in df.columns:
        raise ValueError('공공데이터에는 영양성분함량기준량 또는 기준량 열이 필요해요. 100g 기준값만 사용할 수 있어요.')
    import pandas as pd
    result={}; skipped=0
    for _,row in df.iterrows():
        if basis and str(row[basis]).replace(' ','').lower() not in ('100g','100그램'):
            skipped+=1;continue
        food=normalize_food(str(row[cols['food']]))
        if not food or food=='nan': skipped+=1;continue
        values={}
        for k in FIELDS:
            try:
                values[k]=float(str(row[cols[k]]).replace(',',''))
                if not math.isfinite(values[k]) or values[k]<0: raise ValueError()
            except (ValueError,TypeError):
                values={};break
        if not values: skipped+=1;continue
        # Keep first match. Duplicates may refer to different foods/processing conditions;
        # users should select precise matching rows manually.
        result.setdefault(food,values)
    return result,skipped
