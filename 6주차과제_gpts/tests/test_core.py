import pandas as pd
import pytest
from chef_core import (load_demo_nutrients, load_recipes, parse_inventory, recipe_nutrition,
                       rank_recipes, radar_target_ratio, normalize_official_csv)


def test_data_exists_and_recipe_coverage():
    db=load_demo_nutrients()
    assert len(db) >= 20
    for r in load_recipes():
        nutrition,missing=recipe_nutrition(r,db)
        assert not missing
        assert nutrition['kcal'] > 0
        assert nutrition['protein_g'] >= 0


def test_korean_amount_parsing():
    inv,w=parse_inventory('계란 2개, 두부 반 모, 시금치 100g',load_demo_nutrients())
    assert inv['계란']==100
    assert inv['두부']==150
    assert inv['시금치']==100
    assert w


def test_unknown_inventory_does_not_hallucinate():
    inv,warning=parse_inventory('초코 로봇 2개',load_demo_nutrients())
    assert not inv and warning


def test_ranked_score_and_constraints():
    inv={'두부':200,'계란':100,'양파':100,'시금치':100,'식용유':10}
    results=rank_recipes(inv,'고단백',25,top_k=20)
    assert results and all(0<=r['score']<=100 for r in results)
    assert results==sorted(results,key=lambda x:x['score'],reverse=True)
    assert results[0]['name']
    assert all(r['nutrients'] is not None for r in results)
    for r in rank_recipes(inv,'균형',25,excluded={'두부'},top_k=20):
        assert '두부' not in r['ingredients']
    for r in rank_recipes(inv,'균형',25,allergen_exclusions={'난류'},top_k=20):
        assert '계란' not in r['ingredients']


def test_missing_db_not_estimated():
    db=load_demo_nutrients()
    del db['계란']
    egg=next(r for r in load_recipes() if '계란' in r['ingredients'])
    val,missing=recipe_nutrition(egg,db)
    assert val is None and missing==['계란']


def test_radar_scale():
    r=radar_target_ratio({'kcal':600,'protein_g':50,'carbs_g':70,'fat_g':30,'fiber_g':15},'고단백')
    assert all(v==100 for v in r.values())


def test_upload_public_column_mapping_and_100g_filter():
    df=pd.DataFrame([{'식품명':'계란','영양성분함량기준량':'100g','에너지(kcal)':143,'탄수화물(g)':0.7,
                      '단백질(g)':12.6,'지방(g)':9.5,'식이섬유(g)':0,'나트륨(mg)':142},
                     {'식품명':'우유','영양성분함량기준량':'200ml','에너지(kcal)':120,'탄수화물(g)':10,
                      '단백질(g)':6,'지방(g)':7,'식이섬유(g)':0,'나트륨(mg)':80}])
    parsed,skipped=normalize_official_csv(df)
    assert set(parsed)=={'계란'} and skipped==1


def test_score_changes_with_stock():
    a=rank_recipes({'두부':150,'계란':100,'양파':50,'시금치':60,'식용유':5},'균형',25,top_k=12)
    b=rank_recipes({'바나나':100,'플레인요거트':170,'아몬드':15},'균형',25,top_k=12)
    get=lambda arr,name:next(r['coverage'] for r in arr if r['name']==name)
    assert get(a,'두부 달걀 스크램블') > get(b,'두부 달걀 스크램블')


def test_weight_customization_affects_scores():
    inv={'두부':150,'계란':100,'양파':50,'시금치':60,'식용유':5}
    base=rank_recipes(inv,'균형',20,top_k=12)
    custom=rank_recipes(inv,'균형',20,top_k=12,weights={'우선 소진':0,'재료 활용':50})
    score=lambda ranked,name:next(r['score'] for r in ranked if r['name']==name)
    assert score(base,'두부 달걀 스크램블') != score(custom,'두부 달걀 스크램블')
