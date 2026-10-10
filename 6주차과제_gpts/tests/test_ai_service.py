"""Offline tests mock Responses API: no paid calls, no secrets needed."""
import json
import pytest
from ai_service import (MODEL, analyze_food_label, create_recipe_candidates,
                        recognize_fridge_photo, validate_ai_recipes, RECIPE_SCHEMA)
import ai_service
from chef_core import ALLERGENS, load_demo_nutrients, rank_recipes

class FakeResponses:
    def __init__(self, obj):
        self.obj = obj
        self.kwargs=None
    def create(self,**kwargs):
        self.kwargs=kwargs
        return type('Response',(),{'output_text':json.dumps(self.obj,ensure_ascii=False)})()

class FakeClient:
    def __init__(self, response):
        self.responses=response


def stub(monkeypatch,payload):
    fake=FakeResponses(payload)
    monkeypatch.setattr(ai_service,'OpenAI',lambda **kwargs:FakeClient(fake))
    return fake


def sample(name='계란 시금치 볶음', food1='계란', food2='시금치'):
    return {'name':name,'minutes':12,'difficulty':'쉬움', 'rationale':'가진 재료 활용',
            'ingredients':[{'food':food1,'grams':100},{'food':food2,'grams':50}],
            'steps':['재료를 손질한다.','충분히 익힌다.']}


def test_model_is_luna_and_structured_recipe_request(monkeypatch):
    fake=stub(monkeypatch,{'recipes':[sample()]})
    raw=create_recipe_candidates('sk-test',{'계란':100,'시금치':90},'고단백',20,[],[],[],['계란','시금치'])
    assert raw['recipes'][0]['name']=='계란 시금치 볶음'
    assert fake.kwargs['model']=='gpt-6-luna'
    assert fake.kwargs['store'] is False
    assert fake.kwargs['text']['format']['strict'] is True
    assert '계란' in fake.kwargs['input']


def test_generated_recipe_is_scored_by_python():
    payload={'recipes':[sample()]}
    candidate,alerts=validate_ai_recipes(payload,set(load_demo_nutrients()),set(),set(),ALLERGENS)
    assert len(candidate)==1 and not alerts
    ranked=rank_recipes({'계란':100,'시금치':90},'균형',20,recipes=candidate)
    assert len(ranked)==1 and isinstance(ranked[0]['score'],float)
    assert ranked[0]['coverage']==100


def test_safety_filter_excludes_invalid_and_allergen_recipe():
    payload={'recipes':[sample('A','계란','시금치'),sample('B','없는 재료','양파'),sample('C','두부','계란')]}
    accepted,notices=validate_ai_recipes(payload,set(load_demo_nutrients()),{'두부'},{'난류'},ALLERGENS)
    assert not accepted and len(notices)==3


def test_photo_label_uses_luna_vision_and_nullable_unknown(monkeypatch):
    record={'product_name':'샘플 두부','basis':'100g', 'energy_kcal':None,
            'carbohydrate_g':None,'protein_g':None,'fat_g':None,
            'fiber_g':None,'sugar_g':None,'sodium_mg':None,'saturated_fat_g':None,
            'ingredients_text':'대두, 물','allergen_text':'대두 함유',
            'uncertainties':['사진 흐림']}
    fake=stub(monkeypatch,record)
    got=analyze_food_label('sk-test',b'fakebytes','image/png')
    assert got['energy_kcal'] is None
    assert fake.kwargs['model']==MODEL
    assert fake.kwargs['input'][0]['content'][1]['type']=='input_image'
    assert fake.kwargs['input'][0]['content'][1]['image_url'].startswith('data:image/png;base64,')


def test_fridge_recognition_uses_image(monkeypatch):
    fake=stub(monkeypatch,{'items':[{'food':'계란','quantity_text':'','uncertain':False}], 'notes':[]})
    got=recognize_fridge_photo('sk-test',b'imagedata','image/jpeg')
    assert got['items'][0]['food']=='계란'
    assert fake.kwargs['input'][0]['content'][1]['type']=='input_image'


def test_invalid_image_rejected():
    with pytest.raises(ValueError):
        analyze_food_label('sk-test',b'x','text/plain')
    with pytest.raises(ValueError):
        analyze_food_label('sk-test',b'x'*(8*1024*1024+1),'image/png')


def test_api_key_required():
    with pytest.raises(ValueError):
        create_recipe_candidates('',{},'균형',20,[],[],[],[])
