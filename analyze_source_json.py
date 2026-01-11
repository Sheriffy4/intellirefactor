#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ исходного JSON файла для улучшения дистиллятора
"""

import json

def analyze_json():
    # Загружаем исходный JSON
    with open('expert_analysis_output/expert_analysis_detailed_20260109_132347.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=== СТРУКТУРА ИСХОДНОГО JSON ===")
    print("Top-level keys:")
    for key in data.keys():
        print(f"  - {key}")
    
    print("\n=== GOLDEN TRACES ANALYSIS ===")
    golden_traces = data.get('golden_traces', {})
    print(f"Golden traces keys: {list(golden_traces.keys())}")
    
    # Анализ real_usage_scenarios
    scenarios = golden_traces.get('real_usage_scenarios', {})
    print(f"\nReal usage scenarios methods: {len(scenarios)}")
    for method_name, calls in list(scenarios.items())[:5]:
        call_count = len(calls) if isinstance(calls, list) else 0
        print(f"  {method_name}: {call_count} calls")
    
    # Пример dispatch_attack вызова
    dispatch_calls = scenarios.get('self.dispatch_attack', [])
    if dispatch_calls:
        print(f"\nFirst dispatch_attack call:")
        first_call = dispatch_calls[0]
        for key, value in first_call.items():
            print(f"  {key}: {value}")
    
    # Анализ realistic_test_data
    print("\n=== REALISTIC TEST DATA ===")
    test_data = golden_traces.get('realistic_test_data', {})
    print(f"Test data keys: {list(test_data.keys())}")
    
    task_types = test_data.get('task_types', [])
    print(f"Task types ({len(task_types)}):")
    if isinstance(task_types, list):
        for tt in task_types[:5]:
            print(f"  - {tt}")
    else:
        print(f"  Type: {type(task_types)}, Value: {task_types}")
    
    # Анализ параметров
    params = test_data.get('parameters', {})
    print(f"\nParameters ({len(params) if isinstance(params, dict) else 'N/A'}):")
    if isinstance(params, dict):
        for key in list(params.keys())[:5]:
            value = params[key]
            print(f"  {key}: {str(value)[:100]}...")
    
    # Анализ packet_info
    packet_info = test_data.get('packet_info', {})
    print(f"\nPacket info ({len(packet_info) if isinstance(packet_info, dict) else 'N/A'}):")
    if isinstance(packet_info, dict):
        for key in list(packet_info.keys())[:5]:
            value = packet_info[key]
            print(f"  {key}: {str(value)[:100]}...")
    
    # Поиск результатов/сегментов
    print("\n=== RESULTS/SEGMENTS SEARCH ===")
    # Ищем в разных местах
    search_keys = ['results', 'outputs', 'segments', 'attack_results']
    found_sections = []
    
    for key in search_keys:
        if key in data:
            found_sections.append(key)
            section = data[key]
            print(f"Found '{key}' section: {type(section)} with {len(section) if hasattr(section, '__len__') else 'N/A'} items")
            
            if isinstance(section, dict) and section:
                first_key = list(section.keys())[0]
                print(f"  First item key: {first_key}")
                first_item = section[first_key]
                print(f"  First item type: {type(first_item)}")
                if hasattr(first_item, '__len__'):
                    print(f"  First item length: {len(first_item)}")
    
    # Анализ дубликатов на предмет определений типов
    print("\n=== TYPE DEFINITIONS IN DUPLICATES ===")
    duplicates = data.get('duplicates', {}).get('duplicates', [])
    type_defs = []
    
    for dup in duplicates[:100]:  # Проверяем первые 100 дубликатов
        if isinstance(dup, dict):
            content = dup.get('content_preview', '')
            if 'AttackRecipe' in content or 'SegmentTuple' in content or 'TypeAlias' in content:
                type_defs.append(content[:200])
    
    print(f"Found {len(type_defs)} potential type definitions")
    for i, td in enumerate(type_defs[:3]):
        print(f"{i+1}. {td}")

if __name__ == '__main__':
    analyze_json()
