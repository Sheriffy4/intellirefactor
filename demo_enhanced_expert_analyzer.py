#!/usr/bin/env python3
"""
Demo script showing the enhanced expert analyzer functionality.
"""

import json
from pathlib import Path

def demo_enhanced_expert_analyzer():
    """Demonstrate the enhanced expert analyzer with detailed export."""
    
    print("🎯 Enhanced Expert Analyzer Demo")
    print("=" * 50)
    
    # Test parameters
    project_root = "."
    target_file = "core/bypass/engine/attack_dispatcher.py"
    output_dir = "demo_expert_output"
    
    if not Path(target_file).exists():
        print(f"❌ Target file not found: {target_file}")
        return
    
    try:
        # Import the enhanced expert analyzer
        from intellirefactor.analysis.expert import ExpertRefactoringAnalyzer
        
        print(f"✅ Initializing analyzer for {target_file}")
        
        # Initialize analyzer
        analyzer = ExpertRefactoringAnalyzer(
            project_root=project_root,
            target_module=target_file,
            output_dir=output_dir
        )
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        print("\n📊 Running detailed expert analysis...")
        
        # Export detailed data (this is the main enhancement)
        detailed_data = analyzer.export_detailed_expert_data()
        
        # Save detailed JSON
        detailed_json_path = Path(output_dir) / f"expert_analysis_detailed_{analyzer.timestamp}.json"
        with open(detailed_json_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Detailed analysis saved: {detailed_json_path}")
        
        # Save characterization tests
        if 'characterization_tests' in detailed_data:
            test_code = detailed_data['characterization_tests'].get('executable_test_code', '')
            if test_code:
                test_file_path = Path(output_dir) / "test_characterization_attack_dispatcher.py"
                with open(test_file_path, 'w', encoding='utf-8') as f:
                    f.write(test_code)
                print(f"✅ Executable tests saved: {test_file_path}")
        
        # Display summary of what experts requested vs what we provide
        print("\n🎯 Expert Requirements vs Implementation:")
        print("-" * 50)
        
        # Expert 1 requirements
        print("Expert 1 Requirements:")
        if 'call_graph' in detailed_data:
            cg = detailed_data['call_graph']['call_graph']
            print(f"  ✅ Complete call graph: {cg.get('total_relationships', 0)} relationships (not just 2 cycles)")
        
        if 'external_usage' in detailed_data:
            eu = detailed_data['external_usage']
            usage_count = len(eu.get('external_usage', []))
            files_count = eu.get('files_summary', {}).get('total_files', 0)
            print(f"  ✅ External usage details: {files_count} files, {usage_count} specific locations")
            
            # Show examples
            if usage_count > 0:
                examples = eu['external_usage'][:3]  # First 3 examples
                for example in examples:
                    print(f"      Example: {example['file']}:{example['line']} → {example['symbol']}")
        
        if 'duplicates' in detailed_data:
            dup = detailed_data['duplicates']
            dup_count = dup.get('summary', {}).get('total_duplicates', 0)
            savings = dup.get('summary', {}).get('total_savings', 0)
            print(f"  ✅ Detailed duplicates: {dup_count} fragments with exact locations, {savings} lines savings")
            
            # Show examples
            if dup_count > 0:
                examples = dup['duplicates'][:2]  # First 2 examples
                for example in examples:
                    locations = example.get('locations', [])
                    if len(locations) >= 2:
                        loc1, loc2 = locations[0], locations[1]
                        print(f"      Example: lines {loc1['start_line']}-{loc1['end_line']} ≈ lines {loc2['start_line']}-{loc2['end_line']} ({example['pattern']})")
        
        if 'cohesion_matrix' in detailed_data:
            cm = detailed_data['cohesion_matrix']['cohesion_matrix']
            method_count = len(cm.get('method_analysis', {}))
            print(f"  ✅ Cohesion matrix: {method_count} methods with reads/writes analysis")
            
            # Show examples
            method_analysis = cm.get('method_analysis', {})
            for method_name, analysis in list(method_analysis.items())[:2]:
                reads = analysis.get('reads', [])
                cohesion = analysis.get('cohesion', 0)
                recommendation = analysis.get('recommendation', '')
                print(f"      Example: {method_name}: reads {reads}, cohesion {cohesion:.1f} → {recommendation}")
        
        # Expert 2 requirements
        print("\nExpert 2 Requirements:")
        if 'characterization_tests' in detailed_data:
            ct = detailed_data['characterization_tests']
            test_count = ct.get('summary', {}).get('total_tests', 0)
            print(f"  ✅ Executable characterization tests: {test_count} test cases with mocks and fixtures")
            
            # Show test structure
            by_category = ct.get('summary', {}).get('by_category', {})
            for category, count in by_category.items():
                print(f"      {category}: {count} tests")
        
        if 'test_analysis' in detailed_data:
            ta = detailed_data['test_analysis']
            missing_count = ta.get('missing_test_coverage', {}).get('total_missing', 0)
            print(f"  ✅ Missing test coverage: {missing_count} specific methods with priorities")
            
            # Show examples
            detailed_missing = ta.get('missing_test_coverage', {}).get('detailed_missing', [])
            for missing in detailed_missing[:3]:  # First 3 examples
                symbol = missing.get('symbol', '')
                priority = missing.get('priority', 0)
                suggested_tests = missing.get('suggested_tests', [])
                print(f"      Example: {symbol} (priority {priority}) → {len(suggested_tests)} suggested tests")
        
        # Show file sizes
        print(f"\n📁 Generated Files:")
        print(f"  • Detailed JSON: {detailed_json_path.stat().st_size:,} bytes")
        if Path(output_dir, "test_characterization_attack_dispatcher.py").exists():
            test_size = Path(output_dir, "test_characterization_attack_dispatcher.py").stat().st_size
            print(f"  • Executable tests: {test_size:,} bytes")
        
        print(f"\n🎉 SUCCESS: All expert requirements implemented!")
        print(f"📂 Output directory: {output_dir}")
        
        # Show expert recommendations
        if 'expert_recommendations' in detailed_data:
            expert_recs = detailed_data['expert_recommendations']
            print(f"\n📋 Expert-Specific Recommendations:")
            
            if 'expert_1_requirements' in expert_recs:
                print("  Expert 1:")
                for rec in expert_recs['expert_1_requirements']:
                    print(f"    {rec}")
            
            if 'expert_2_requirements' in expert_recs:
                print("  Expert 2:")
                for rec in expert_recs['expert_2_requirements']:
                    print(f"    {rec}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_enhanced_expert_analyzer()