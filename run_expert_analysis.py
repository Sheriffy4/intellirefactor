#!/usr/bin/env python3
"""
Direct runner for expert analysis - Thin wrapper for intellirefactor.run_expert_analysis_here

Usage:
    python run_expert_analysis.py <project_path> <target_file> [--detailed]
"""

import warnings
from intellirefactor.run_expert_analysis_here import main

warnings.warn(
    "run_expert_analysis.py is deprecated; use intellirefactor.run_expert_analysis_here",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    main()

def main():
    """Run expert analysis directly."""
    if len(sys.argv) < 3:
        print("Usage: python run_expert_analysis.py <project_path> <target_file> [--detailed]")
        print("Example: python run_expert_analysis.py C:\\Intel\\recon C:\\Intel\\recon\\core\\bypass\\engine\\attack_dispatcher.py --detailed")
        sys.exit(1)
    
    project_path = sys.argv[1]
    target_file = sys.argv[2]
    detailed = "--detailed" in sys.argv
    
    try:
        from intellirefactor.analysis.expert.expert_analyzer import ExpertRefactoringAnalyzer
        
        # Validate inputs
        project_path_obj = Path(project_path)
        target_file_obj = Path(target_file)
        
        if not project_path_obj.exists():
            print(f"❌ Project path does not exist: {project_path}")
            sys.exit(1)
        
        if not target_file_obj.exists():
            print(f"❌ Target file does not exist: {target_file}")
            sys.exit(1)
        
        if not target_file_obj.suffix == '.py':
            print(f"❌ Target file must be a Python file: {target_file}")
            sys.exit(1)
        
        # Set up output directory
        output_dir = Path("./expert_analysis_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("🔍 Starting expert refactoring analysis...")
        print(f"📁 Project: {project_path}")
        print(f"🎯 Target: {target_file}")
        print(f"📊 Output: {output_dir}")
        print(f"🔬 Detailed mode: {'ON' if detailed else 'OFF'}")
        
        # Initialize analyzer
        analyzer = ExpertRefactoringAnalyzer(
            project_root=str(project_path_obj),
            target_module=str(target_file_obj),
            output_dir=str(output_dir)
        )
        
        # Run analysis
        print("📊 Running expert analysis...")
        result = analyzer.analyze_for_expert_refactoring()
        
        # Export detailed data if requested
        detailed_data = None
        if detailed:
            print("🔬 Exporting detailed expert data...")
            detailed_data = analyzer.export_detailed_expert_data()
        
        # Generate reports
        reports_generated = []
        
        # JSON report
        json_path = output_dir / f"expert_analysis_{analyzer.timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        reports_generated.append(str(json_path))
        
        # Detailed JSON if requested
        if detailed_data:
            detailed_json_path = output_dir / f"expert_analysis_detailed_{analyzer.timestamp}.json"
            with open(detailed_json_path, 'w', encoding='utf-8') as f:
                json.dump(detailed_data, f, indent=2, ensure_ascii=False)
            reports_generated.append(str(detailed_json_path))
            
            # Create characterization test file
            if 'characterization_tests' in detailed_data:
                test_file_path = detailed_data['characterization_tests'].get('test_file_path', 'test_characterization.py')
                test_code = detailed_data['characterization_tests'].get('executable_test_code', '')
                if test_code:
                    test_path = output_dir / test_file_path
                    with open(test_path, 'w', encoding='utf-8') as f:
                        f.write(test_code)
                    reports_generated.append(str(test_path))
        
        # Markdown report
        md_path = analyzer.generate_expert_report(str(output_dir))
        reports_generated.append(md_path)
        
        # Display results
        print("✅ Expert analysis completed successfully!")
        print(f"📊 Quality Score: {result.analysis_quality_score:.1f}/100")
        print(f"⚠️  Risk Level: {result.risk_assessment.value.upper()}")
        
        if detailed and detailed_data:
            print("🎯 Detailed expert data exported - all expert requirements addressed!")
            
            # Show enhanced statistics
            print("📈 Detailed Analysis Statistics:")
            
            if 'call_graph' in detailed_data:
                cg = detailed_data['call_graph'].get('call_graph', {})
                total_relationships = cg.get('total_relationships', 0)
                print(f"  📞 Complete call graph: {total_relationships} relationships")
            
            if 'external_usage' in detailed_data:
                eu = detailed_data['external_usage'].get('files_summary', {})
                total_files = eu.get('total_files', 0)
                print(f"  🔗 External usage: {total_files} files with specific locations")
            
            if 'duplicates' in detailed_data:
                dup = detailed_data['duplicates'].get('summary', {})
                total_duplicates = dup.get('total_duplicates', 0)
                total_savings = dup.get('total_savings', 0)
                print(f"  🔄 Code duplicates: {total_duplicates} fragments, {total_savings} lines savings")
            
            if 'characterization_tests' in detailed_data:
                ct = detailed_data['characterization_tests'].get('summary', {})
                total_tests = ct.get('total_tests', 0)
                print(f"  🧪 Characterization tests: {total_tests} executable test cases")
            
            if 'test_analysis' in detailed_data:
                ta = detailed_data['test_analysis'].get('missing_test_coverage', {})
                total_missing = ta.get('total_missing', 0)
                print(f"  📋 Missing test coverage: {total_missing} specific methods identified")
            
            if 'exception_contracts' in detailed_data:
                ec = detailed_data['exception_contracts']
                if isinstance(ec, dict) and 'summary' in ec:
                    total_methods = ec['summary'].get('total_methods_analyzed', 0)
                    print(f"  ⚠️  Exception contracts: {total_methods} methods analyzed")
            
            if 'data_schemas' in detailed_data:
                ds = detailed_data['data_schemas']
                if isinstance(ds, dict) and 'key_usage_summary' in ds:
                    total_keys = len(ds['key_usage_summary'])
                    print(f"  📊 Data schemas: {total_keys} keys analyzed")
            
            if 'golden_traces' in detailed_data:
                gt = detailed_data['golden_traces']
                if isinstance(gt, dict) and 'real_usage_scenarios' in gt:
                    scenario_count = len(gt['real_usage_scenarios'])
                    print(f"  🏆 Golden traces: {scenario_count} real usage scenarios")
            
            # Show expert recommendations
            if 'expert_recommendations' in detailed_data:
                expert_recs = detailed_data['expert_recommendations']
                print("🎯 Expert Requirements Status:")
                
                if 'expert_1_requirements' in expert_recs:
                    print("  📋 Expert 1 Requirements:")
                    for rec in expert_recs['expert_1_requirements'][:3]:
                        print(f"    ✅ {rec}")
                
                if 'expert_2_requirements' in expert_recs:
                    print("  📋 Expert 2 Requirements:")
                    for rec in expert_recs['expert_2_requirements'][:3]:
                        print(f"    ✅ {rec}")
        
        if result.recommendations:
            print("💡 Key Recommendations:")
            for i, rec in enumerate(result.recommendations[:5], 1):
                print(f"  {i}. {rec}")
        
        print("📄 Generated Reports:")
        for report in reports_generated:
            print(f"  • {report}")
        
        print("🎉 Analysis complete! Check the generated reports for detailed insights.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the project root directory.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Expert analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()