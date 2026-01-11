#!/usr/bin/env python3
"""
Direct runner for expert analysis from intellirefactor directory.

Usage:
    python run_expert_analysis_here.py <project_path> <target_file> [--detailed]
"""

import sys
import os
from pathlib import Path

# Add project root to Python path (parent directory)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Run expert analysis directly."""
    if len(sys.argv) < 3:
        print("Usage: python run_expert_analysis_here.py <project_path> <target_file> [--detailed]")
        print("Example: python run_expert_analysis_here.py C:\\Intel\\recon C:\\Intel\\recon\\core\\bypass\\engine\\attack_dispatcher.py --detailed")
        sys.exit(1)
    
    project_path = sys.argv[1]
    target_file = sys.argv[2]
    detailed = "--detailed" in sys.argv
    
    try:
        from intellirefactor.analysis.expert.expert_analyzer import ExpertRefactoringAnalyzer
        
        # Simple output since rich might not be available
        print("🔍 Starting expert refactoring analysis...")
        print(f"📁 Project: {project_path}")
        print(f"🎯 Target: {target_file}")
        print(f"🔬 Detailed mode: {'ON' if detailed else 'OFF'}")
        
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
        output_dir = project_path_obj / "expert_analysis_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📊 Output directory: {output_dir}")
        
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
            
            # Show expert recommendations summary
            if 'expert_recommendations' in detailed_data:
                expert_recs = detailed_data['expert_recommendations']
                print("🎯 Expert Requirements Status:")
                
                if 'expert_1_requirements' in expert_recs:
                    print(f"  📋 Expert 1: {len(expert_recs['expert_1_requirements'])} requirements addressed")
                
                if 'expert_2_requirements' in expert_recs:
                    print(f"  📋 Expert 2: {len(expert_recs['expert_2_requirements'])} requirements addressed")
        
        if result.recommendations:
            print("💡 Key Recommendations:")
            for i, rec in enumerate(result.recommendations[:3], 1):
                print(f"  {i}. {rec}")
        
        print("📄 Generated Reports:")
        for report in reports_generated:
            print(f"  • {report}")
        
        print("🎉 Analysis complete! Check the generated reports for detailed insights.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed.")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"❌ Expert analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()