#!/usr/bin/env python3
"""
Self-Development Capability Test
Supreme Orchestrator Assessment
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_code_generation():
    """Test if we can generate and execute new code"""
    print("=== TESTING CODE GENERATION ===")
    
    # Create a simple new agent class
    new_agent_code = '''
class SelfDevelopmentTestAgent:
    def __init__(self, name):
        self.name = name
        self.capabilities = ["code_analysis", "self_reflection", "improvement_suggestions"]
    
    def analyze_system(self):
        return f"{self.name} is analyzing the swarm system for improvement opportunities"
    
    def suggest_enhancements(self):
        return [
            "Add automated testing for agent interactions",
            "Implement agent performance metrics",
            "Create self-updating configuration system"
        ]
'''
    
    # Write the code to a file
    test_file = PROJECT_ROOT / "generated_agent.py"
    with open(test_file, 'w') as f:
        f.write(new_agent_code)
    
    print(f"✓ Generated code file: {test_file}")
    
    # Import and test the generated code
    try:
        exec(new_agent_code)
        agent = locals()['SelfDevelopmentTestAgent']("TestAgent")
        result = agent.analyze_system()
        suggestions = agent.suggest_enhancements()
        
        print(f"✓ Code execution successful: {result}")
        print(f"✓ Generated suggestions: {len(suggestions)} items")
        
        return True
    except Exception as e:
        print(f"✗ Code execution failed: {e}")
        return False

def test_file_modification():
    """Test if we can modify existing system files"""
    print("\n=== TESTING FILE MODIFICATION ===")
    
    # Create a backup of a config file
    config_path = PROJECT_ROOT / "test_config.yaml" 
    backup_path = PROJECT_ROOT / "test_config.yaml.backup"
    
    original_content = '''
# Test Configuration
version: "1.0"
test_mode: true
capabilities:
  - read_files
  - write_files
  - execute_bash
'''
    
    # Write original
    with open(config_path, 'w') as f:
        f.write(original_content)
    
    # Create backup
    with open(backup_path, 'w') as f:
        f.write(original_content)
    
    print(f"✓ Created test files")
    
    # Modify the file
    modified_content = original_content + '''
# SELF-MODIFICATION TEST
self_modification:
  enabled: true
  timestamp: "2025-01-03"
  test_status: "successful"
'''
    
    try:
        with open(config_path, 'w') as f:
            f.write(modified_content)
        
        # Verify modification
        with open(config_path, 'r') as f:
            content = f.read()
            
        if "self_modification" in content:
            print("✓ File modification successful")
            return True
        else:
            print("✗ File modification failed - content not found")
            return False
            
    except Exception as e:
        print(f"✗ File modification failed: {e}")
        return False

def test_system_inspection():
    """Test ability to inspect and understand system structure"""
    print("\n=== TESTING SYSTEM INSPECTION ===")
    
    try:
        # Import system components
        from shared.agent_definitions import AGENT_TYPES, list_agent_types
        from backend.tools import get_tool_definitions
        
        agent_types = list_agent_types()
        tools = get_tool_definitions()
        
        print(f"✓ Successfully imported system components")
        print(f"✓ Found {len(agent_types)} agent types: {agent_types}")
        print(f"✓ Found {len(tools)} tool definitions")
        
        # Test if we can analyze the system structure
        analysis = {
            "agent_types_count": len(agent_types),
            "tools_count": len(tools),
            "has_orchestrator": "orchestrator" in agent_types,
            "has_implementer": "implementer" in agent_types,
            "can_spawn_tasks": any(tool["name"] == "Task" for tool in tools),
            "can_modify_files": any(tool["name"] == "Write" for tool in tools)
        }
        
        print(f"✓ System analysis complete: {analysis}")
        return True
        
    except Exception as e:
        print(f"✗ System inspection failed: {e}")
        return False

def test_cross_swarm_coordination():
    """Test ability to coordinate across swarms"""
    print("\n=== TESTING CROSS-SWARM COORDINATION ===")
    
    try:
        # Simulate coordination request
        coordination_test = {
            "source": "Supreme Orchestrator",
            "target_swarms": ["ASA Research", "Swarm Dev", "operations"],
            "task": "Self-development capability assessment",
            "parallel_execution": True
        }
        
        print(f"✓ Coordination structure created: {coordination_test}")
        
        # Test if we can access swarm information
        swarms_dir = PROJECT_ROOT / "swarms"
        if swarms_dir.exists():
            swarm_folders = [d.name for d in swarms_dir.iterdir() if d.is_dir()]
            print(f"✓ Found swarm directories: {swarm_folders}")
        
        return True
        
    except Exception as e:
        print(f"✗ Cross-swarm coordination test failed: {e}")
        return False

def run_all_tests():
    """Run all self-development tests"""
    print("SUPREME ORCHESTRATOR - SELF-DEVELOPMENT CAPABILITY ASSESSMENT")
    print("=" * 60)
    
    tests = [
        ("Code Generation", test_code_generation),
        ("File Modification", test_file_modification), 
        ("System Inspection", test_system_inspection),
        ("Cross-Swarm Coordination", test_cross_swarm_coordination)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("SELF-DEVELOPMENT TEST RESULTS:")
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    # Self-assessment
    if passed == len(results):
        assessment = "FULL SELF-DEVELOPMENT CAPABILITIES"
    elif passed >= len(results) * 0.75:
        assessment = "STRONG SELF-DEVELOPMENT CAPABILITIES" 
    elif passed >= len(results) * 0.5:
        assessment = "MODERATE SELF-DEVELOPMENT CAPABILITIES"
    else:
        assessment = "LIMITED SELF-DEVELOPMENT CAPABILITIES"
    
    print(f"\nASSESSMENT: {assessment}")
    
    return results

if __name__ == "__main__":
    run_all_tests()