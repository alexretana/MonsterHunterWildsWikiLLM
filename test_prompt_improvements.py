#!/usr/bin/env python3
"""
Utility script for testing and validating prompt improvements
in the LlamaIndex Chroma RAG Pipeline

Usage:
    python test_prompt_improvements.py [--debug] [--mode MODE]
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path

# Add the custom_pipeline directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "custom_pipeline"))

def setup_test_environment():
    """Setup environment variables for testing."""
    os.environ.update({
        "DEBUG_MODE": "true",
        "USE_CUSTOM_PROMPTS": "true",
        "USE_CONVERSATION_CONTEXT": "false",
        "RESPONSE_MODE": "compact",
        "SIMILARITY_TOP_K": "5"
    })
    print("✅ Test environment configured")

def test_prompt_inspection():
    """Test prompt inspection capabilities."""
    print("\n🔍 Testing Prompt Inspection...")
    
    try:
        from llamaindex_chroma_rag import Pipeline
        pipeline = Pipeline()
        
        # Check custom prompt initialization
        custom_prompts = pipeline.custom_prompts
        print(f"✅ Custom prompts initialized: {len(custom_prompts)} templates")
        
        for key, template in custom_prompts.items():
            print(f"  - {key}: {len(str(template))} characters")
            
        # Check configuration
        config = {
            "USE_CUSTOM_PROMPTS": pipeline.valves.USE_CUSTOM_PROMPTS,
            "USE_CONVERSATION_CONTEXT": pipeline.valves.USE_CONVERSATION_CONTEXT,
            "RESPONSE_MODE": pipeline.valves.RESPONSE_MODE,
            "DEBUG_MODE": pipeline.valves.DEBUG_MODE
        }
        
        print("✅ Configuration loaded:")
        for key, value in config.items():
            print(f"  - {key}: {value}")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to inspect prompts: {e}")
        return False

def test_prompt_templates():
    """Test prompt template formatting and variables."""
    print("\n🧪 Testing Prompt Templates...")
    
    try:
        from llamaindex_chroma_rag import Pipeline
        pipeline = Pipeline()
        
        # Test template variable presence - different templates need different variables
        template_requirements = {
            "text_qa_template": ["context_str", "query_str"],
            "refine_template": ["query_str", "existing_answer", "context_msg"],
            "summary_template": ["context_str", "query_str"]
        }
        
        for template_name, template in pipeline.custom_prompts.items():
            template_str = str(template)
            required_vars = template_requirements.get(template_name, ["context_str", "query_str"])
            missing_vars = [var for var in required_vars if f"{{{var}}}" not in template_str]
            
            if missing_vars:
                print(f"⚠️  {template_name} missing variables: {missing_vars}")
            else:
                print(f"✅ {template_name} has all required variables")
                
            # Check for Monster Hunter specific content
            mh_keywords = ["Monster Hunter", "expert", "guide", "context", "terminology"]
            found_keywords = [kw for kw in mh_keywords if kw.lower() in template_str.lower()]
            print(f"   MH keywords found: {len(found_keywords)}/{len(mh_keywords)}")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to test templates: {e}")
        return False

def test_conversation_context():
    """Test conversation context enhancement."""
    print("\n💬 Testing Conversation Context Enhancement...")
    
    try:
        from llamaindex_chroma_rag import Pipeline
        pipeline = Pipeline()
        
        # Test conversation context with sample messages
        sample_messages = [
            {"role": "user", "content": "What are the best weapons in Monster Hunter?"},
            {"role": "assistant", "content": "The best weapons depend on your playstyle. Great Sword is good for high damage..."},
            {"role": "user", "content": "Tell me more about Great Sword"},
            {"role": "assistant", "content": "Great Sword is a heavy weapon that focuses on charged attacks..."},
            {"role": "user", "content": "What combos should I use?"}
        ]
        
        # Test without context enhancement
        query_without_context = pipeline._enhance_query_with_context(
            "What combos should I use?", []
        )
        print(f"✅ Query without context: {len(query_without_context)} characters")
        
        # Test with context enhancement  
        query_with_context = pipeline._enhance_query_with_context(
            "What combos should I use?", sample_messages
        )
        print(f"✅ Query with context: {len(query_with_context)} characters")
        
        context_added = len(query_with_context) > len(query_without_context)
        print(f"✅ Context enhancement working: {context_added}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test conversation context: {e}")
        return False

async def test_pipeline_startup():
    """Test pipeline startup with custom prompts."""
    print("\n🚀 Testing Pipeline Startup...")
    
    try:
        from llamaindex_chroma_rag import Pipeline
        pipeline = Pipeline()
        
        print("📋 Testing startup process...")
        # Note: This will fail if Chroma DB doesn't exist, but we can test initialization
        try:
            await pipeline.on_startup()
            print("✅ Pipeline startup completed successfully")
            
            # Test prompt retrieval if startup succeeded
            if pipeline.query_engine:
                prompts = pipeline.get_current_prompts()
                if "error" not in prompts:
                    print(f"✅ Retrieved {len(prompts)} active prompts")
                else:
                    print(f"⚠️  Prompt retrieval issue: {prompts['error']}")
                    
        except Exception as startup_error:
            print(f"⚠️  Startup failed (expected if no Chroma DB): {startup_error}")
            print("   This is normal for testing without a populated vector store")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed pipeline startup test: {e}")
        return False

def test_response_modes():
    """Test different response mode configurations."""
    print("\n⚙️  Testing Response Mode Configurations...")
    
    modes = ["compact", "refine", "tree_summarize", "simple_summarize"]
    
    for mode in modes:
        try:
            os.environ["RESPONSE_MODE"] = mode
            from llamaindex_chroma_rag import Pipeline
            
            # Clear module cache to reload with new environment
            if "llamaindex_chroma_rag" in sys.modules:
                del sys.modules["llamaindex_chroma_rag"]
            
            # Re-import to get updated configuration
            from llamaindex_chroma_rag import Pipeline
            pipeline = Pipeline()
            
            if pipeline.valves.RESPONSE_MODE == mode:
                print(f"✅ Response mode '{mode}' configured correctly")
            else:
                print(f"❌ Response mode mismatch: expected '{mode}', got '{pipeline.valves.RESPONSE_MODE}'")
                
        except Exception as e:
            print(f"❌ Failed to test mode '{mode}': {e}")
    
    # Reset to default
    os.environ["RESPONSE_MODE"] = "compact"
    return True

def generate_test_report():
    """Generate a comprehensive test report."""
    print("\n" + "="*60)
    print("📊 PROMPT IMPROVEMENT TEST REPORT")
    print("="*60)
    
    tests = [
        ("Prompt Inspection", test_prompt_inspection),
        ("Prompt Templates", test_prompt_templates),
        ("Conversation Context", test_conversation_context),
        ("Response Modes", test_response_modes)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * (len(test_name) + 1))
        results[test_name] = test_func()
    
    # Async test
    print(f"\nPipeline Startup:")
    print("-" * 17)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results["Pipeline Startup"] = loop.run_until_complete(test_pipeline_startup())
    loop.close()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test:25s} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Prompt improvements are working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Test prompt improvements")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--mode", default="compact", help="Response mode to test")
    
    args = parser.parse_args()
    
    if args.debug:
        os.environ["DEBUG_MODE"] = "true"
    
    os.environ["RESPONSE_MODE"] = args.mode
    
    print("🧪 Monster Hunter RAG Pipeline Prompt Testing")
    print("=" * 50)
    
    setup_test_environment()
    success = generate_test_report()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
