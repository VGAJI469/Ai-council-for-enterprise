#!/usr/bin/env python
"""Quick validation test to ensure all core components work."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test imports
try:
    from agents.evolution.agent_factory import AgentFactory
    from council.voting.weighted_aggregator import WeightedAggregator
    from council.debate.council_session import CouncilSession
    from agents.base.llm_client import LocalLLMClient
    print("[OK] All imports successful")
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    sys.exit(1)

# Test factory initialization
try:
    import yaml
    with open('config/agents.yaml') as f:
        cfg = yaml.safe_load(f)
    print("[OK] Config loaded")
except Exception as e:
    print(f"[ERROR] Config error: {e}")
    sys.exit(1)

# Test agent factory
try:
    factory = AgentFactory()
    agents = factory.build_council(cfg)
    print(f"[OK] Built council with {len(agents)} agents")
    for agent in agents:
        print(f"  - {agent.role} (model: {agent.preferred_model})")
except Exception as e:
    print(f"[ERROR] Agent factory error: {e}")
    sys.exit(1)

# Test aggregator
try:
    agg = WeightedAggregator()
    print("[OK] Weighted aggregator initialized")
except Exception as e:
    print(f"[ERROR] Aggregator error: {e}")
    sys.exit(1)

# Test session creation
try:
    session = CouncilSession("test-session-1", agents)
    print(f"[OK] Created council session: {session.session_id}")
except Exception as e:
    print(f"[ERROR] Session error: {e}")
    sys.exit(1)

print("\n[OK] All core systems validated successfully!")
print("\nYou can now run: python council/debate/boardroom_debate.py")
