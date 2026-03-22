"""
Agent Factory - Builds the council from config.
Injects a single shared LocalLLMClient into all agents
to avoid creating N separate Ollama connections.
"""
from agents.base.llm_client import LocalLLMClient
from agents.roles.ceo_agent import CEOAgent
from agents.roles.cfo_agent import CFOAgent
from agents.roles.marketing_agent import MarketingAgent
from agents.roles.pr_agent import PRAgent
from agents.roles.legal_agent import LegalAgent

ROLE_MAP = {
    "strategic_growth":     CEOAgent,
    "financial_stability":  CFOAgent,
    "market_expansion":     MarketingAgent,
    "reputation_risk":      PRAgent,
    "regulatory_compliance":LegalAgent,
}


class AgentFactory:

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.llm_client = LocalLLMClient(base_url=ollama_url)

    def build_council(self, agent_cfg: dict, model=None) -> list:
        """Instantiate all five council agents from config."""
        agents = []
        for role_key, cfg in agent_cfg["agents"].items():
            AgentClass = ROLE_MAP.get(cfg["focus"])
            if AgentClass:
                agent = AgentClass(
                    agent_id=f"{role_key}_v1",
                    config=cfg,
                    model=model,
                    llm_client=self.llm_client,
                )
                agents.append(agent)
        return agents

    def create_from_config(self, agent_id: str, config: dict, model=None):
        """Create a single agent — used by EvolutionController for replacements."""
        AgentClass = ROLE_MAP.get(config["focus"])
        if not AgentClass:
            raise ValueError(f"Unknown agent focus: {config['focus']}")
        return AgentClass(
            agent_id=agent_id,
            config=config,
            model=model,
            llm_client=self.llm_client,
        )
