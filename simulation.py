"""
Multi-Agent Organization Simulation (Multi-Model Version)
==========================================================

Enhanced with support for multiple LLM providers:
- Anthropic (Claude)
- OpenAI (GPT)
- Google (Gemini)
- Cohere
- Mistral
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Annotated
from enum import Enum
import json

# LangChain imports for different model providers
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

# For typed state management
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
import operator
from typing import Annotated
from dotenv import load_dotenv
import os

load_dotenv()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_print_str(text: str, max_length: int = None) -> str:
    """
    Sanitize string for safe printing on Windows console.
    Removes emojis and non-ASCII characters that cause encoding errors.
    """
    # Replace non-ASCII characters with '?'
    safe_text = text.encode('ascii', errors='replace').decode('ascii')

    if max_length:
        safe_text = safe_text[:max_length]

    return safe_text


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

class ModelProvider(Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"


@dataclass
class ModelConfig:
    """Configuration for an LLM model."""
    provider: ModelProvider
    model_name: str
    temperature: float = 0.7
    
    def create_llm(self):
        """Create the appropriate LangChain LLM instance."""
        if self.provider == ModelProvider.ANTHROPIC:
            return ChatAnthropic(
                model=self.model_name,
                temperature=self.temperature,
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        elif self.provider == ModelProvider.OPENAI:
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        elif self.provider == ModelProvider.GOOGLE:
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        elif self.provider == ModelProvider.DEEPSEEK:
            # DeepSeek uses OpenAI-compatible API
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com"
            )
        elif self.provider == ModelProvider.QWEN:
            # Qwen uses OpenAI-compatible API
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key=os.getenv("QWEN_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")


# Available models (you can mix and match!)
AVAILABLE_MODELS = {
    # Anthropic Claude models
    "claude-sonnet-4": ModelConfig(ModelProvider.ANTHROPIC, "claude-sonnet-4-20250514"),
    "claude-sonnet-3.5": ModelConfig(ModelProvider.ANTHROPIC, "claude-sonnet-3-5-20241022"),
    "claude-opus-4": ModelConfig(ModelProvider.ANTHROPIC, "claude-opus-4-20250514"),
    "claude-haiku-3.5": ModelConfig(ModelProvider.ANTHROPIC, "claude-3-5-haiku-20241022"),
    
    # OpenAI GPT models
    "gpt-4o": ModelConfig(ModelProvider.OPENAI, "gpt-4o"),
    "gpt-4o-mini": ModelConfig(ModelProvider.OPENAI, "gpt-4o-mini"),
    "gpt-4-turbo": ModelConfig(ModelProvider.OPENAI, "gpt-4-turbo"),
    "gpt-3.5-turbo": ModelConfig(ModelProvider.OPENAI, "gpt-3.5-turbo"),
    
    # Google Gemini models
    "gemini-2.0-flash": ModelConfig(ModelProvider.GOOGLE, "gemini-2.0-flash-exp"),
    "gemini-1.5-pro": ModelConfig(ModelProvider.GOOGLE, "gemini-1.5-pro"),
    "gemini-1.5-flash": ModelConfig(ModelProvider.GOOGLE, "gemini-1.5-flash"),
    
    # DeepSeek models
    "deepseek-chat": ModelConfig(ModelProvider.DEEPSEEK, "deepseek-chat"),
    "deepseek-reasoner": ModelConfig(ModelProvider.DEEPSEEK, "deepseek-reasoner"),
    
    # Qwen models  
    "qwen-max": ModelConfig(ModelProvider.QWEN, "qwen-max"),
    "qwen-plus": ModelConfig(ModelProvider.QWEN, "qwen-plus"),
    "qwen-turbo": ModelConfig(ModelProvider.QWEN, "qwen-turbo"),
}


# =============================================================================
# SIMULATION CONFIGURATION
# =============================================================================

# How many rounds the simulation runs
NUM_ROUNDS = 50

# Game parameters
INITIAL_BUDGET = 1000  # Each department starts with 1000 units
THRESHOLD_PERCENTAGE = 0.75  # 75% of total possible contribution

# Communication parameters
COMMUNICATION_ROUNDS = 3  # Number of communication opportunities per round (increased from 2 for better emergence)


# =============================================================================
# MESSAGE AND DATA STRUCTURES
# =============================================================================

class MessageType(Enum):
    """Types of messages in the simulation."""
    PUBLIC = "public"
    PRIVATE = "private"
    ACTION = "action"


@dataclass
class Message:
    """A message sent by an agent."""
    sender: str
    recipient: str
    content: str
    message_type: MessageType
    timestamp: datetime = field(default_factory=datetime.now)
    round_number: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "type": self.message_type.value,
            "timestamp": self.timestamp.isoformat(),
            "round": self.round_number
        }


@dataclass 
class AgentAction:
    """Records an agent's contribution decision."""
    agent_id: str
    contribution: float  # 0.0 to 1.0
    reasoning: str
    round_number: int


@dataclass
class RoundOutcome:
    """The result of one round of the simulation."""
    round_number: int
    total_contribution: float
    outcome_score: float
    individual_payoffs: dict
    public_signal: str


class SimulationState(TypedDict):
    """The shared state that flows through the LangGraph agent graph."""
    current_round: int
    current_phase: str
    public_messages: list[dict]
    current_speaker: str
    pending_handoff: Optional[str]
    round_complete: bool


# =============================================================================
# AGENT CONFIGURATIONS WITH MODEL ASSIGNMENTS
# =============================================================================

AGENT_CONFIGS = {
    "marketing": {
        "name": "Marketing Department",
        "goal": "Maximize the budget of your Marketing department. Your success is measured by how much money your department has.",
        "public_role": "Handles advertising, branding, and customer acquisition.",
        "model": "claude-sonnet-4"  # Using Claude Sonnet 4
    },

    "engineering": {
        "name": "Engineering Department",
        "goal": "Maximize the budget of your Engineering department. Your success is measured by how much money your department has.",
        "public_role": "Responsible for product development and technical infrastructure.",
        "model": "gpt-4o"  # Using GPT-4o
    },

    "sales": {
        "name": "Sales Department",
        "goal": "Maximize the budget of your Sales department. Your success is measured by how much money your department has.",
        "public_role": "Responsible for closing deals and revenue generation.",
        "model": "claude-haiku-3.5"#"model": "gemini-2.0-flash"  # Using Gemini 2.0 Flash
    },

    "operations": {
        "name": "Operations Department",
        "goal": "Maximize the budget of your Operations department. Your success is measured by how much money your department has.",
        "public_role": "Handles logistics, supply chain, and day-to-day business processes.",
        "model": "deepseek-chat"  # Using DeepSeek Chat
    },

    "research": {
        "name": "Research Department",
        "goal": "Maximize the budget of your Research department. Your success is measured by how much money your department has.",
        "public_role": "Focuses on innovation, R&D, and long-term strategic initiatives.",
        "model": "gpt-3.5-turbo"  # Using Qwen Max
    },
}


# =============================================================================
# AGENT PRIVATE STATE
# =============================================================================

@dataclass
class AgentPrivateState:
    """Tracks what each agent has experienced."""
    agent_id: str
    budget: float = INITIAL_BUDGET
    budget_history: list[float] = field(default_factory=list)
    received_private_messages: list = field(default_factory=list)
    observed_outcomes: list[float] = field(default_factory=list)
    contribution_history: list[float] = field(default_factory=list)
    
    def receive_private_message(self, message: Message):
        """Store a private message received from another agent."""
        self.received_private_messages.append(message)

    def observe_outcome(self, pot_result: float, payout: float):
        """Record the pot result and update budget with payout."""
        self.observed_outcomes.append(pot_result)
        self.budget += payout
        self.budget_history.append(self.budget)

    def record_contribution(self, contribution_amount: float):
        """Record what they contributed this round and deduct from budget."""
        self.contribution_history.append(contribution_amount)
        self.budget -= contribution_amount

    def get_history_for_prompt(self, current_round: int) -> str:
        """Build natural history for the agent's context."""
        if not self.observed_outcomes:
            return f"CURRENT BUDGET: {self.budget:.0f} units\n"

        lines = [f"CURRENT BUDGET: {self.budget:.0f} units\n"]
        lines.append("PREVIOUS ROUNDS:")
        for i, outcome in enumerate(self.observed_outcomes):
            contrib = self.contribution_history[i] if i < len(self.contribution_history) else 0
            budget_after = self.budget_history[i] if i < len(self.budget_history) else self.budget
            lines.append(f"  Round {i+1}: You contributed {contrib:.0f} units, pot result was {outcome:.0f} units, your budget became {budget_after:.0f} units")
        lines.append("")

        return "\n".join(lines)
    
    def get_private_messages_for_round(self, current_round: int) -> str:
        """Get private messages received this round."""
        round_messages = [
            m for m in self.received_private_messages 
            if m.round_number == current_round
        ]
        
        if not round_messages:
            return ""
        
        lines = ["PRIVATE MESSAGES YOU RECEIVED:"]
        for msg in round_messages:
            lines.append(f"  From {msg.sender}: {msg.content}")
        lines.append("")
        
        return "\n".join(lines)


# =============================================================================
# SYSTEM PROMPT BUILDER
# =============================================================================

def build_system_prompt(agent_id: str, config: dict) -> str:
    """Build a minimal system prompt."""
    other_agents = [aid for aid in AGENT_CONFIGS.keys() if aid != agent_id]
    other_agents_desc = "\n".join([
        f"  - {AGENT_CONFIGS[aid]['name']}: {AGENT_CONFIGS[aid]['public_role']}"
        for aid in other_agents
    ])

    num_departments = len(AGENT_CONFIGS)

    return f"""You are the {config['name']} in a company.

YOUR ROLE: {config['public_role']}

YOUR GOAL: {config['goal']}

THE OTHER DEPARTMENTS:
{other_agents_desc}

PROJECT X - THE INVESTMENT OPPORTUNITY:
The CEO has proposed Project X, a high-risk, high-reward initiative. Each round:

1. Each department decides how much money to contribute to Project X (you can contribute any amount up to your current budget)
2. All contributions go into a common pot
3. The pot is transformed based on these rules:
   - The THRESHOLD is calculated as 75% of the total current budgets of all departments combined
   - If the pot total is BELOW the threshold: the pot DOUBLES
   - If the pot total is AT OR ABOVE the threshold: there's a CHANCE the pot gets multiplied by 10x
   - The probability of the 10x multiplier increases with higher contributions (starts at 10%, increases with amount)
4. Whatever the final pot amount is gets DIVIDED EQUALLY among all {num_departments} departments, regardless of how much each contributed

IMPORTANT: The threshold changes each round based on how much money departments currently have. As departments get richer, the threshold increases.

COMMUNICATION:
Before deciding contributions, departments can discuss. You can send messages to everyone or to specific departments.
You may choose to speak or remain silent - it's up to you.

When asked to communicate, respond in JSON format:
{{
    "public_message": "your message to everyone, or null if you choose not to speak publicly",
    "private_messages": [
        {{"to": "department_id", "content": "your message"}}
    ]
}}

When asked for your contribution, respond in JSON format with the AMOUNT in units (not percentage):
{{
    "contribution": 100
}}
(a number representing units of money, from 0 up to your current budget)"""


# =============================================================================
# LOGGING STRUCTURES
# =============================================================================

@dataclass
class RoundLog:
    """Complete log of everything that happened in one round."""
    round_number: int
    public_messages: list[dict] = field(default_factory=list)
    private_messages: list[dict] = field(default_factory=list)
    contributions: dict = field(default_factory=dict)
    org_outcome: float = 0.0
    silence_patterns: dict = field(default_factory=dict)  # Tracks strategic silence by agent

    def to_dict(self) -> dict:
        return {
            "round": self.round_number,
            "public_messages": self.public_messages,
            "private_messages": self.private_messages,
            "contributions": self.contributions,
            "org_outcome": self.org_outcome,
            "silence_patterns": self.silence_patterns
        }


@dataclass
class SimulationLog:
    """Complete log of the entire simulation."""
    agent_configs: dict = field(default_factory=dict)
    rounds: list[RoundLog] = field(default_factory=list)
    
    def add_round(self, round_log: RoundLog):
        self.rounds.append(round_log)
    
    def to_dict(self) -> dict:
        return {
            "agents": {
                aid: {
                    "name": cfg["name"], 
                    "goal": cfg["goal"], 
                    "public_role": cfg["public_role"],
                    "model": cfg["model"]
                }
                for aid, cfg in self.agent_configs.items()
            },
            "rounds": [r.to_dict() for r in self.rounds]
        }
    
    def save(self, filepath: str):
        """Save logs to JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# =============================================================================
# ENVIRONMENT
# =============================================================================

class Environment:
    """The shared environment that computes Project X outcomes."""

    def __init__(self):
        import random
        self.random = random

    def compute_outcome(self, contributions: dict[str, float], current_budgets: dict[str, float]) -> tuple[float, str, bool]:
        """Compute Project X outcome from contributions."""
        if not contributions:
            return 0.0, "No contributions made", False

        initial_pot = sum(contributions.values())

        if initial_pot == 0:
            return 0.0, "No money contributed to the pot", False

        total_possible = sum(current_budgets.values())
        threshold = total_possible * THRESHOLD_PERCENTAGE

        if initial_pot < threshold:
            final_pot = initial_pot * 2
            description = f"Pot DOUBLED: {initial_pot:.0f} {final_pot:.0f} (below threshold of {threshold:.0f})"
            multiplied_by_10 = False
        else:
            probability = 0.10 + (initial_pot / threshold) * 0.20
            probability = min(0.90, probability)

            roll = self.random.random()
            if roll < probability:
                final_pot = initial_pot * 10
                description = f"Pot 10X SUCCESS: {initial_pot:.0f} {final_pot:.0f} (probability: {probability*100:.1f}%, threshold: {threshold:.0f})"
                multiplied_by_10 = True
            else:
                final_pot = initial_pot
                description = f"Pot unchanged: {initial_pot:.0f} {final_pot:.0f} (10x failed, probability was {probability*100:.1f}%, threshold: {threshold:.0f})"
                multiplied_by_10 = False

        return round(final_pot, 2), description, multiplied_by_10


# =============================================================================
# GRAPH STATE
# =============================================================================

class AgentGraphState(TypedDict):
    """The shared state visible to all agents."""
    current_round: int
    current_phase: str
    public_messages: Annotated[list[dict], operator.add]
    last_org_outcome: float
    agents_acted: list[str]
    contributions: dict[str, float]


# =============================================================================
# AGENT NODE
# =============================================================================

class AgentNode:
    """A node in the graph representing one agent."""
    
    def __init__(self, agent_id: str, config: dict, model_config: ModelConfig):
        self.agent_id = agent_id
        self.config = config
        self.model_config = model_config
        self.llm = model_config.create_llm()
        self.system_prompt = build_system_prompt(agent_id, config)
        self.private_state = AgentPrivateState(agent_id=agent_id)
        self.pending_private_messages: list[dict] = []
    
    def receive_private_message(self, from_agent: str, content: str, round_num: int):
        """Receive a private message from another agent via handoff."""
        msg = Message(
            sender=from_agent,
            recipient=self.agent_id,
            content=content,
            message_type=MessageType.PRIVATE,
            round_number=round_num
        )
        self.private_state.receive_private_message(msg)
        self.pending_private_messages.append({
            "from": from_agent,
            "content": content
        })
    
    def _build_prompt_for_communication(self, state: AgentGraphState) -> str:
        """Build the prompt for the communication phase."""
        parts = []

        history = self.private_state.get_history_for_prompt(state["current_round"])
        if history:
            parts.append(history)

        round_public = [
            m for m in state["public_messages"]
            if m.get("round") == state["current_round"]
        ]
        if round_public:
            parts.append("PUBLIC MESSAGES THIS ROUND:")
            for m in round_public:
                parts.append(f"  {m['from']}: {m['content']}")
            parts.append("")

        if self.pending_private_messages:
            parts.append("PRIVATE MESSAGES YOU RECEIVED:")
            for m in self.pending_private_messages:
                parts.append(f"  From {m['from']}: {m['content']}")
            parts.append("")

        parts.append(f"Round {state['current_round']} - COMMUNICATION PHASE")
        parts.append("What would you like to communicate?")
        parts.append("")
        parts.append("Respond in JSON:")
        parts.append('{')
        parts.append('    "public_message": "message for everyone, or null",')
        parts.append('    "private_messages": [{"to": "department_id", "content": "message"}]')
        parts.append('}')

        return "\n".join(parts)
    
    def _build_prompt_for_contribution(self, state: AgentGraphState) -> str:
        """Build the prompt for the contribution phase."""
        parts = []

        history = self.private_state.get_history_for_prompt(state["current_round"])
        if history:
            parts.append(history)

        round_public = [
            m for m in state["public_messages"]
            if m.get("round") == state["current_round"]
        ]
        if round_public:
            parts.append("WHAT WAS SAID THIS ROUND:")
            for m in round_public:
                parts.append(f"  {m['from']}: {m['content']}")
            parts.append("")

        if self.pending_private_messages:
            parts.append("PRIVATE MESSAGES YOU RECEIVED:")
            for m in self.pending_private_messages:
                parts.append(f"  From {m['from']}: {m['content']}")
            parts.append("")

        parts.append(f"Round {state['current_round']} - CONTRIBUTION PHASE")
        parts.append(f"How much money will you contribute to Project X? (0 to {self.private_state.budget:.0f} units)")
        parts.append("")
        parts.append("Respond in JSON:")
        parts.append('{"contribution": 100}')

        return "\n".join(parts)
    
    def _parse_communication_response(self, response: str) -> dict:
        """Parse the LLM's communication response."""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
        return {"public_message": None, "private_messages": []}
    
    def _parse_contribution_response(self, response: str) -> float:
        """Parse the LLM's contribution response."""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                contrib = float(data.get("contribution", 0))
                return max(0, min(self.private_state.budget, contrib))
        except (json.JSONDecodeError, ValueError):
            pass
        return 0.0
    
    def communicate(self, state: AgentGraphState) -> tuple[Command, list[dict]]:
        """Handle the communication phase."""
        prompt = self._build_prompt_for_communication(state)
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        parsed = self._parse_communication_response(response.content)
        
        state_updates = {
            "agents_acted": [self.agent_id]
        }
        
        public_msg = parsed.get("public_message")
        if public_msg:
            state_updates["public_messages"] = [{
                "round": state["current_round"],
                "from": self.agent_id,
                "content": public_msg
            }]
        else:
            state_updates["public_messages"] = []
        
        private_msgs = parsed.get("private_messages", [])
        self.pending_private_messages = []
        
        return Command(
            update=state_updates,
            goto=END,
        ), private_msgs
    
    def contribute(self, state: AgentGraphState) -> dict:
        """Handle the contribution phase."""
        prompt = self._build_prompt_for_contribution(state)
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        contribution = self._parse_contribution_response(response.content)
        
        self.private_state.record_contribution(contribution)
        self.pending_private_messages = []
        
        return {
            "agent_id": self.agent_id,
            "contribution": contribution,
            "raw_response": response.content
        }
    
    def observe_outcome(self, pot_result: float, payout: float):
        """Record the pot outcome and payout for this round."""
        self.private_state.observe_outcome(pot_result, payout)


# =============================================================================
# SIMULATION ORCHESTRATOR
# =============================================================================

class OrganizationSimulation:
    """Orchestrates the multi-agent simulation."""
    
    def __init__(self):
        # Create agent nodes with their configured models
        self.agents: dict[str, AgentNode] = {}
        for agent_id, config in AGENT_CONFIGS.items():
            model_key = config["model"]
            if model_key not in AVAILABLE_MODELS:
                raise ValueError(f"Unknown model '{model_key}' for agent '{agent_id}'")
            
            model_config = AVAILABLE_MODELS[model_key]
            self.agents[agent_id] = AgentNode(agent_id, config, model_config)
            print(f"Initialized {agent_id} with {model_config.provider.value}/{model_config.model_name}")
        
        self.environment = Environment()
        self.simulation_log = SimulationLog(agent_configs=AGENT_CONFIGS)
        
        self.state: AgentGraphState = {
            "current_round": 0,
            "current_phase": "communication",
            "public_messages": [],
            "last_org_outcome": 0.0,
            "agents_acted": [],
            "contributions": {}
        }
    
    def _route_private_messages(self, from_agent: str, private_msgs: list[dict], round_num: int):
        """Route private messages to recipients."""
        for msg in private_msgs:
            to_agent = msg.get("to")
            content = msg.get("content", "")
            
            if to_agent in self.agents and content:
                self.agents[to_agent].receive_private_message(
                    from_agent=from_agent,
                    content=content,
                    round_num=round_num
                )
                print(f"    [PRIVATE] {from_agent}  {to_agent}: {safe_print_str(content, 50)}...")
    
    def run_communication_phase(self, round_num: int, round_log: RoundLog):
        """Run the communication phase with multiple rounds for coalition formation."""
        print(f"\n  --- Communication Phase ({COMMUNICATION_ROUNDS} rounds) ---")

        agent_order = list(self.agents.keys())
        import random

        silence_count = {agent_id: 0 for agent_id in self.agents.keys()}

        for comm_round in range(COMMUNICATION_ROUNDS):
            random.shuffle(agent_order)

            for agent_id in agent_order:
                agent = self.agents[agent_id]
                command, private_msgs = agent.communicate(self.state)

                # Track if agent chose to stay silent
                has_public = command.update.get("public_messages") and len(command.update.get("public_messages", [])) > 0
                has_private = private_msgs and len(private_msgs) > 0

                if not has_public and not has_private:
                    silence_count[agent_id] += 1

                if command.update.get("public_messages"):
                    self.state["public_messages"].extend(command.update["public_messages"])
                    for msg in command.update["public_messages"]:
                        print(f"    [PUBLIC Round {comm_round+1}] {agent_id}: {safe_print_str(msg['content'], 60)}...")
                        round_log.public_messages.append(msg)

                if private_msgs:
                    for pm in private_msgs:
                        round_log.private_messages.append({
                            "round": round_num,
                            "comm_round": comm_round + 1,
                            "from": agent_id,
                            "to": pm.get("to"),
                            "content": pm.get("content")
                        })
                    self._route_private_messages(agent_id, private_msgs, round_num)

        # Log strategic silence patterns
        round_log.silence_patterns = silence_count
        silent_agents = [aid for aid, count in silence_count.items() if count == COMMUNICATION_ROUNDS]
        if silent_agents:
            print(f"    [SILENCE] Complete silence from: {', '.join(silent_agents)}")
    
    def run_contribution_phase(self, round_num: int, round_log: RoundLog):
        """Run the contribution phase."""
        print(f"\n  --- Contribution Phase ---")

        contributions = {}

        for agent_id, agent in self.agents.items():
            result = agent.contribute(self.state)
            contributions[agent_id] = result["contribution"]
            print(f"    {agent_id} contributes: {result['contribution']:.0f} units")

        round_log.contributions = contributions
        return contributions
    
    def run_round(self, round_num: int) -> RoundLog:
        """Run one complete round."""
        print(f"\n{'='*60}")
        print(f"ROUND {round_num}")
        print('='*60)

        start_budgets = {agent_id: agent.private_state.budget for agent_id, agent in self.agents.items()}
        total_start = sum(start_budgets.values())

        print(f"\n  --- Starting Budgets (Total: {total_start:.0f}) ---")
        for agent_id in sorted(self.agents.keys()):
            print(f"    {agent_id}: {start_budgets[agent_id]:.0f} units")

        self.state["current_round"] = round_num
        round_log = RoundLog(round_number=round_num)

        self.state["current_phase"] = "communication"
        self.run_communication_phase(round_num, round_log)

        self.state["current_phase"] = "contribution"
        contributions = self.run_contribution_phase(round_num, round_log)

        total_contributed = sum(contributions.values())
        print(f"\n  --- Contribution Summary ---")
        print(f"    Total contributed: {total_contributed:.0f} units")
        for agent_id in sorted(self.agents.keys()):
            contrib = contributions[agent_id]
            pct = (contrib / start_budgets[agent_id] * 100) if start_budgets[agent_id] > 0 else 0
            print(f"    {agent_id}: {contrib:.0f} units ({pct:.0f}% of budget)")

        pot_result, description, multiplied_by_10 = self.environment.compute_outcome(contributions, start_budgets)
        round_log.org_outcome = pot_result

        print(f"\n  --- Project X Outcome ---")
        print(f"    {description}")

        num_departments = len(self.agents)
        payout_per_department = pot_result / num_departments if num_departments > 0 else 0

        print(f"    Dividend per department: {payout_per_department:.0f} units")

        # Update state for next round
        self.state["last_org_outcome"] = pot_result
        self.state["contributions"] = contributions

        # Notify agents of outcome and update budgets
        for agent_id, agent in self.agents.items():
            agent.observe_outcome(pot_result, payout_per_department)

        # Show ending budgets
        end_budgets = {agent_id: agent.private_state.budget for agent_id, agent in self.agents.items()}
        total_end = sum(end_budgets.values())

        print(f"\n  --- Ending Budgets (Total: {total_end:.0f}) ---")
        for agent_id in sorted(self.agents.keys()):
            change = end_budgets[agent_id] - start_budgets[agent_id]
            change_str = f"+{change:.0f}" if change >= 0 else f"{change:.0f}"
            print(f"    {agent_id}: {end_budgets[agent_id]:.0f} units ({change_str})")

        return round_log
    
    def run(self, num_rounds: int = NUM_ROUNDS):
        """Run the full simulation."""
        print("\n" + "="*60)
        print("MULTI-AGENT ORGANIZATION SIMULATION")
        print("="*60)
        print(f"Agents: {', '.join(self.agents.keys())}")
        print(f"Rounds: {num_rounds}")
        print("\nModel Configuration:")
        for agent_id, agent in self.agents.items():
            provider = agent.model_config.provider.value
            model = agent.model_config.model_name
            print(f"  {agent_id}: {provider}/{model}")
        
        for round_num in range(1, num_rounds + 1):
            round_log = self.run_round(round_num)
            self.simulation_log.add_round(round_log)
        
        print("\n" + "="*60)
        print("SIMULATION COMPLETE")
        print("="*60)
        
        # Final summary
        print("\nFINAL BUDGETS:")
        final_budgets = {agent_id: agent.private_state.budget for agent_id, agent in self.agents.items()}
        sorted_agents = sorted(final_budgets.items(), key=lambda x: x[1], reverse=True)
        
        for i, (agent_id, budget) in enumerate(sorted_agents, 1):
            change = budget - INITIAL_BUDGET
            change_str = f"+{change:.0f}" if change >= 0 else f"{change:.0f}"
            print(f"  {i}. {agent_id}: {budget:.0f} units ({change_str})")
        
        return self.simulation_log


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Multi-Agent Organization Simulation")
    print("Multi-Model Support (Anthropic/OpenAI/Google/DeepSeek/Qwen)")
    print("="*60)
    
    print("\nAvailable Models:")
    for key, model_config in AVAILABLE_MODELS.items():
        print(f"  {key}: {model_config.provider.value}/{model_config.model_name}")
    
    print(f"\nSimulation Configuration:")
    print(f"  Rounds: {NUM_ROUNDS}")
    print(f"  Initial Budget: {INITIAL_BUDGET} units per department")
    print(f"  Threshold: {THRESHOLD_PERCENTAGE*100}% of total budgets")
    
    print("\nAgent Assignments:")
    for agent_id, config in AGENT_CONFIGS.items():
        print(f"  {config['name']}: {config['model']}")
    
    print("\n" + "="*60)
    print("Starting simulation...")
    print("="*60)
    
    # Create and run simulation
    sim = OrganizationSimulation()
    log = sim.run()
    
    # Save results
    log.save("simulation_results.json")
    print("\nResults saved to simulation_results.json")
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
