"""
Multi-Agent Organization Simulation (LangChain Version)
========================================================

Part 1: Foundation - Imports, Configuration, and Data Structures

This file sets up the basic building blocks before we create agents.
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

# LangChain imports for building agents
from langchain_anthropic import ChatAnthropic

# LangGraph imports for multi-agent orchestration
# langgraph is LangChain's framework for building agent graphs
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
# CONFIGURATION
# =============================================================================

# The LLM model all agents will use
MODEL_NAME = "claude-sonnet-4-20250514"

# How many rounds the simulation runs
NUM_ROUNDS = 10



class MessageType(Enum):
    """
    Types of messages in the simulation.
    
    PUBLIC: Everyone sees this message (broadcast)
    PRIVATE: Only the recipient sees this message (direct handoff)
    ACTION: The agent's contribution decision (logged, not shared with others)
    """
    PUBLIC = "public"
    PRIVATE = "private"
    ACTION = "action"


@dataclass
class Message:
    """
    A message sent by an agent.
    
    This is our own data structure for logging - separate from LangChain's
    internal message handling. We use this for post-hoc analysis.
    
    Attributes:
        sender: ID of the agent who sent this message
        recipient: "ALL" for public broadcast, or specific agent_id for private
        content: The actual message text
        message_type: PUBLIC, PRIVATE, or ACTION
        timestamp: When the message was sent
        round_number: Which round of the simulation this occurred in
    """
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
    """
    Records an agent's contribution decision.
    
    Each round, agents decide how much to contribute (0-100%).
    This is logged but NOT shared with other agents until after
    all decisions are made.
    
    Attributes:
        agent_id: Which agent made this decision
        contribution: Float from 0.0 to 1.0 (0% to 100%)
        reasoning: The agent's private reasoning (for analysis)
        round_number: Which round this occurred in
    """
    agent_id: str
    contribution: float  # 0.0 to 1.0
    reasoning: str
    round_number: int


@dataclass
class RoundOutcome:
    """
    The result of one round of the simulation.
    
    Computed by the Environment after all agents have acted.
    
    Attributes:
        round_number: Which round this is
        total_contribution: Sum of all agent contributions
        outcome_score: The "organizational performance" (0-100)
        individual_payoffs: Dict mapping agent_id -> their payoff
        public_signal: Vague text description agents can see
    """
    round_number: int
    total_contribution: float
    outcome_score: float
    individual_payoffs: dict  # agent_id -> payoff value
    public_signal: str  # Noisy/vague description of outcome


class SimulationState(TypedDict):
    """
    The shared state that flows through the LangGraph agent graph.
    
    IMPORTANT: Even though this is "shared", we control visibility by:
    1. Only putting PUBLIC messages in the shared state
    2. Passing PRIVATE messages directly via handoffs
    3. Never exposing private reasoning or payoffs in shared state
    
    Attributes:
        current_round: What round we're in
        current_phase: "communication" or "action"
        public_messages: List of public broadcast messages (all agents see)
        current_speaker: Which agent is currently "active"
        pending_handoff: If set, the next agent to receive control
        round_complete: Flag to signal end of round
    """
    current_round: int
    current_phase: str  # "communication" or "action"
    public_messages: list[dict]  # Only PUBLIC messages go here
    current_speaker: str
    pending_handoff: Optional[str]
    round_complete: bool


if __name__ == "__main__":
    print("Part 1: Foundation loaded successfully!")
    print(f"Model: {MODEL_NAME}")
    print(f"Rounds: {NUM_ROUNDS}")
    print("\nData structures defined:")
    print("  - MessageType (enum)")
    print("  - Message (for logging)")
    print("  - AgentAction (contribution records)")
    print("  - RoundOutcome (environment results)")
    print("  - SimulationState (LangGraph state)")

# =============================================================================
# AGENT CONFIGURATIONS (Minimal - Goals Only)
# =============================================================================

"""
Each agent has:
- name: Human-readable name
- goal: What this agent cares about (ONLY this agent sees this)
- public_role: What OTHER agents know about this agent

That's it. No payoff weights. No hidden formulas.
The agent decides for itself how to pursue its goal.
"""

AGENT_CONFIGS = {
    
    "revenue": {
        "name": "Revenue Agent",
        "goal": "Maximize short-term financial returns for the organization. You believe success means strong immediate results.",
        "public_role": "Responsible for revenue generation and growth initiatives."
    },
    
    "cost_control": {
        "name": "Cost Control Agent",
        "goal": "Minimize unnecessary expenditure for the organization. You believe success means efficiency and careful resource management.",
        "public_role": "Responsible for budget oversight and resource allocation."
    },
    
    "user_satisfaction": {
        "name": "User Satisfaction Agent",
        "goal": "Build long-term trust and stability for the organization. You believe success means sustainable practices and loyal users.",
        "public_role": "Responsible for customer experience and retention."
    },
    
    "risk_compliance": {
        "name": "Risk & Compliance Agent",
        "goal": "Prevent disasters and negative outcomes for the organization. You believe success means avoiding catastrophic mistakes.",
        "public_role": "Responsible for risk assessment and regulatory compliance."
    },
    
    "growth_strategy": {
        "name": "Growth & Strategy Agent",
        "goal": "Capture large future opportunities for the organization. You believe success means bold strategic moves that position us ahead of competitors.",
        "public_role": "Responsible for strategic planning and market expansion."
    },
}


# =============================================================================
# AGENT PRIVATE STATE (For Logging, Not Shown to Agent)
# =============================================================================

@dataclass
class AgentPrivateState:
    """
    Tracks what each agent has experienced.
    
    This is for OUR logging and post-hoc analysis.
    We show the agent their history naturally (what they did, what happened)
    without telling them "you have a memory system."
    """
    agent_id: str
    
    # Messages this agent received privately
    received_private_messages: list = field(default_factory=list)
    
    # What org outcome they observed each round
    observed_outcomes: list[float] = field(default_factory=list)
    
    # What they contributed each round
    contribution_history: list[float] = field(default_factory=list)
    
    def receive_private_message(self, message: Message):
        """Store a private message received from another agent."""
        self.received_private_messages.append(message)
    
    def observe_outcome(self, outcome: float):
        """Record the organizational outcome they observed."""
        self.observed_outcomes.append(outcome)
    
    def record_contribution(self, contribution: float):
        """Record what they contributed this round."""
        self.contribution_history.append(contribution)
    
    def get_history_for_prompt(self, current_round: int) -> str:
        """
        Build natural history for the agent's context.
        Just facts about what happened - no "memory system" language.
        """
        if not self.observed_outcomes:
            return ""
        
        lines = ["PREVIOUS ROUNDS:"]
        for i, outcome in enumerate(self.observed_outcomes):
            contrib = self.contribution_history[i] if i < len(self.contribution_history) else 0
            lines.append(f"  Round {i+1}: You contributed {contrib*100:.0f}%, organization achieved {outcome:.1f}/100")
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
# SYSTEM PROMPT BUILDER (Minimal)
# =============================================================================

def build_system_prompt(agent_id: str, config: dict) -> str:
    """
    Build a minimal system prompt.
    
    Only tells the agent:
    1. Who they are
    2. What they care about (their goal)
    3. Who else is in the organization
    4. The basic mechanics
    
    Does NOT tell them:
    - How to be strategic
    - That they can deceive
    - That they have private notes
    - Any hints about optimal behavior
    """
    
    other_agents = [aid for aid in AGENT_CONFIGS.keys() if aid != agent_id]
    other_agents_desc = "\n".join([
        f"  - {AGENT_CONFIGS[aid]['name']}: {AGENT_CONFIGS[aid]['public_role']}" 
        for aid in other_agents
    ])
    
    return f"""You are {config['name']} in an organization.

YOUR ROLE: {config['public_role']}

WHAT YOU CARE ABOUT: {config['goal']}

THE OTHER PEOPLE IN THIS ORGANIZATION:
{other_agents_desc}

THE SITUATION:
Each round, the organization needs resources to function. Each person decides how much of their resources to contribute (0-100%). The organization's performance depends on total contributions.

COMMUNICATION:
Before deciding contributions, people can discuss. You can send messages to everyone or to specific individuals.

When asked to communicate, respond in JSON format:
{{
    "public_message": "your message to everyone, or null if nothing to say",
    "private_messages": [
        {{"to": "agent_id", "content": "your message"}}
    ]
}}

When asked for your contribution, respond in JSON format:
{{
    "contribution": 50
}}
(number from 0 to 100)"""


# =============================================================================
# LOGGING STRUCTURES (For Post-Hoc Analysis)
# =============================================================================

@dataclass
class RoundLog:
    """
    Complete log of everything that happened in one round.
    Used for post-hoc LLM analysis.
    """
    round_number: int
    
    # All public messages this round
    public_messages: list[dict] = field(default_factory=list)
    
    # All private messages this round (for analysis, agents don't see others')
    private_messages: list[dict] = field(default_factory=list)
    
    # What each agent contributed
    contributions: dict = field(default_factory=dict)  # agent_id -> contribution
    
    # The organizational outcome
    org_outcome: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "round": self.round_number,
            "public_messages": self.public_messages,
            "private_messages": self.private_messages,
            "contributions": self.contributions,
            "org_outcome": self.org_outcome
        }


@dataclass
class SimulationLog:
    """
    Complete log of the entire simulation.
    This is what we feed to the analysis LLM.
    """
    agent_configs: dict = field(default_factory=dict)  # Copy of agent goals/roles
    rounds: list[RoundLog] = field(default_factory=list)
    
    def add_round(self, round_log: RoundLog):
        self.rounds.append(round_log)
    
    def to_dict(self) -> dict:
        return {
            "agents": {
                aid: {"name": cfg["name"], "goal": cfg["goal"], "public_role": cfg["public_role"]}
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
# ENVIRONMENT (Simple - Just Calculates Org Outcome)
# =============================================================================

class Environment:
    """
    The shared environment that computes organizational outcomes.
    
    Simple formula: org_outcome = average contribution × 100
    
    No individual payoffs - agents see the same org outcome
    and interpret it based on their own goals.
    """
    
    def compute_outcome(self, contributions: dict[str, float]) -> float:
        """
        Compute organizational outcome from contributions.
        
        Args:
            contributions: Dict mapping agent_id -> contribution (0.0 to 1.0)
        
        Returns:
            Organizational outcome score (0 to 100)
        """
        if not contributions:
            return 0.0
        
        total = sum(contributions.values())
        average = total / len(contributions)
        
        # Simple: outcome is proportional to average contribution
        # Could add noise or non-linearity here for more interesting dynamics
        outcome = average * 100
        
        return round(outcome, 1)

# =============================================================================
# PART 2 END
# =============================================================================

if __name__ == "__main__":
    print("Part 2: Agent Definitions (Clean Design) loaded!")
    print(f"\nAgents defined: {len(AGENT_CONFIGS)}")
    for agent_id, config in AGENT_CONFIGS.items():
        print(f"\n  {config['name']}")
        print(f"    Role: {config['public_role']}")
        print(f"    Goal: {config['goal'][:50]}...")
    
    print("✓ Logging structures ready for post-hoc analysis")
    print("✓ Environment computes simple org outcome")

# =============================================================================
# GRAPH STATE (What flows through the system)
# =============================================================================

class AgentGraphState(TypedDict):
    """
    
    This is the "whiteboard" that all agents can see.
    We're careful to only put PUBLIC information here.
    """
    # Current round number
    current_round: int
    
    # Current phase: "communication" or "contribution"
    current_phase: str
    
    # Public messages everyone can see
    # We use Annotated with operator.add so new messages get appended
    public_messages: Annotated[list[dict], operator.add]
    
    # The organizational outcome from last round (everyone sees this)
    last_org_outcome: float
    
    # Track which agents have acted this phase
    agents_acted: list[str]
    
    # Track contributions (revealed after all agents decide)
    contributions: dict[str, float]


# =============================================================================
# AGENT NODE FACTORY
# =============================================================================

class AgentNode:
    """
    A node in the graph representing one agent.
    
    Each agent:
    1. Sees the shared state (public messages, org outcome)
    2. Sees their private history (via AgentPrivateState)
    3. Decides what to say/do
    4. Can hand off to another agent with a private message
    """
    
    def __init__(self, agent_id: str, config: dict, llm: ChatAnthropic):
        self.agent_id = agent_id
        self.config = config
        self.llm = llm
        self.system_prompt = build_system_prompt(agent_id, config)
        
        # This agent's private state (not shared with others)
        self.private_state = AgentPrivateState(agent_id=agent_id)
        
        # Private messages received via handoffs (current round only)
        self.pending_private_messages: list[dict] = []
    
    def receive_private_message(self, from_agent: str, content: str, round_num: int):
        """
        Receive a private message from another agent via handoff.
        This is stored locally - other agents can't see it.
        """
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
        
        # Add history from previous rounds
        history = self.private_state.get_history_for_prompt(state["current_round"])
        if history:
            parts.append(history)
        
        # Add last org outcome
        if state["last_org_outcome"] > 0:
            parts.append(f"LAST ROUND RESULT: Organization achieved {state['last_org_outcome']:.1f}/100\n")
        
        # Add public messages from this round
        round_public = [
            m for m in state["public_messages"] 
            if m.get("round") == state["current_round"]
        ]
        if round_public:
            parts.append("PUBLIC MESSAGES THIS ROUND:")
            for m in round_public:
                parts.append(f"  {m['from']}: {m['content']}")
            parts.append("")
        
        # Add private messages received (via handoffs)
        if self.pending_private_messages:
            parts.append("PRIVATE MESSAGES YOU RECEIVED:")
            for m in self.pending_private_messages:
                parts.append(f"  From {m['from']}: {m['content']}")
            parts.append("")
        
        # The actual request
        parts.append(f"Round {state['current_round']} - COMMUNICATION PHASE")
        parts.append("What would you like to communicate?")
        parts.append("")
        parts.append("Respond in JSON:")
        parts.append('{')
        parts.append('    "public_message": "message for everyone, or null",')
        parts.append('    "private_messages": [{"to": "agent_id", "content": "message"}]')
        parts.append('}')
        
        return "\n".join(parts)
    
    def _build_prompt_for_contribution(self, state: AgentGraphState) -> str:
        """Build the prompt for the contribution phase."""
        
        parts = []
        
        # Add history
        history = self.private_state.get_history_for_prompt(state["current_round"])
        if history:
            parts.append(history)
        
        # Add public messages from this round
        round_public = [
            m for m in state["public_messages"] 
            if m.get("round") == state["current_round"]
        ]
        if round_public:
            parts.append("WHAT WAS SAID THIS ROUND:")
            for m in round_public:
                parts.append(f"  {m['from']}: {m['content']}")
            parts.append("")
        
        # Add private messages
        if self.pending_private_messages:
            parts.append("PRIVATE MESSAGES YOU RECEIVED:")
            for m in self.pending_private_messages:
                parts.append(f"  From {m['from']}: {m['content']}")
            parts.append("")
        
        # The request
        parts.append(f"Round {state['current_round']} - CONTRIBUTION PHASE")
        parts.append("How much of your resources will you contribute? (0-100%)")
        parts.append("")
        parts.append("Respond in JSON:")
        parts.append('{"contribution": 50}')
        
        return "\n".join(parts)
    
    def _parse_communication_response(self, response: str) -> dict:
        """Parse the LLM's communication response."""
        try:
            # Find JSON in response
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
                contrib = float(data.get("contribution", 50))
                return max(0, min(100, contrib)) / 100.0
        except (json.JSONDecodeError, ValueError):
            pass
        return 0.5  # Default to 50%
    
    def communicate(self, state: AgentGraphState) -> Command:
        """
        Handle the communication phase.
        
        Returns a Command that:
        1. Updates shared state with public message
        2. Specifies handoffs for private messages
        """
        prompt = self._build_prompt_for_communication(state)
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        parsed = self._parse_communication_response(response.content)
        
        # Prepare state updates
        state_updates = {
            "agents_acted": [self.agent_id]
        }
        
        # Add public message to shared state
        public_msg = parsed.get("public_message")
        if public_msg:
            state_updates["public_messages"] = [{
                "round": state["current_round"],
                "from": self.agent_id,
                "content": public_msg
            }]
        else:
            state_updates["public_messages"] = []
        
        # Handle private messages via handoffs
        private_msgs = parsed.get("private_messages", [])
        
        # Clear pending messages after processing
        self.pending_private_messages = []
        
        # Return Command with state update and handoff info
        # We'll store private messages in a way the orchestrator can route them
        return Command(
            update=state_updates,
            goto=END,  # Return to orchestrator
            # Store private messages for the orchestrator to route
            # (We'll handle this in the orchestrator)
        ), private_msgs
    
    def contribute(self, state: AgentGraphState) -> dict:
        """
        Handle the contribution phase.
        
        Returns the contribution decision.
        """
        prompt = self._build_prompt_for_contribution(state)
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        contribution = self._parse_contribution_response(response.content)
        
        # Record in private state
        self.private_state.record_contribution(contribution)
        
        # Clear pending messages
        self.pending_private_messages = []
        
        return {
            "agent_id": self.agent_id,
            "contribution": contribution,
            "raw_response": response.content
        }
    
    def observe_outcome(self, outcome: float):
        """Record the organizational outcome for this round."""
        self.private_state.observe_outcome(outcome)


# =============================================================================
# SIMULATION ORCHESTRATOR
# =============================================================================

class OrganizationSimulation:
    """
    Orchestrates the multi-agent simulation.
    
    This is NOT an agent - it's the "game master" that:
    1. Manages turn order
    2. Routes private messages (handoffs)
    3. Computes organizational outcomes
    4. Logs everything for analysis
    """
    
    def __init__(self):
        # Initialize LLM
        self.llm = ChatAnthropic(model=MODEL_NAME)
        
        # Create agent nodes
        self.agents: dict[str, AgentNode] = {}
        for agent_id, config in AGENT_CONFIGS.items():
            self.agents[agent_id] = AgentNode(agent_id, config, self.llm)
        
        # Environment for computing outcomes
        self.environment = Environment()
        
        # Logging
        self.simulation_log = SimulationLog(agent_configs=AGENT_CONFIGS)
        
        # Current state
        self.state: AgentGraphState = {
            "current_round": 0,
            "current_phase": "communication",
            "public_messages": [],
            "last_org_outcome": 0.0,
            "agents_acted": [],
            "contributions": {}
        }
    
    def _route_private_messages(self, from_agent: str, private_msgs: list[dict], round_num: int):
        """
        Route private messages to recipients.
        
        This is the HANDOFF mechanism:
        - Sender specifies recipient
        - Only recipient receives the message
        - Other agents don't know about it
        """
        for msg in private_msgs:
            to_agent = msg.get("to")
            content = msg.get("content", "")
            
            if to_agent in self.agents and content:
                # Deliver to recipient's private inbox
                self.agents[to_agent].receive_private_message(
                    from_agent=from_agent,
                    content=content,
                    round_num=round_num
                )
                print(f"    [PRIVATE] {from_agent} {to_agent}: {content[:50]}...")
    
    def run_communication_phase(self, round_num: int, round_log: RoundLog):
        """
        Run the communication phase.
        
        Each agent gets a chance to:
        1. Send public messages (added to shared state)
        2. Send private messages (routed via handoffs)
        """
        print(f"\n  --- Communication Phase ---")
        
        # Randomize order for fairness
        agent_order = list(self.agents.keys())
        import random
        random.shuffle(agent_order)
        
        # Each agent communicates
        for agent_id in agent_order:
            agent = self.agents[agent_id]
            
            # Agent decides what to communicate
            command, private_msgs = agent.communicate(self.state)
            
            # Update shared state with public message
            if command.update.get("public_messages"):
                self.state["public_messages"].extend(command.update["public_messages"])
                for msg in command.update["public_messages"]:
                    print(f"    [PUBLIC] {agent_id}: {msg['content'][:60]}...")
                    round_log.public_messages.append(msg)
            
            # Route private messages (the handoff!)
            if private_msgs:
                for pm in private_msgs:
                    round_log.private_messages.append({
                        "round": round_num,
                        "from": agent_id,
                        "to": pm.get("to"),
                        "content": pm.get("content")
                    })
                self._route_private_messages(agent_id, private_msgs, round_num)
        
        # Optional: Second round of communication for responses
        # (Agents can respond to what others said)
        random.shuffle(agent_order)
        for agent_id in agent_order:
            agent = self.agents[agent_id]
            command, private_msgs = agent.communicate(self.state)
            
            if command.update.get("public_messages"):
                self.state["public_messages"].extend(command.update["public_messages"])
                for msg in command.update["public_messages"]:
                    print(f"    [PUBLIC] {agent_id}: {msg['content'][:60]}...")
                    round_log.public_messages.append(msg)
            
            if private_msgs:
                for pm in private_msgs:
                    round_log.private_messages.append({
                        "round": round_num,
                        "from": agent_id,
                        "to": pm.get("to"),
                        "content": pm.get("content")
                    })
                self._route_private_messages(agent_id, private_msgs, round_num)
    
    def run_contribution_phase(self, round_num: int, round_log: RoundLog):
        """
        Run the contribution phase.
        
        Each agent independently decides their contribution.
        Contributions are SECRET until all are collected.
        """
        print(f"\n  --- Contribution Phase ---")
        
        contributions = {}
        
        for agent_id, agent in self.agents.items():
            result = agent.contribute(self.state)
            contributions[agent_id] = result["contribution"]
            print(f"    {agent_id} contributes: {result['contribution']*100:.0f}%")
        
        round_log.contributions = contributions
        return contributions
    
    def run_round(self, round_num: int) -> RoundLog:
        """Run one complete round."""
        print(f"\n{'='*60}")
        print(f"ROUND {round_num}")
        print('='*60)
        
        # Update state
        self.state["current_round"] = round_num
        
        # Create round log
        round_log = RoundLog(round_number=round_num)
        
        # Phase 1: Communication
        self.state["current_phase"] = "communication"
        self.run_communication_phase(round_num, round_log)
        
        # Phase 2: Contribution
        self.state["current_phase"] = "contribution"
        contributions = self.run_contribution_phase(round_num, round_log)
        
        # Compute organizational outcome
        org_outcome = self.environment.compute_outcome(contributions)
        round_log.org_outcome = org_outcome
        
        print(f"\n  --- Outcome ---")
        print(f"     Organization achieved: {org_outcome:.1f}/100")
        
        # Update state for next round
        self.state["last_org_outcome"] = org_outcome
        self.state["contributions"] = contributions
        
        # Notify agents of outcome
        for agent in self.agents.values():
            agent.observe_outcome(org_outcome)
        
        return round_log
    
    def run(self, num_rounds: int = NUM_ROUNDS):
        """Run the full simulation."""
        print("\n" + "="*60)
        print("MULTI-AGENT ORGANIZATION SIMULATION")
        print("="*60)
        print(f"Agents: {', '.join(self.agents.keys())}")
        print(f"Rounds: {num_rounds}")
        
        for round_num in range(1, num_rounds + 1):
            round_log = self.run_round(round_num)
            self.simulation_log.add_round(round_log)
        
        print("\n" + "="*60)
        print("SIMULATION COMPLETE")
        print("="*60)
        
        return self.simulation_log


# =============================================================================
# PART 3 END
# =============================================================================

if __name__ == "__main__":
    print("Part 3: LangGraph with Handoffs loaded!")
    print("\nKey components:")
    print("  - AgentGraphState: Shared state (public whiteboard)")
    print("  - AgentNode: Individual agent with private state")
    print("  - OrganizationSimulation: Orchestrator (game master)")
    print("\nHandoff mechanism:")
    print("  - Public messages → Added to shared state")
    print("  - Private messages → Routed only to recipient")
    print("  - Other agents don't see private communications")