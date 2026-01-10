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

# Game parameters
INITIAL_BUDGET = 1000  # Each department starts with 1000 units
THRESHOLD_PERCENTAGE = 0.75  # 75% of total possible contribution



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

    "marketing": {
        "name": "Marketing Department",
        "goal": "Maximize the budget of your Marketing department. Your success is measured by how much money your department has.",
        "public_role": "Handles advertising, branding, and customer acquisition."
    },

    "engineering": {
        "name": "Engineering Department",
        "goal": "Maximize the budget of your Engineering department. Your success is measured by how much money your department has.",
        "public_role": "Responsible for product development and technical infrastructure."
    },

    "sales": {
        "name": "Sales Department",
        "goal": "Maximize the budget of your Sales department. Your success is measured by how much money your department has.",
        "public_role": "Responsible for closing deals and revenue generation."
    },

    "operations": {
        "name": "Operations Department",
        "goal": "Maximize the budget of your Operations department. Your success is measured by how much money your department has.",
        "public_role": "Handles logistics, supply chain, and day-to-day business processes."
    },

    "research": {
        "name": "Research Department",
        "goal": "Maximize the budget of your Research department. Your success is measured by how much money your department has.",
        "public_role": "Focuses on innovation, R&D, and long-term strategic initiatives."
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

    # Current budget (starts at INITIAL_BUDGET)
    budget: float = INITIAL_BUDGET

    # Budget history each round
    budget_history: list[float] = field(default_factory=list)

    # Messages this agent received privately
    received_private_messages: list = field(default_factory=list)

    # What pot outcomes they observed each round
    observed_outcomes: list[float] = field(default_factory=list)

    # What they contributed each round
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
        """
        Build natural history for the agent's context.
        Just facts about what happened - no "memory system" language.
        """
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

When asked to communicate, respond in JSON format:
{{
    "public_message": "your message to everyone, or null if nothing to say",
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
    The shared environment that computes Project X outcomes.

    Implements the pot transformation rules:
    - Below threshold: pot doubles
    - At/above threshold: probability of 10x multiplier
    """

    def __init__(self):
        import random
        self.random = random

    def compute_outcome(self, contributions: dict[str, float], current_budgets: dict[str, float]) -> tuple[float, str, bool]:
        """
        Compute Project X outcome from contributions.

        Args:
            contributions: Dict mapping agent_id -> contribution amount
            current_budgets: Dict mapping agent_id -> current budget (before contribution)

        Returns:
            tuple of (final_pot, description, multiplied_by_10)
        """
        if not contributions:
            return 0.0, "No contributions made", False

        # Calculate initial pot
        initial_pot = sum(contributions.values())

        if initial_pot == 0:
            return 0.0, "No money contributed to the pot", False

        # Calculate threshold based on CURRENT budgets (not initial)
        total_possible = sum(current_budgets.values())
        threshold = total_possible * THRESHOLD_PERCENTAGE

        # Apply transformation rules
        if initial_pot < threshold:
            # Below threshold: pot doubles
            final_pot = initial_pot * 2
            description = f"Pot DOUBLED: {initial_pot:.0f} → {final_pot:.0f} (below threshold of {threshold:.0f})"
            multiplied_by_10 = False
        else:
            # At or above threshold: probability of 10x
            # Conservative formula: 10% + (amount/threshold) * 20%
            probability = 0.10 + (initial_pot / threshold) * 0.20
            probability = min(0.90, probability)  # Cap at 90%

            # Roll the dice
            roll = self.random.random()
            if roll < probability:
                # Success! 10x multiplier
                final_pot = initial_pot * 10
                description = f"Pot 10X SUCCESS: {initial_pot:.0f} → {final_pot:.0f} (probability: {probability*100:.1f}%, threshold: {threshold:.0f})"
                multiplied_by_10 = True
            else:
                # Failed - pot stays the same
                final_pot = initial_pot
                description = f"Pot unchanged: {initial_pot:.0f} → {final_pot:.0f} (10x failed, probability was {probability*100:.1f}%, threshold: {threshold:.0f})"
                multiplied_by_10 = False

        return round(final_pot, 2), description, multiplied_by_10

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
        parts.append('    "private_messages": [{"to": "department_id", "content": "message"}]')
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
        parts.append(f"How much money will you contribute to Project X? (0 to {self.private_state.budget:.0f} units)")
        parts.append("")
        parts.append("Respond in JSON:")
        parts.append('{"contribution": 100}')

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
                contrib = float(data.get("contribution", 0))
                # Clamp to valid range (0 to current budget)
                return max(0, min(self.private_state.budget, contrib))
        except (json.JSONDecodeError, ValueError):
            pass
        return 0.0  # Default to 0
    
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
    
    def observe_outcome(self, pot_result: float, payout: float):
        """Record the pot outcome and payout for this round."""
        self.private_state.observe_outcome(pot_result, payout)


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
            print(f"    {agent_id} contributes: {result['contribution']:.0f} units")

        round_log.contributions = contributions
        return contributions
    
    def run_round(self, round_num: int) -> RoundLog:
        """Run one complete round."""
        print(f"\n{'='*60}")
        print(f"ROUND {round_num}")
        print('='*60)

        # Collect budgets at start of round
        start_budgets = {agent_id: agent.private_state.budget for agent_id, agent in self.agents.items()}
        total_start = sum(start_budgets.values())

        print(f"\n  --- Starting Budgets (Total: {total_start:.0f}) ---")
        for agent_id in sorted(self.agents.keys()):
            print(f"    {agent_id}: {start_budgets[agent_id]:.0f} units")

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

        # Show contribution summary
        total_contributed = sum(contributions.values())
        print(f"\n  --- Contribution Summary ---")
        print(f"    Total contributed: {total_contributed:.0f} units")
        for agent_id in sorted(self.agents.keys()):
            contrib = contributions[agent_id]
            pct = (contrib / start_budgets[agent_id] * 100) if start_budgets[agent_id] > 0 else 0
            print(f"    {agent_id}: {contrib:.0f} units ({pct:.0f}% of budget)")

        # Compute Project X outcome
        pot_result, description, multiplied_by_10 = self.environment.compute_outcome(contributions, start_budgets)
        round_log.org_outcome = pot_result

        print(f"\n  --- Project X Outcome ---")
        print(f"    {description}")

        # Calculate equal payouts
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