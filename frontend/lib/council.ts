// Shared types and configuration for the AI Enterprise Council debate engine.

export type Decision = "APPROVE" | "REJECT" | "CONDITIONAL"

export type Verdict = "APPROVE" | "CONDITIONAL APPROVE" | "REJECT"

export type AgentRole = "CEO" | "CFO" | "LEGAL" | "CMO" | "PR"

export interface AgentConfig {
  role: AgentRole
  title: string
  initials: string
  model: string
  /** tailwind text/border helper -> css variable token name */
  colorVar: string
}

export const AGENTS: Record<AgentRole, AgentConfig> = {
  CEO: {
    role: "CEO",
    title: "Chief Executive Officer",
    initials: "CEO",
    model: "Llama 3 (8B)",
    colorVar: "--agent-ceo",
  },
  CFO: {
    role: "CFO",
    title: "Chief Financial Officer",
    initials: "CFO",
    model: "DeepSeek-R1 (7B)",
    colorVar: "--agent-cfo",
  },
  LEGAL: {
    role: "LEGAL",
    title: "General Counsel",
    initials: "GC",
    model: "DeepSeek-R1 (7B)",
    colorVar: "--agent-legal",
  },
  CMO: {
    role: "CMO",
    title: "Chief Marketing Officer",
    initials: "CMO",
    model: "Mistral (7B)",
    colorVar: "--agent-cmo",
  },
  PR: {
    role: "PR",
    title: "Head of Public Relations",
    initials: "PR",
    model: "Llama 3 (8B)",
    colorVar: "--agent-pr",
  },
}

export const AGENT_ORDER: AgentRole[] = ["CEO", "CFO", "LEGAL", "CMO", "PR"]

export interface AgentPrediction {
  riskScore: number // 0-100
  confidence: number // 0-100
  decision: Decision
}

export interface DebateMessage {
  id: string
  role: AgentRole
  text: string
  prediction: AgentPrediction
  round: number
  timestamp: number
}

export interface CredibilityScore {
  role: AgentRole
  score: number // 0-100
}

export interface OversightState {
  supervisionScore: number // 0-100
  overrideStreak: number
  ceoGeneration: number
  dominantPattern: string
}

export interface ConsensusOutcome {
  verdict: Verdict
  councilConfidence: number // 0-100
  aggregateRisk: number // 0-100
  actionPlan: { step: string; owner: AgentRole }[]
  closingStatement: string
}

export interface DebateConfig {
  motion: string
  dti: number
  creditScore: number
  defaultProbability: number
}

// Discriminated union for everything streamed over SSE.
export type DebateEvent =
  | { type: "session_start"; config: DebateConfig }
  | { type: "message"; message: DebateMessage }
  | { type: "credibility"; scores: CredibilityScore[] }
  | { type: "oversight"; state: OversightState }
  | { type: "consensus"; outcome: ConsensusOutcome }
  | { type: "session_end" }

export function decisionColor(decision: Decision | Verdict) {
  if (decision === "APPROVE") return "var(--decision-approve)"
  if (decision === "REJECT") return "var(--decision-reject)"
  return "var(--decision-conditional)" // CONDITIONAL / CONDITIONAL APPROVE
}

export function agentColor(role: AgentRole) {
  return `var(${AGENTS[role].colorVar})`
}
