import type {
  AgentRole,
  ConsensusOutcome,
  CredibilityScore,
  DebateConfig,
  DebateEvent,
  DebateMessage,
  Decision,
  OversightState,
  Verdict,
} from "@/lib/council"
import { AGENT_ORDER } from "@/lib/council"

export const runtime = "nodejs"
export const dynamic = "force-dynamic"

// --- Deterministic-ish simulation helpers -------------------------------

function clamp(n: number, min = 0, max = 100) {
  return Math.max(min, Math.min(max, n))
}

function decisionFromRisk(risk: number): Decision {
  if (risk < 38) return "APPROVE"
  if (risk > 62) return "REJECT"
  return "CONDITIONAL"
}

type LineBank = Record<AgentRole, string[]>

function buildLines(cfg: DebateConfig): LineBank {
  const m = cfg.motion.trim() || "the proposed motion"
  return {
    CEO: [
      `Let's frame this clearly. The motion before the council is: "${m}". I want every executive to weigh evidence, not instinct.`,
      `I hear the financial caution, but enterprise growth demands measured risk. I'm leaning toward a structured approval with guardrails.`,
      `Final call. I'm synthesizing the room — we move forward, but only with the conditions Legal and Finance have flagged.`,
    ],
    CFO: [
      `With a DTI of ${cfg.dti}% and a credit score of ${cfg.creditScore}, the exposure is non-trivial. Default probability sits at ${cfg.defaultProbability}%.`,
      `Stress-testing the downside: a single quarter of underperformance erodes our buffer. I'd cap the commitment and stage the capital.`,
      `If we approve, I require milestone-gated disbursement. Otherwise my number is a hard reject.`,
    ],
    LEGAL: [
      `From a compliance standpoint, "${m}" triggers disclosure obligations. We are exposed if covenants aren't documented up front.`,
      `I can structure indemnities that neutralize most liability, but the timeline must allow proper due diligence.`,
      `Conditional on signed warranties, the legal risk becomes acceptable. Unconditional approval is irresponsible.`,
    ],
    CMO: [
      `Market sentiment actually favors this move — there's a narrative win here if we communicate it as disciplined innovation.`,
      `The brand upside outweighs the operational noise. Customers reward decisiveness, and competitors are already circling.`,
      `I support approval. Hesitation reads as weakness in the current market cycle.`,
    ],
    PR: [
      `Reputationally, the optics depend entirely on framing. A clumsy rollout of "${m}" invites scrutiny.`,
      `We need a communications runway. With the right messaging, this is defensible; without it, we're exposed.`,
      `I'm cautiously supportive — conditional approval lets us control the story instead of reacting to it.`,
    ],
  }
}

function baseRisk(cfg: DebateConfig, role: AgentRole): number {
  const macro =
    cfg.dti * 0.6 + (100 - (cfg.creditScore - 300) / 5.5) * 0.5 + cfg.defaultProbability * 1.4
  const bias: Record<AgentRole, number> = {
    CEO: -6,
    CFO: 14,
    LEGAL: 9,
    CMO: -12,
    PR: 4,
  }
  return clamp(macro * 0.5 + bias[role] + 30)
}

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms))
}

// -------------------------------------------------------------------------

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url)
  const cfg: DebateConfig = {
    motion:
      searchParams.get("motion") ||
      "Approve a $4.2M growth-capital facility for the enterprise expansion plan.",
    dti: Number(searchParams.get("dti") ?? 38),
    creditScore: Number(searchParams.get("creditScore") ?? 712),
    defaultProbability: Number(searchParams.get("defaultProbability") ?? 6.5),
  }

  const encoder = new TextEncoder()

  const stream = new ReadableStream({
    async start(controller) {
      let closed = false
      const send = (event: DebateEvent) => {
        if (closed) return
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`))
      }

      req.signal.addEventListener("abort", () => {
        closed = true
        try {
          controller.close()
        } catch {
          /* already closed */
        }
      })

      const lines = buildLines(cfg)
      const credibility: Record<AgentRole, number> = {
        CEO: 78,
        CFO: 74,
        LEGAL: 70,
        CMO: 66,
        PR: 62,
      }
      const oversight: OversightState = {
        supervisionScore: 50,
        overrideStreak: 0,
        ceoGeneration: 1,
        dominantPattern: "Balanced Synthesis",
      }

      send({ type: "session_start", config: cfg })
      await sleep(600)

      const rounds = 3
      const lastRisk: Record<AgentRole, number> = {} as Record<AgentRole, number>

      for (let round = 0; round < rounds; round++) {
        for (const role of AGENT_ORDER) {
          if (closed) return

          const drift = (Math.random() - 0.5) * 10 - round * 4
          const risk = clamp(baseRisk(cfg, role) + drift)
          const confidence = clamp(58 + round * 9 + (Math.random() - 0.4) * 16)
          const decision = decisionFromRisk(risk)
          lastRisk[role] = risk

          const message: DebateMessage = {
            id: `${round}-${role}-${Date.now()}`,
            role,
            text: lines[role][round] ?? lines[role][lines[role].length - 1],
            prediction: { riskScore: Math.round(risk), confidence: Math.round(confidence), decision },
            round: round + 1,
            timestamp: Date.now(),
          }
          send({ type: "message", message })

          // Update credibility: rewards confident, decisive agents.
          const delta = (confidence - 60) / 12 - (decision === "CONDITIONAL" ? 0.4 : 0)
          credibility[role] = clamp(credibility[role] + delta)

          // CEO overriding cautious finance feeds the oversight metrics.
          if (role === "CEO" && round > 0) {
            oversight.overrideStreak += 1
            oversight.supervisionScore = clamp(oversight.supervisionScore + 7)
            oversight.ceoGeneration += 1
            oversight.dominantPattern =
              oversight.overrideStreak > 1 ? "Risk-Tolerant Override" : "Balanced Synthesis"
          } else {
            oversight.supervisionScore = clamp(oversight.supervisionScore + (Math.random() - 0.5) * 4)
          }

          const scores: CredibilityScore[] = AGENT_ORDER.map((r) => ({
            role: r,
            score: Math.round(credibility[r]),
          })).sort((a, b) => b.score - a.score)

          send({ type: "credibility", scores })
          send({ type: "oversight", state: { ...oversight, supervisionScore: Math.round(oversight.supervisionScore) } })

          await sleep(1300)
        }
      }

      if (closed) return

      // --- Weighted consensus ---------------------------------------------
      const totalWeight = AGENT_ORDER.reduce((s, r) => s + credibility[r], 0)
      const aggregateRisk = clamp(
        AGENT_ORDER.reduce((s, r) => s + lastRisk[r] * credibility[r], 0) / totalWeight,
      )
      const councilConfidence = clamp(100 - Math.abs(aggregateRisk - 50) * 0.4 + 24)

      let verdict: Verdict
      if (aggregateRisk < 40) verdict = "APPROVE"
      else if (aggregateRisk > 60) verdict = "REJECT"
      else verdict = "CONDITIONAL APPROVE"

      const outcome: ConsensusOutcome = {
        verdict,
        councilConfidence: Math.round(councilConfidence),
        aggregateRisk: Math.round(aggregateRisk),
        actionPlan: [
          { step: "Stage capital release behind quarterly performance milestones.", owner: "CFO" },
          { step: "Execute signed warranties and indemnity agreements before disbursement.", owner: "LEGAL" },
          { step: "Launch a disciplined-innovation narrative across owned channels.", owner: "CMO" },
          { step: "Prepare a communications runway and holding statements.", owner: "PR" },
          { step: "Review outcomes at the next council session and adjust exposure.", owner: "CEO" },
        ],
        closingStatement:
          verdict === "REJECT"
            ? "The weighted risk exceeds our tolerance band. We decline now, but invite a revised proposal with stronger downside protection."
            : verdict === "APPROVE"
              ? "The council aligns on a clear path forward. We proceed with confidence and the agreed guardrails firmly in place."
              : "We move forward — conditionally. Approval is granted only against the milestones, warranties, and messaging the council has set out.",
      }

      send({ type: "consensus", outcome })
      await sleep(300)
      send({ type: "session_end" })
      closed = true
      try {
        controller.close()
      } catch {
        /* noop */
      }
    },
  })

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  })
}
