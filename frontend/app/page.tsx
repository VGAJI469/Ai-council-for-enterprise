"use client"

import { useCallback, useRef, useState } from "react"
import { Award } from "lucide-react"
import {
  AGENT_ORDER,
  type ConsensusOutcome,
  type CredibilityScore,
  type DebateConfig,
  type DebateEvent,
  type DebateMessage,
  type OversightState,
} from "@/lib/council"
import { CouncilHeader } from "@/components/council-header"
import { DebateBoard } from "@/components/debate-board"
import { CredibilityPanel } from "@/components/credibility-panel"
import { OversightPanel } from "@/components/oversight-panel"
import { ConsensusPanel } from "@/components/consensus-panel"

const INITIAL_CONFIG: DebateConfig = {
  motion: "Approve a $4.2M growth-capital facility for the enterprise expansion plan.",
  dti: 38,
  creditScore: 712,
  defaultProbability: 6.5,
}

const INITIAL_CREDIBILITY: CredibilityScore[] = AGENT_ORDER.map((role) => ({ role, score: 70 }))

const INITIAL_OVERSIGHT: OversightState = {
  supervisionScore: 50,
  overrideStreak: 0,
  ceoGeneration: 1,
  dominantPattern: "Balanced Synthesis",
}

export default function Page() {
  const [status, setStatus] = useState<"idle" | "live">("idle")
  const [config, setConfig] = useState<DebateConfig>(INITIAL_CONFIG)
  const [messages, setMessages] = useState<DebateMessage[]>([])
  const [credibility, setCredibility] = useState<CredibilityScore[]>(INITIAL_CREDIBILITY)
  const [oversight, setOversight] = useState<OversightState>(INITIAL_OVERSIGHT)
  const [outcome, setOutcome] = useState<ConsensusOutcome | null>(null)
  const [showOutcome, setShowOutcome] = useState(false)
  const esRef = useRef<EventSource | null>(null)

  const startDebate = useCallback((cfg: DebateConfig) => {
    esRef.current?.close()
    setConfig(cfg)
    setMessages([])
    setOutcome(null)
    setShowOutcome(false)
    setOversight(INITIAL_OVERSIGHT)
    setStatus("live")

    const params = new URLSearchParams({
      motion: cfg.motion,
      dti: String(cfg.dti),
      creditScore: String(cfg.creditScore),
      defaultProbability: String(cfg.defaultProbability),
    })
    const es = new EventSource(`/api/debate?${params.toString()}`)
    esRef.current = es

    es.onmessage = (e) => {
      const event = JSON.parse(e.data) as DebateEvent
      switch (event.type) {
        case "message":
          setMessages((prev) => [...prev, event.message])
          break
        case "credibility":
          setCredibility(event.scores)
          break
        case "oversight":
          setOversight(event.state)
          break
        case "consensus":
          setOutcome(event.outcome)
          setShowOutcome(true)
          break
        case "session_end":
          setStatus("idle")
          es.close()
          esRef.current = null
          break
      }
    }

    es.onerror = () => {
      setStatus("idle")
      es.close()
      esRef.current = null
    }
  }, [])

  const resetOversight = useCallback(() => {
    setOversight(INITIAL_OVERSIGHT)
  }, [])

  return (
    <main className="mx-auto flex min-h-screen max-w-[1400px] flex-col gap-5 p-4 md:p-6 lg:h-screen">
      <CouncilHeader
        status={status}
        config={config}
        onStart={startDebate}
        disabled={status === "live"}
      />

      <div className="grid min-h-0 flex-1 gap-5 lg:grid-cols-[1fr_340px]">
        <div className="flex min-h-[60vh] flex-col lg:min-h-0">
          <DebateBoard messages={messages} status={status} />
        </div>

        <aside className="flex flex-col gap-5 overflow-y-auto">
          <CredibilityPanel scores={credibility} />
          <OversightPanel state={oversight} onReset={resetOversight} disabled={status === "live"} />
          {outcome ? (
            <button
              type="button"
              onClick={() => setShowOutcome(true)}
              className="glass inline-flex items-center justify-center gap-2 rounded-2xl px-4 py-3 text-sm font-semibold text-primary ring-1 ring-primary/30 transition hover:bg-primary/10"
            >
              <Award className="size-4" />
              View Consensus Outcome
            </button>
          ) : null}
        </aside>
      </div>

      <ConsensusPanel
        outcome={outcome}
        open={showOutcome}
        onClose={() => setShowOutcome(false)}
      />
    </main>
  )
}
