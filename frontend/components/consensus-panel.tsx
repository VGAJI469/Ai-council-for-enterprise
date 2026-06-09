"use client"

import { AnimatePresence, motion } from "framer-motion"
import { Award, CheckCircle2, X } from "lucide-react"
import {
  AGENTS,
  agentColor,
  type ConsensusOutcome,
  decisionColor,
} from "@/lib/council"
import { CircularGauge } from "@/components/circular-gauge"
import { ProgressMeter } from "@/components/progress-meter"

interface ConsensusPanelProps {
  outcome: ConsensusOutcome | null
  open: boolean
  onClose: () => void
}

export function ConsensusPanel({ outcome, open, onClose }: ConsensusPanelProps) {
  return (
    <AnimatePresence>
      {open && outcome ? (
        <>
          <motion.div
            key="backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 z-40 bg-[oklch(0.12_0.04_274_/_0.7)] backdrop-blur-sm"
          />
          <motion.aside
            key="panel"
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 30, stiffness: 280 }}
            className="glass fixed right-0 top-0 z-50 flex h-full w-full max-w-md flex-col overflow-y-auto rounded-l-2xl p-6"
            role="dialog"
            aria-label="Consensus outcome"
          >
            <ConsensusContent outcome={outcome} onClose={onClose} />
          </motion.aside>
        </>
      ) : null}
    </AnimatePresence>
  )
}

function ConsensusContent({
  outcome,
  onClose,
}: {
  outcome: ConsensusOutcome
  onClose: () => void
}) {
  const vColor = decisionColor(outcome.verdict)
  return (
    <>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Award className="size-5 text-primary" />
          <h2 className="text-base font-semibold tracking-tight">Consensus &amp; Outcome</h2>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close outcome panel"
          className="flex size-8 items-center justify-center rounded-lg border border-border bg-secondary text-muted-foreground transition hover:text-foreground"
        >
          <X className="size-4" />
        </button>
      </div>

      <div
        className="mt-5 rounded-2xl border p-5 text-center"
        style={{
          borderColor: `color-mix(in oklch, ${vColor} 50%, transparent)`,
          background: `color-mix(in oklch, ${vColor} 12%, transparent)`,
        }}
      >
        <p className="text-[10px] font-medium uppercase tracking-[0.2em] text-muted-foreground">
          Weighted Verdict
        </p>
        <p className="mt-2 text-2xl font-bold tracking-tight" style={{ color: vColor }}>
          {outcome.verdict}
        </p>
      </div>

      <div className="mt-5 flex items-center justify-between gap-4 rounded-2xl border border-border bg-[oklch(0.18_0.04_274_/_0.5)] p-4">
        <CircularGauge
          value={outcome.councilConfidence}
          size={104}
          color="var(--accent)"
          label="Council"
          sublabel="confidence"
        />
        <div className="flex-1">
          <p className="mb-1.5 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
            Aggregate Risk Score
          </p>
          <div className="flex items-center gap-2">
            <span className="font-mono text-xl font-semibold tabular-nums">
              {outcome.aggregateRisk}
            </span>
            <span className="text-xs text-muted-foreground">/ 100</span>
          </div>
          <div className="mt-2">
            <ProgressMeter value={outcome.aggregateRisk} color="var(--agent-ceo)" height={10} />
          </div>
        </div>
      </div>

      <div className="mt-6">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          CEO Closing Action Plan
        </h3>
        <ol className="mt-3 flex flex-col gap-2.5">
          {outcome.actionPlan.map((item, i) => {
            const color = agentColor(item.owner)
            return (
              <li
                key={i}
                className="flex gap-3 rounded-xl border border-border bg-[oklch(0.18_0.04_274_/_0.5)] p-3"
              >
                <span
                  className="flex size-6 shrink-0 items-center justify-center rounded-full font-mono text-xs font-bold"
                  style={{
                    background: `color-mix(in oklch, ${color} 22%, transparent)`,
                    color,
                  }}
                >
                  {i + 1}
                </span>
                <div className="min-w-0">
                  <p className="text-pretty text-sm leading-relaxed text-foreground/90">
                    {item.step}
                  </p>
                  <span className="mt-1 inline-block text-[10px] font-medium uppercase tracking-wider" style={{ color }}>
                    Owner: {AGENTS[item.owner].title}
                  </span>
                </div>
              </li>
            )
          })}
        </ol>
      </div>

      <div className="mt-6 rounded-2xl border border-border bg-[oklch(0.18_0.04_274_/_0.5)] p-4">
        <div className="mb-2 flex items-center gap-2">
          <CheckCircle2 className="size-4 text-primary" />
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Closing Statement
          </h3>
        </div>
        <p className="text-pretty text-sm italic leading-relaxed text-foreground/90">
          &ldquo;{outcome.closingStatement}&rdquo;
        </p>
      </div>
    </>
  )
}
