"use client"

import { Eye, Flame, GitBranch, RotateCcw, Zap } from "lucide-react"
import type { OversightState } from "@/lib/council"
import { CircularGauge } from "@/components/circular-gauge"

interface OversightPanelProps {
  state: OversightState
  onReset: () => void
  disabled: boolean
}

function StatCard({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode
  label: string
  value: string | number
}) {
  return (
    <div className="rounded-xl border border-border bg-[oklch(0.18_0.04_274_/_0.5)] p-3">
      <div className="flex items-center gap-1.5 text-muted-foreground">
        {icon}
        <span className="text-[10px] font-medium uppercase tracking-wider">{label}</span>
      </div>
      <p className="mt-1.5 truncate text-sm font-semibold tabular-nums text-foreground">{value}</p>
    </div>
  )
}

export function OversightPanel({ state, onReset, disabled }: OversightPanelProps) {
  return (
    <section className="glass rounded-2xl p-5">
      <div className="mb-4 flex items-center gap-2">
        <Eye className="size-4 text-primary" />
        <h2 className="text-sm font-semibold tracking-tight">CEO Supervision &amp; Oversight</h2>
      </div>

      <div className="flex items-center justify-center py-1">
        <CircularGauge
          value={state.supervisionScore}
          size={132}
          color="var(--accent)"
          label="Supervision"
          sublabel="composite score"
        />
      </div>

      <div className="mt-4 grid grid-cols-2 gap-2.5">
        <StatCard
          icon={<Flame className="size-3.5" />}
          label="Override Streak"
          value={`${state.overrideStreak}x`}
        />
        <StatCard
          icon={<Zap className="size-3.5" />}
          label="CEO Generation"
          value={`Gen ${state.ceoGeneration}`}
        />
        <div className="col-span-2">
          <StatCard
            icon={<GitBranch className="size-3.5" />}
            label="Dominant Override Pattern"
            value={state.dominantPattern}
          />
        </div>
      </div>

      <button
        type="button"
        onClick={onReset}
        disabled={disabled}
        className="mt-4 inline-flex w-full items-center justify-center gap-2 rounded-xl border border-border bg-secondary px-4 py-2.5 text-xs font-semibold text-secondary-foreground transition hover:bg-[oklch(0.32_0.05_274_/_0.7)] disabled:cursor-not-allowed disabled:opacity-50"
      >
        <RotateCcw className="size-3.5" />
        Reset Oversight
      </button>
    </section>
  )
}
