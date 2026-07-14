"use client"

import { useState } from "react"
import { Building2, Play, Circle } from "lucide-react"
import type { DebateConfig } from "@/lib/council"

interface CouncilHeaderProps {
  status: "idle" | "live"
  config: DebateConfig
  onStart: (config: DebateConfig) => void
  disabled: boolean
}

export function CouncilHeader({ status, config, onStart, disabled }: CouncilHeaderProps) {
  const [motion, setMotion] = useState(config.motion)
  const [dti, setDti] = useState(config.dti)
  const [creditScore, setCreditScore] = useState(config.creditScore)
  const [defaultProbability, setDefaultProbability] = useState(config.defaultProbability)

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    onStart({ motion, dti, creditScore, defaultProbability })
  }

  return (
    <header className="glass rounded-2xl p-5 md:p-6">
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div className="flex items-center gap-3">
          <div className="flex size-11 items-center justify-center rounded-xl bg-primary/15 text-primary ring-1 ring-primary/30">
            <Building2 className="size-6" />
          </div>
          <div>
            <h1 className="text-pretty text-2xl font-bold tracking-tight text-primary md:text-3xl" style={{ fontFamily: 'var(--font-dm-sans)' }}>
              Executive Council Dashboard
            </h1>
            <p className="mt-1 text-xs text-muted-foreground">
              Decision deliberation &middot; weighted consensus analysis
            </p>
          </div>
        </div>

        <div
          className="inline-flex items-center gap-2 self-start rounded-full border border-border px-3 py-1.5 text-xs font-medium md:self-auto"
          style={{
            background:
              status === "live"
                ? "color-mix(in oklch, var(--agent-ceo) 16%, transparent)"
                : "var(--secondary)",
          }}
        >
          {status === "live" ? (
            <>
              <span
                className="size-2 rounded-full bg-[var(--agent-ceo)]"
                style={{ animation: "pulse-dot 1.1s ease-in-out infinite" }}
              />
              <Circle className="size-3.5 text-[var(--agent-ceo)] fill-[var(--agent-ceo)]" />
              <span className="text-[var(--agent-ceo)]">Live Streaming</span>
            </>
          ) : (
            <>
              <span className="size-2 rounded-full bg-muted-foreground" />
              <span className="text-muted-foreground">Idle</span>
            </>
          )}
        </div>
      </div>

      <form onSubmit={handleSubmit} className="mt-5 grid gap-4 lg:grid-cols-[1fr_auto]">
        <div className="flex flex-col gap-2">
          <label htmlFor="motion" className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Decision Motion
          </label>
          <textarea
            id="motion"
            value={motion}
            onChange={(e) => setMotion(e.target.value)}
            rows={2}
            disabled={disabled}
            placeholder="Enter the motion for council deliberation..."
            className="w-full resize-none rounded-xl border-2 border-primary/30 bg-white px-4 py-3 text-sm leading-relaxed outline-none transition placeholder:text-muted-foreground focus:border-primary focus:ring-2 focus:ring-primary/20 disabled:opacity-60"
          />
        </div>

        <div className="flex flex-col gap-3">
          <div className="grid grid-cols-3 gap-3">
            <ParamInput
              label="DTI %"
              value={dti}
              onChange={setDti}
              min={0}
              max={100}
              step={0.5}
              disabled={disabled}
            />
            <ParamInput
              label="Credit Score"
              value={creditScore}
              onChange={setCreditScore}
              min={300}
              max={850}
              step={1}
              disabled={disabled}
            />
            <ParamInput
              label="Default Prob %"
              value={defaultProbability}
              onChange={setDefaultProbability}
              min={0}
              max={100}
              step={0.1}
              disabled={disabled}
            />
          </div>
          <button
            type="submit"
            disabled={disabled}
            className="inline-flex items-center justify-center gap-2 rounded-xl bg-primary px-5 py-2.5 text-sm font-semibold text-primary-foreground shadow-[0_0_24px_-4px_var(--primary)] transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Play className="size-4 fill-current" />
            {disabled ? "Session in progress..." : "Begin Council Deliberation"}
          </button>
        </div>
      </form>
    </header>
  )
}

function ParamInput({
  label,
  value,
  onChange,
  min,
  max,
  step,
  disabled,
}: {
  label: string
  value: number
  onChange: (n: number) => void
  min: number
  max: number
  step: number
  disabled: boolean
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
        {label}
      </label>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full rounded-xl border-2 border-primary/30 bg-white px-2.5 py-2 font-mono text-sm tabular-nums outline-none transition focus:border-primary focus:ring-2 focus:ring-primary/20 disabled:opacity-60"
      />
    </div>
  )
}
