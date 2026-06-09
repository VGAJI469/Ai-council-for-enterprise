"use client"

import { motion } from "framer-motion"
import { Gauge } from "lucide-react"
import { AGENTS, agentColor, type CredibilityScore } from "@/lib/council"
import { ProgressMeter } from "@/components/progress-meter"

export function CredibilityPanel({ scores }: { scores: CredibilityScore[] }) {
  return (
    <section className="glass rounded-2xl p-5">
      <div className="mb-4 flex items-center gap-2">
        <Gauge className="size-4 text-primary" />
        <h2 className="text-sm font-semibold tracking-tight">Agent Credibility</h2>
      </div>
      <ul className="flex flex-col gap-3.5">
        {scores.map((s) => {
          const cfg = AGENTS[s.role]
          const color = agentColor(s.role)
          return (
            <motion.li key={s.role} layout transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}>
              <div className="mb-1.5 flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span
                    className="flex size-6 items-center justify-center rounded-md text-[9px] font-bold"
                    style={{
                      background: `color-mix(in oklch, ${color} 22%, transparent)`,
                      color,
                    }}
                  >
                    {cfg.initials}
                  </span>
                  <span className="text-xs font-medium text-foreground/90">{cfg.role}</span>
                </div>
                <span className="font-mono text-xs font-semibold tabular-nums" style={{ color }}>
                  {s.score}%
                </span>
              </div>
              <ProgressMeter value={s.score} color={color} />
            </motion.li>
          )
        })}
      </ul>
    </section>
  )
}
