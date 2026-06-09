"use client"

import { useEffect, useRef } from "react"
import { AnimatePresence, motion } from "framer-motion"
import { Terminal } from "lucide-react"
import { AGENTS, agentColor, type DebateMessage } from "@/lib/council"
import { PredictionBadges } from "@/components/prediction-badges"

function AgentAvatar({ role }: { role: DebateMessage["role"] }) {
  const cfg = AGENTS[role]
  const color = agentColor(role)
  return (
    <div
      className="flex size-10 shrink-0 items-center justify-center rounded-full text-xs font-bold tracking-tight"
      style={{
        background: `color-mix(in oklch, ${color} 22%, transparent)`,
        color,
        boxShadow: `0 0 0 1px color-mix(in oklch, ${color} 45%, transparent), 0 0 14px -2px ${color}`,
      }}
    >
      {cfg.initials}
    </div>
  )
}

function MessageBubble({ message }: { message: DebateMessage }) {
  const cfg = AGENTS[message.role]
  const color = agentColor(message.role)
  return (
    <motion.li
      layout
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
      className="flex gap-3"
    >
      <AgentAvatar role={message.role} />
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm font-semibold" style={{ color }}>
            {cfg.title}
          </span>
          <span className="rounded-md border border-border bg-secondary px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
            {cfg.model}
          </span>
          <span className="ml-auto font-mono text-[10px] text-muted-foreground">
            R{message.round} &middot;{" "}
            {new Date(message.timestamp).toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
              second: "2-digit",
            })}
          </span>
        </div>
        <div
          className="mt-1.5 rounded-xl rounded-tl-sm border border-border bg-[oklch(0.21_0.04_274_/_0.55)] px-3.5 py-2.5"
          style={{ borderLeft: `2px solid ${color}` }}
        >
          <p className="text-pretty text-sm leading-relaxed text-foreground/90">{message.text}</p>
          <PredictionBadges prediction={message.prediction} />
        </div>
      </div>
    </motion.li>
  )
}

interface DebateBoardProps {
  messages: DebateMessage[]
  status: "idle" | "live"
}

export function DebateBoard({ messages, status }: DebateBoardProps) {
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" })
  }, [messages.length])

  return (
    <section className="glass flex min-h-0 flex-col rounded-2xl">
      <div className="flex items-center gap-2 border-b border-border px-5 py-3.5">
        <Terminal className="size-4 text-primary" />
        <h2 className="text-sm font-semibold tracking-tight">Live Debate Board</h2>
        <span className="ml-auto font-mono text-[11px] text-muted-foreground">
          {messages.length} statement{messages.length === 1 ? "" : "s"}
        </span>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
        {messages.length === 0 ? (
          <div className="flex h-full min-h-64 flex-col items-center justify-center text-center">
            <div className="flex size-14 items-center justify-center rounded-2xl bg-primary/10 text-primary ring-1 ring-primary/20">
              <Terminal className="size-7" />
            </div>
            <p className="mt-4 text-sm font-medium">
              {status === "live" ? "Convening the council..." : "No active session"}
            </p>
            <p className="mt-1 max-w-xs text-xs text-muted-foreground">
              Submit a motion above and start a debate session to watch the executives deliberate in
              real time.
            </p>
          </div>
        ) : (
          <ul className="flex flex-col gap-5">
            <AnimatePresence initial={false}>
              {messages.map((m) => (
                <MessageBubble key={m.id} message={m} />
              ))}
            </AnimatePresence>
            <div ref={endRef} />
          </ul>
        )}
      </div>
    </section>
  )
}
