import { type AgentPrediction, type Decision, decisionColor } from "@/lib/council"
import { CheckCircle2, XCircle, AlertTriangle } from "lucide-react"

function DecisionIcon({ decision }: { decision: Decision }) {
  if (decision === "APPROVE") return <CheckCircle2 className="size-3.5" />
  if (decision === "REJECT") return <XCircle className="size-3.5" />
  return <AlertTriangle className="size-3.5" />
}

function MiniBar({ value, color }: { value: number; color: string }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-[oklch(0.7_0.08_280_/_0.14)]">
      <div
        className="h-full rounded-full"
        style={{ width: `${value}%`, background: color, transition: "width 0.6s ease" }}
      />
    </div>
  )
}

export function PredictionBadges({ prediction }: { prediction: AgentPrediction }) {
  const color = decisionColor(prediction.decision)
  return (
    <div className="mt-3 grid grid-cols-3 gap-2 rounded-lg border border-border bg-[oklch(0.18_0.04_274_/_0.5)] p-2.5">
      <div className="flex flex-col gap-1">
        <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
          Risk
        </span>
        <span className="font-mono text-sm font-semibold tabular-nums">{prediction.riskScore}</span>
        <MiniBar value={prediction.riskScore} color="var(--agent-ceo)" />
      </div>
      <div className="flex flex-col gap-1">
        <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
          Confidence
        </span>
        <span className="font-mono text-sm font-semibold tabular-nums">{prediction.confidence}</span>
        <MiniBar value={prediction.confidence} color="var(--accent)" />
      </div>
      <div className="flex flex-col items-start gap-1">
        <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
          Decision
        </span>
        <span
          className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] font-semibold"
          style={{ color, background: `color-mix(in oklch, ${color} 18%, transparent)` }}
        >
          <DecisionIcon decision={prediction.decision} />
          {prediction.decision}
        </span>
      </div>
    </div>
  )
}
