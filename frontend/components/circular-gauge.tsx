"use client"

interface CircularGaugeProps {
  value: number // 0-100
  size?: number
  strokeWidth?: number
  color?: string
  trackColor?: string
  label?: string
  sublabel?: string
  className?: string
}

export function CircularGauge({
  value,
  size = 120,
  strokeWidth = 10,
  color = "var(--primary)",
  trackColor = "oklch(0.7 0.08 280 / 0.16)",
  label,
  sublabel,
  className,
}: CircularGaugeProps) {
  const v = Math.max(0, Math.min(100, value))
  const radius = (size - strokeWidth) / 2
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (v / 100) * circumference

  return (
    <div className={className} style={{ width: size, height: size, position: "relative" }}>
      <svg width={size} height={size} className="-rotate-90" aria-hidden="true">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={trackColor}
          strokeWidth={strokeWidth}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{
            transition: "stroke-dashoffset 0.8s cubic-bezier(0.22, 1, 0.36, 1)",
            filter: `drop-shadow(0 0 6px ${color})`,
          }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="font-mono text-2xl font-semibold tabular-nums text-foreground">
          {Math.round(v)}
          <span className="text-sm text-muted-foreground">%</span>
        </span>
        {label ? (
          <span className="mt-0.5 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
            {label}
          </span>
        ) : null}
        {sublabel ? <span className="text-[10px] text-muted-foreground">{sublabel}</span> : null}
      </div>
    </div>
  )
}
