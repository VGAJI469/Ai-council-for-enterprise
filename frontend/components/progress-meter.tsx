interface ProgressMeterProps {
  value: number // 0-100
  color?: string
  className?: string
  height?: number
}

export function ProgressMeter({
  value,
  color = "var(--primary)",
  className,
  height = 8,
}: ProgressMeterProps) {
  const v = Math.max(0, Math.min(100, value))
  return (
    <div
      className={`w-full overflow-hidden rounded-full bg-[oklch(0.7_0.08_280_/_0.14)] ${className ?? ""}`}
      style={{ height }}
      role="progressbar"
      aria-valuenow={Math.round(v)}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div
        className="h-full rounded-full"
        style={{
          width: `${v}%`,
          background: color,
          boxShadow: `0 0 10px ${color}`,
          transition: "width 0.7s cubic-bezier(0.22, 1, 0.36, 1)",
        }}
      />
    </div>
  )
}
