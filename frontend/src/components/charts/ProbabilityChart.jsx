function ProbabilityChart({ teams, metric, colorClass, trackClass, valueFormatter }) {
  return (
    <div className="space-y-3">
      {teams.map((team) => {
        const value = Number(team[metric] || 0)
        const width = Math.max(0, Math.min(100, value))

        return (
          <div key={`${metric}-${team.team}`}>
            <div className="mb-1 flex items-center justify-between text-sm">
              <span className="text-slate-200">{team.team}</span>
              <span className="text-slate-300">{valueFormatter(value)}</span>
            </div>
            <div className={`h-2 overflow-hidden rounded-full ${trackClass}`}>
              <div
                className={`h-full rounded-full transition-all duration-700 ${colorClass}`}
                style={{ width: `${width}%` }}
              />
            </div>
          </div>
        )
      })}
    </div>
  )
}

export default ProbabilityChart
