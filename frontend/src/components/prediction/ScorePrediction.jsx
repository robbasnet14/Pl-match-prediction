function ScorePrediction({ home, away, score }) {
  return (
    <div className="rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-sm">
      <p className="text-slate-400">Likely Score</p>
      <p className="mt-1 font-semibold text-slate-100">
        {home} {score} {away}
      </p>
    </div>
  )
}

export default ScorePrediction
