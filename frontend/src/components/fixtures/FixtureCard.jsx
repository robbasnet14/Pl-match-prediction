function FixtureCard({ fixture }) {
  return (
    <article className="card-hover rounded-xl border border-white/10 bg-white/5 p-4">
      <div className="flex items-center justify-between text-xs text-slate-400">
        <span>{fixture.time}</span>
        <span>{fixture.venue}</span>
      </div>
      <div className="mt-3 flex items-center justify-between gap-2">
        <p className="font-semibold text-slate-100">{fixture.home}</p>
        <span className="text-slate-500">vs</span>
        <p className="font-semibold text-slate-100">{fixture.away}</p>
      </div>
    </article>
  )
}

export default FixtureCard
