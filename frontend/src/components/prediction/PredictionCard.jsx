import ProbabilityBar from './ProbabilityBar'
import ScorePrediction from './ScorePrediction'

function toPercent(value, fallback = 0) {
  const num = Number(value)
  if (!Number.isFinite(num)) return fallback
  return Math.max(0, Math.min(100, Math.round(num)))
}

function toGoal(value, fallback = 0) {
  const num = Number(value)
  if (!Number.isFinite(num)) return fallback
  return Math.max(0, num)
}

function PredictionCard({ prediction }) {
  const home = prediction?.home || 'Home'
  const away = prediction?.away || 'Away'
  const fixture = prediction?.fixture || `${home} vs ${away}`
  const confidence = toPercent(prediction?.confidence, 0)
  const probabilities = prediction?.probabilities || {}
  const homeProb = toPercent(probabilities.home, 34)
  const drawProb = toPercent(probabilities.draw, 22)
  const awayProb = toPercent(probabilities.away, 34)
  const expectedGoals = prediction?.expectedGoals || {}
  const homeXg = toGoal(expectedGoals.home, null)
  const awayXg = toGoal(expectedGoals.away, null)
  const totalXg = toGoal(expectedGoals.total, null)
  const topScorelines = Array.isArray(prediction?.topScorelines) ? prediction.topScorelines.slice(0, 6) : []
  const goalOutlook = prediction?.goalOutlook || {}
  const goalMarkets = prediction?.goalMarkets || {}
  const total3Plus = Number(goalOutlook.total3PlusPct)
  const total4Plus = Number(goalOutlook.total4PlusPct)
  const btts = Number(goalOutlook.bttsPct)
  const home3Plus = Number(goalOutlook.home3PlusPct)
  const away3Plus = Number(goalOutlook.away3PlusPct)
  const over25 = Number(goalMarkets.over25Pct)
  const under25 = Number(goalMarkets.under25Pct)
  const over35 = Number(goalMarkets.over35Pct)
  const under35 = Number(goalMarkets.under35Pct)
  const marketBtts = Number(goalMarkets.bttsPct)
  const noBtts = Number(goalMarkets.noBttsPct)
  const coverage = prediction?.dataCoverage || {}
  const homeMatchesInTraining = Number(coverage.homeMatchesInTraining)
  const awayMatchesInTraining = Number(coverage.awayMatchesInTraining)
  const usedFallbackProbabilities = Boolean(coverage.usedFallbackProbabilities)
  const lowCoverage =
    Number.isFinite(homeMatchesInTraining) &&
    Number.isFinite(awayMatchesInTraining) &&
    (homeMatchesInTraining < 20 || awayMatchesInTraining < 20)

  return (
    <article className="card-hover rounded-xl border border-white/10 bg-white/5 p-4">
      <p className="text-xs uppercase tracking-[0.14em] text-slate-400">{fixture}</p>
      <h4 className="mt-2 text-sm font-semibold text-white">Confidence {confidence}%</h4>

      <div className="mt-4 space-y-3">
        <ProbabilityBar label="Home Win" value={homeProb} color="#22c55e" />
        <ProbabilityBar label="Draw" value={drawProb} color="#f59e0b" />
        <ProbabilityBar label="Away Win" value={awayProb} color="#38bdf8" />
      </div>

      <div className="mt-4">
        <ScorePrediction home={home} away={away} score={prediction?.score || '0-0'} />
      </div>

      {(usedFallbackProbabilities || lowCoverage) && (
        <div className="mt-4 rounded-xl border border-amber-300/30 bg-amber-300/10 px-4 py-3 text-xs text-amber-100">
          Low data coverage for one or both teams. Prediction may be less reliable.
        </div>
      )}

      <div className="mt-4 rounded-xl border border-cyan-400/20 bg-cyan-400/5 px-4 py-3">
        <p className="text-slate-400">Expected Goals (xG)</p>
        {homeXg == null || awayXg == null || totalXg == null ? (
          <p className="mt-1 text-sm text-slate-300">Not available</p>
        ) : (
          <div className="mt-2 grid grid-cols-3 gap-2 text-xs text-slate-200">
            <p>
              <span className="text-slate-400">Home</span>{' '}
              <span className="font-semibold text-white">{homeXg.toFixed(2)}</span>
            </p>
            <p>
              <span className="text-slate-400">Away</span>{' '}
              <span className="font-semibold text-white">{awayXg.toFixed(2)}</span>
            </p>
            <p>
              <span className="text-slate-400">Total</span>{' '}
              <span className="font-semibold text-white">{totalXg.toFixed(2)}</span>
            </p>
          </div>
        )}
      </div>

      <div className="mt-4 rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3">
        <p className="text-slate-400">Goal Outlook</p>
        {Number.isFinite(total3Plus) ? (
          <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-slate-200">
            <p>
              <span className="text-slate-400">3+ total goals</span>{' '}
              <span className="font-semibold text-white">{total3Plus.toFixed(1)}%</span>
            </p>
            <p>
              <span className="text-slate-400">4+ total goals</span>{' '}
              <span className="font-semibold text-white">{total4Plus.toFixed(1)}%</span>
            </p>
            <p>
              <span className="text-slate-400">BTTS</span>{' '}
              <span className="font-semibold text-white">{btts.toFixed(1)}%</span>
            </p>
            <p>
              <span className="text-slate-400">Home 3+ / Away 3+</span>{' '}
              <span className="font-semibold text-white">
                {home3Plus.toFixed(1)}% / {away3Plus.toFixed(1)}%
              </span>
            </p>
          </div>
        ) : (
          <p className="mt-1 text-sm text-slate-300">Not available</p>
        )}
      </div>

      <div className="mt-4 rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3">
        <p className="text-slate-400">Goal Markets</p>
        {Number.isFinite(over25) ? (
          <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-slate-200">
            <p>
              <span className="text-slate-400">Over 2.5 / Under 2.5</span>{' '}
              <span className="font-semibold text-white">
                {over25.toFixed(1)}% / {under25.toFixed(1)}%
              </span>
            </p>
            <p>
              <span className="text-slate-400">Over 3.5 / Under 3.5</span>{' '}
              <span className="font-semibold text-white">
                {over35.toFixed(1)}% / {under35.toFixed(1)}%
              </span>
            </p>
            <p>
              <span className="text-slate-400">BTTS / No BTTS</span>{' '}
              <span className="font-semibold text-white">
                {marketBtts.toFixed(1)}% / {noBtts.toFixed(1)}%
              </span>
            </p>
          </div>
        ) : (
          <p className="mt-1 text-sm text-slate-300">Not available</p>
        )}
      </div>

      <div className="mt-4 rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3">
        <p className="text-slate-400">Top Scorelines</p>
        {topScorelines.length === 0 ? (
          <p className="mt-1 text-sm text-slate-300">Not available</p>
        ) : (
          <div className="mt-2 space-y-1 text-sm text-slate-200">
            {topScorelines.map((row, idx) => (
              <div key={`${row.score || 'score'}-${idx}`} className="flex items-center justify-between">
                <span className="font-medium text-white">{row.score || 'N/A'}</span>
                <span className="text-slate-300">
                  {Number.isFinite(Number(row.probabilityPct)) ? Number(row.probabilityPct).toFixed(1) : '0.0'}%
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </article>
  )
}

export default PredictionCard
